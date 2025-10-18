# exp3_feature_space_vis.py

import os
import argparse
import random
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ===============================
# 固定随机种子，确保可复现
# ===============================
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===============================
# 1. 图像预处理
# ===============================
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

# ===============================
# 2. ResNet50 特征提取器
# ===============================
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.features = nn.Sequential(
            original_model.conv1,
            original_model.bn1,
            original_model.relu,
            original_model.maxpool,
            original_model.layer1,
            original_model.layer2,
            original_model.layer3,
            original_model.layer4,
            original_model.avgpool
        )
    def forward(self, x):
        x = self.features(x)       # [B,2048,1,1]
        x = torch.flatten(x, 1)    # [B,2048]
        return x

# ===============================
# 3. ViT-B/16 特征提取器（兼容不同版本的属性名）
# ===============================
class ViTFeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        # Patch embedding
        self.patch_embed = original_model.conv_proj

        # 兼容：某些 torchvision 版本里叫 cls_token，另一些版本里可能叫 class_token
        if hasattr(original_model, "cls_token"):
            self.cls_token = original_model.cls_token
        elif hasattr(original_model, "class_token"):
            self.cls_token = original_model.class_token
        else:
            raise AttributeError("未找到 ViT 模型中的 cls_token 或 class_token 属性，请检查 torchvision 版本。")

        # 兼容：positional embedding 可能叫 pos_embedding 或 pos_embed
        if hasattr(original_model.encoder, "pos_embedding"):
            self.pos_embed = original_model.encoder.pos_embedding
        elif hasattr(original_model.encoder, "pos_embed"):
            self.pos_embed = original_model.encoder.pos_embed
        else:
            raise AttributeError("未找到 ViT 模型中的 pos_embedding 或 pos_embed 属性，请检查 torchvision 版本。")

        # EncoderBlock 列表 与 最后 LayerNorm
        self.encoder_layers = original_model.encoder.layers
        self.norm = original_model.encoder.ln

    def forward(self, x):
        B = x.shape[0]
        # x: [B,3,224,224]
        x = self.patch_embed(x)             # [B,768,14,14]
        x = torch.flatten(x, 2)             # [B,768,196]
        x = x.transpose(1, 2)               # [B,196,768]

        # [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B,1,768]
        x = torch.cat((cls_tokens, x), dim=1)          # [B,197,768]

        # 加位置信息
        x = x + self.pos_embed                        # [B,197,768]

        # 逐层通过 EncoderBlock
        for block in self.encoder_layers:
            x = block(x)                              # 每层 EncoderBlock

        # 最终 LayerNorm
        x = self.norm(x)                              # [B,197,768]

        # 返回 [CLS] token 对应的向量 → [B,768]
        return x[:, 0, :]

# ===============================
# 4. 批量提取 ResNet50 特征
# ===============================
def extract_resnet_features(model, image_paths, preprocess, device, batch_size=32):
    model.eval()
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            imgs = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                imgs.append(preprocess(img))
            imgs_tensor = torch.stack(imgs, dim=0).to(device)  # [B,3,224,224]
            feats = model(imgs_tensor)                         # [B,2048]
            all_feats.append(feats.cpu().numpy())
    all_feats = np.concatenate(all_feats, axis=0)            # [N,2048]
    return all_feats

# ===============================
# 5. 批量提取 ViT 特征
# ===============================
def extract_vit_features(model, image_paths, preprocess, device, batch_size=32):
    model.eval()
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            imgs = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                imgs.append(preprocess(img))
            imgs_tensor = torch.stack(imgs, dim=0).to(device)  # [B,3,224,224]
            feats = model(imgs_tensor)                          # [B,768]
            all_feats.append(feats.cpu().numpy())
    all_feats = np.concatenate(all_feats, axis=0)              # [N,768]
    return all_feats

# ===============================
# 6. 收集图像路径与标签
# ===============================
def collect_image_paths(test_root):
    class_names = sorted(os.listdir(test_root))
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    image_paths = []
    labels = []
    for cls in class_names:
        cls_dir = os.path.join(test_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            image_paths.append(os.path.join(cls_dir, fname))
            labels.append(class_to_idx[cls])
    return image_paths, np.array(labels), class_names

# ===============================
# 7. PCA 预处理（可选）
# ===============================
def maybe_apply_pca(features, n_components=50, use_pca=True):
    """
    如果 use_pca=True，就对 features 用 PCA 降到 min(n_components, n_samples-1) 维，返回 [N, min(n_components,n_samples-1)]，
    并打印累计方差；否则直接返回原 features。
    """
    if not use_pca:
        return features

    # N = 特征矩阵的行数（样本数）
    N = features.shape[0]
    # PCA 的最大主成分数最多是 N-1
    target_dim = min(n_components, N - 1)
    print(f"Applying PCA: {features.shape[1]} → {target_dim}  (原样本数 {N})")
    pca = PCA(n_components=target_dim, random_state=42)
    feats_pca = pca.fit_transform(features)
    print(f" - PCA explained variance ratio sum: {pca.explained_variance_ratio_.sum():.4f}")
    return feats_pca

# ===============================
# 8. 运行 t-SNE 并返回 2D 坐标
# ===============================
def run_tsne(features, perplexity=30, learning_rate=200, n_iter=1000):
    print(f"Running t-SNE on data shape {features.shape} ...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        init="pca",
        metric="euclidean",
        random_state=42,
        verbose=1
    )
    X_2d = tsne.fit_transform(features)  # [N,2]
    print(" t-SNE done. Output shape:", X_2d.shape)
    return X_2d

# ===============================
# 9. 绘制并保存 t-SNE 散点图
# ===============================
def plot_tsne(embedding, labels, class_names, title, output_path, figsize=(6,6)):
    plt.figure(figsize=figsize)
    cmap = plt.get_cmap("tab10")
    for cls_idx, cls_name in enumerate(class_names):
        idxs = np.where(labels == cls_idx)[0]
        plt.scatter(embedding[idxs, 0], embedding[idxs, 1],
                    s=8, color=cmap(cls_idx % 10), label=cls_name, alpha=0.7)
    plt.legend(loc="best", markerscale=2, fontsize="small")
    plt.title(title, fontsize=14)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot → {output_path}")

# ===============================
# 10. 主程序
# ===============================
def main():
    parser = argparse.ArgumentParser(description="实验三：t-SNE 对比 ResNet50 vs ViT-B/16 vs 集成")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="测试集目录，子文件夹为各类别，内含图像")
    parser.add_argument("--output_dir", type=str, default="exp3_tsne_plots",
                        help="t-SNE 可视化图保存目录")
    parser.add_argument("--use_pca", type=lambda x: x.lower()=="true", default=True,
                        help="是否先用 PCA 降到 50 维再做 t-SNE (True/False)")
    parser.add_argument("--perplexity", type=float, default=30.0,
                        help="t-SNE perplexity 参数")
    parser.add_argument("--learning_rate", type=float, default=200.0,
                        help="t-SNE learning_rate 参数")
    parser.add_argument("--n_iter", type=int, default=1000,
                        help="t-SNE 迭代次数")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="提取特征时的批大小")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 10.1 收集图像路径与标签
    image_paths, labels, class_names = collect_image_paths(args.test_dir)
    N = len(image_paths)
    print(f"Found {N} images across {len(class_names)} classes: {class_names}")

    # 10.2 构造 ResNet50 特征提取器
    num_classes = len(class_names)
    resnet_full = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    resnet_full.fc = nn.Linear(resnet_full.fc.in_features, num_classes)
    resnet_full.load_state_dict(torch.load("best_model.pth", map_location=device))
    resnet_full = resnet_full.to(device).eval()
    resnet_extractor = ResNetFeatureExtractor(resnet_full).to(device).eval()

    # 10.3 构造 ViT-B/16 特征提取器
    vit_full = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    vit_full.heads = nn.Sequential(nn.Linear(vit_full.heads.head.in_features, num_classes))
    vit_full.load_state_dict(torch.load("vit_model.pth", map_location=device))
    vit_full = vit_full.to(device).eval()
    vit_extractor = ViTFeatureExtractor(vit_full).to(device).eval()

    # 10.4 提取 ResNet50 特征
    print("Extracting ResNet50 features ...")
    resnet_feats = extract_resnet_features(
        model=resnet_extractor,
        image_paths=image_paths,
        preprocess=preprocess,
        device=device,
        batch_size=args.batch_size
    )  # [N,2048]
    print(" ResNet features shape:", resnet_feats.shape)

    # 10.5 提取 ViT 特征
    print("Extracting ViT-B/16 features ...")
    vit_feats = extract_vit_features(
        model=vit_extractor,
        image_paths=image_paths,
        preprocess=preprocess,
        device=device,
        batch_size=args.batch_size
    )  # [N,768]
    print(" ViT features shape:", vit_feats.shape)

    # 10.6 构造拼接后特征
    ensemble_feats = np.concatenate((resnet_feats, vit_feats), axis=1)  # [N,2816]
    print(" Ensemble features shape:", ensemble_feats.shape)

    # 10.7 PCA 降到 50 维（可选）
    resnet_pca    = maybe_apply_pca(resnet_feats,    n_components=50, use_pca=args.use_pca)  # [N,50] 或 [N,2048]
    vit_pca       = maybe_apply_pca(vit_feats,       n_components=50, use_pca=args.use_pca)  # [N,50] 或 [N,768]
    ensemble_pca  = maybe_apply_pca(ensemble_feats,  n_components=50, use_pca=args.use_pca)  # [N,50] 或 [N,2816]

    # 10.8 t-SNE 降到 2D（保持同一组参数以便可比性）
    X_resnet   = run_tsne(resnet_pca,
                          perplexity=args.perplexity,
                          learning_rate=args.learning_rate,
                          n_iter=args.n_iter)   # [N,2]
    X_vit      = run_tsne(vit_pca,
                          perplexity=args.perplexity,
                          learning_rate=args.learning_rate,
                          n_iter=args.n_iter)   # [N,2]
    X_ensemble = run_tsne(ensemble_pca,
                          perplexity=args.perplexity,
                          learning_rate=args.learning_rate,
                          n_iter=args.n_iter)   # [N,2]

    # 10.9 绘图并保存
    plot_tsne(X_resnet,    labels, class_names,
              title="t-SNE of ResNet50 Features",
              output_path=os.path.join(args.output_dir, "resnet_tsne.png"))

    plot_tsne(X_vit,       labels, class_names,
              title="t-SNE of ViT-B/16 Features",
              output_path=os.path.join(args.output_dir, "vit_tsne.png"))

    plot_tsne(X_ensemble,  labels, class_names,
              title="t-SNE of Ensemble (ResNet50 + ViT) Features",
              output_path=os.path.join(args.output_dir, "ensemble_tsne.png"))

    print("All t-SNE visualizations done!")

if __name__ == "__main__":
    main()

