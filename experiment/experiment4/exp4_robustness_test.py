# exp4_robustness_test.py

from matplotlib import rc

# 设置 matplotlib 使用支持中文的字体，如 Microsoft YaHei
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Microsoft YaHei']})
rc('axes', unicode_minus=False)
rc('text', usetex=False)
rc('mathtext', fontset='custom', rm='Microsoft YaHei', it='Microsoft YaHei:italic', bf='Microsoft YaHei:bold')

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models, datasets
from PIL import Image, ImageEnhance
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse

# -----------------------------
# 说明：
# 1. 对验证集（dataset/validation_jpg） 做几种扰动：
#    A. Random Erasing (遮挡)
#    B. Gaussian Noise (加噪声)
#    C. Brightness Change (亮度变化)
# 2. 分别在 ResNet50、ViT-B/16 和它们的概率平均“集成模型” 上测试精度
# 3. 将结果汇总在表格和柱状图中，方便对比
# -----------------------------

def random_erasing(img, p=1.0, sl=0.02, sh=0.2, r1=0.3, mean=[1,1,1]):
    """
    随机遮挡：类似 torchvision.transforms.RandomErasing
    sl: 遮挡区域相对于图像面积的最小比例
    sh: 遮挡区域相对于图像面积的最大比例
    r1: 遮挡区域长宽比范围
    mean: 遮挡填充颜色（默认白色）
    """
    if random.uniform(0,1) > p:
        return img
    img_np = np.array(img).astype(np.uint8)
    H, W, C = img_np.shape
    area = H * W

    for _ in range(100):
        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1/r1)

        h = int(round(np.sqrt(target_area * aspect_ratio)))
        w = int(round(np.sqrt(target_area / aspect_ratio)))
        if h < H and w < W:
            x1 = random.randint(0, H - h)
            y1 = random.randint(0, W - w)
            img_np[x1:x1+h, y1:y1+w, :] = np.array(mean) * 255
            return Image.fromarray(img_np)
    return img

def add_gaussian_noise(img, sigma=0.1):
    """
    给 PIL 图像添加高斯噪声
    sigma: 噪声标准差，图像范围 [0,1]
    """
    img_np = np.array(img).astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma, img_np.shape).astype(np.float32)
    noisy = img_np + noise
    noisy = np.clip(noisy, 0, 1)
    return Image.fromarray((noisy * 255).astype(np.uint8))

def adjust_brightness(img, factor=0.5):
    """
    PIL 图像亮度调整
    factor < 1.0 变暗，> 1.0 变亮
    """
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

def get_perturbed_image(img, mode):
    """
    根据 mode 生成扰动后的图像
    mode: "erasing" / "noise" / "brightness"
    """
    if mode == "erasing":
        # 随机遮挡参数：p=1.0、遮挡面积 5%~10%、长宽比 0.3~(1/0.3)
        return random_erasing(img, p=1.0, sl=0.05, sh=0.1, r1=0.3, mean=[1,1,1])
    elif mode == "noise":
        return add_gaussian_noise(img, sigma=0.1)
    elif mode == "brightness":
        return adjust_brightness(img, factor=random.choice([0.7, 1.3]))
    else:
        # mode == "orig"
        return img


# ================================
# 将 PerturbedDataset 定义在全局作用域
# ================================
class PerturbedDataset(torch.utils.data.Dataset):
    """
    将原始 image + 指定扰动 mode 组合成新的 Dataset。
    这里使用 Subset 传入原始的(图片, label)对，再根据 mode 进行实际扰动。
    """
    def __init__(self, subset, mode, base_transform, normalize_transform):
        """
        subset: torch.utils.data.Subset，内部包含 (PIL.Image, label)
        mode: "orig"/"erasing"/"noise"/"brightness"
        base_transform: e.g. Resize -> CenterCrop
        normalize_transform: ToTensor -> Normalize
        """
        self.subset = subset
        self.mode = mode
        self.base_transform = base_transform
        self.normalize_transform = normalize_transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # subset[idx] 返回 (PIL.Image, label)
        img, label = self.subset[idx]
        img = self.base_transform(img)
        if self.mode != "orig":
            img = get_perturbed_image(img, self.mode)
        img = self.normalize_transform(img)
        return img, label


def evaluate_model(model, dataloader, device):
    """
    在给定 dataloader 上测试 model 精度
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total * 100.0


def evaluate_ensemble(resnet, vit, dataloader, device):
    """
    对给定 dataloader 评估“集成模型”准确率：
    先分别计算 ResNet50 和 ViT 的 softmax 概率，然后相加平均，再取 argmax。
    """
    resnet.eval()
    vit.eval()
    correct = 0
    total = 0
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            # ResNet50 概率
            out_r = resnet(imgs)                  # [B, num_classes]
            prob_r = softmax(out_r)               # [B, num_classes]
            # ViT 概率
            out_v = vit(imgs)                     # [B, num_classes]
            prob_v = softmax(out_v)               # [B, num_classes]
            # 平均
            prob_avg = (prob_r + prob_v) / 2.0    # [B, num_classes]
            preds = prob_avg.argmax(dim=1)        # [B]
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total * 100.0


def main():
    parser = argparse.ArgumentParser(description="实验四：加入集成模型的鲁棒性测试")
    parser.add_argument("--data_dir",   type=str, default="dataset/validation_jpg",
                        help="验证集目录（ImageFolder 结构）")
    parser.add_argument("--per_class",  type=int, default=50,
                        help="每类测试样本数量")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device",     type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # -------------------------
    # 4.1 构建一个小规模测试集
    # -------------------------
    full_dataset = datasets.ImageFolder(args.data_dir)
    class_to_idxs = {}
    for idx, (_, label) in enumerate(full_dataset.imgs):
        class_to_idxs.setdefault(label, []).append(idx)

    sampled_idxs = []
    for label, idxs in class_to_idxs.items():
        np.random.seed(42)
        chosen = np.random.choice(idxs, size=min(args.per_class, len(idxs)), replace=False)
        sampled_idxs.extend(chosen.tolist())
    subset = torch.utils.data.Subset(full_dataset, sampled_idxs)

    # -------------------------
    # 4.2 定义基础变换与归一化
    # -------------------------
    base_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    normalize_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # -------------------------
    # 4.3 生成 “模式 → dataloader” 映射
    # -------------------------
    modes = ["orig", "erasing", "noise", "brightness"]
    dataloaders = {}
    for mode in modes:
        pert_dataset = PerturbedDataset(
            subset=subset,
            mode=mode,
            base_transform=base_transform,
            normalize_transform=normalize_transform
        )
        dataloaders[mode] = torch.utils.data.DataLoader(
            pert_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,      # 保持多进程加速
            pin_memory=True
        )

    # -------------------------
    # 4.4 加载 ResNet50 与 ViT 模型
    # -------------------------
    num_classes = len(full_dataset.classes)

    # ResNet50
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device)
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    resnet.load_state_dict(torch.load("best_model.pth", map_location=device))
    resnet.to(device).eval()

    # ViT
    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1).to(device)
    vit.heads = nn.Sequential(nn.Linear(vit.heads.head.in_features, num_classes))
    vit.load_state_dict(torch.load("vit_model.pth", map_location=device))
    vit.to(device).eval()

    # -------------------------
    # 4.5 对每种模式分别测试准确率（ResNet、ViT、Ensemble）
    # -------------------------
    results = {"mode": [], "resnet_acc": [], "vit_acc": [], "ens_acc": []}
    for mode in modes:
        dl = dataloaders[mode]
        r_acc   = evaluate_model(resnet, dl, device)
        v_acc   = evaluate_model(vit, dl, device)
        ens_acc = evaluate_ensemble(resnet, vit, dl, device)
        print(f"[{mode}] ResNet50 Acc: {r_acc:.2f}%, ViT Acc: {v_acc:.2f}%, Ensemble Acc: {ens_acc:.2f}%")
        results["mode"].append(mode)
        results["resnet_acc"].append(r_acc)
        results["vit_acc"].append(v_acc)
        results["ens_acc"].append(ens_acc)

    # -------------------------
    # 4.6 绘制柱状图（带上 Ensemble）
    # -------------------------
    x = np.arange(len(modes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(x - width, results["resnet_acc"], width, label="ResNet50")
    ax.bar(x       , results["vit_acc"],   width, label="ViT")
    ax.bar(x + width, results["ens_acc"],   width, label="Ensemble")
    ax.set_xticks(x)
    ax.set_xticklabels(["原始", "遮挡", "噪声", "亮度"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("ResNet50 vs ViT vs Ensemble 在扰动下的鲁棒性对比")
    ax.legend()
    plt.tight_layout()

    os.makedirs("exp4_robustness_ensemble", exist_ok=True)
    plt.savefig("exp4_robustness_ensemble/robustness_bar.png")
    plt.close(fig)

    # 打印结果表格
    print("\n=== 对抗扰动测试结果（含集成） ===")
    print(f"{'模式':<10} | {'ResNet50 (%)':<12} | {'ViT (%)':<8} | {'Ensemble (%)':<12}")
    print("-" *  50)
    for i, mode in enumerate(modes):
        print(f"{mode:<10} | {results['resnet_acc'][i]:<12.2f} | {results['vit_acc'][i]:<8.2f} | {results['ens_acc'][i]:<12.2f}")

    print("实验四（含集成模型）完成，结果保存在 exp4_robustness_ensemble/robustness_bar.png。")


if __name__ == "__main__":
    main()
