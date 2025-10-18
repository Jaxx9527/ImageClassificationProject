# exp1_feature_map_vis.py

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import warnings
from types import MethodType

# -----------------------------
# 说明：
# 1. 本脚本展示 ResNet50 与 ViT-B/16 在同一图像上的“中间特征”可视化：
#    - ResNet50：Hook 出 layer1 / layer3 / layer4 特征图，取通道平均后叠加热力图；
#    - ViT-B/16：对每个 EncoderBlock 内的 MultiheadAttention 强制使用 need_weights=True，
#      Hook 出返回的 attention_weights，然后通过 Attention Rollout（逐层相乘）计算最终
#      [CLS]→各 patch 的注意力分布，生成热力图叠加到原图。
# 2. 使用示例：
#      python exp1_feature_map_vis.py --img_path test/1Bell_tower/202504231722011.jpg --device cpu
#    脚本会生成：
#      exp1_resnet_vis/resnet_layer1.png
#      exp1_resnet_vis/resnet_layer3.png
#      exp1_resnet_vis/resnet_layer4.png
#      exp1_vit_vis/exp1_vit_rollout.png
# -----------------------------

# -----------------------------
# 1. ResNet50 特征 Hook
# -----------------------------
resnet_activations = {}
def get_resnet_hook(name):
    def hook(model, input, output):
        # output: Tensor shape [B, C, H, W]
        resnet_activations[name] = output.detach().cpu()
    return hook

# -----------------------------
# 2. ViT Attention Rollout Hook
# -----------------------------
vit_attentions = []  # 用来保存每层的 attn_output_weights

def get_vit_attention_hook(module, input, output):
    """
    针对 nn.MultiheadAttention，在 forward(query,key,value, need_weights=True) 时，
    output 是 (attn_output, attn_output_weights)。此处我们只保存 attn_output_weights，
    形状为 [B, N, N]。
    """
    attn_weights = output[1].detach().cpu()  # 第二项就是 attention weights
    vit_attentions.append(attn_weights)

# -----------------------------
# 3. 主函数
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="实验一：ResNet50 与 ViT 中间特征可视化（强制 need_weights=True）"
    )
    parser.add_argument("--img_path", type=str, required=True,
                        help="待可视化的图像路径")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu","cuda"], help="运行设备（cpu 或 cuda）")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        warnings.warn("指定 --device cuda，但未检测到 GPU，可用资源，将切换到 CPU。")
        device = torch.device("cpu")

    # -------------------------
    # 4. 加载 ResNet50 并注册 hook
    # -------------------------
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device).eval()
    resnet.layer1.register_forward_hook(get_resnet_hook("layer1"))
    resnet.layer3.register_forward_hook(get_resnet_hook("layer3"))
    resnet.layer4.register_forward_hook(get_resnet_hook("layer4"))

    # -------------------------
    # 5. 加载 ViT-B/16 并“猴子补丁”+注册 hook
    # -------------------------
    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1).to(device).eval()

    # 全局清空 vit_attentions
    global vit_attentions
    vit_attentions = []

    # 对每个 EncoderBlock：把 self_attention.forward 强制改为 need_weights=True 版本，再 hook
    for idx, block in enumerate(vit.encoder.layers):
        if hasattr(block, "self_attention"):
            # 原始 forward
            orig_forward = block.self_attention.forward

            # 定义新的 forward 方法：强制传 need_weights=True
            def forward_with_weights(self, query, key, value, **kwargs):
                # 保持原有的 attn_mask、key_padding_mask 等其他 kwargs
                return orig_forward(query, key, value, need_weights=True, **{k: v for k, v in kwargs.items() if k != "need_weights"})

            # 把新的 forward 绑定到模块实例
            block.self_attention.forward = MethodType(forward_with_weights, block.self_attention)

            # 注册 hook，捕获 forward 返回值的第二个元素（attn_output_weights）
            block.self_attention.register_forward_hook(get_vit_attention_hook)
        else:
            raise RuntimeError(
                f"在第 {idx} 个 EncoderBlock 中未找到 'self_attention' 子模块，"
                "请检查 torchvision 版本或脚本匹配性。"
            )

    # -------------------------
    # 6. 读取并预处理图像
    # -------------------------
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])
    img = Image.open(args.img_path).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # -------------------------
    # 7. 前向推理，触发 hook
    # -------------------------
    with torch.no_grad():
        _ = resnet(input_tensor)  # 触发 ResNet 的 hook
        _ = vit(input_tensor)     # 触发 ViT 的 hook（此时 attention forward 已被替换，输出 attn_weights）

    # -------------------------
    # 8. 可视化 ResNet50 中间特征
    # -------------------------
    os.makedirs("exp1_resnet_vis", exist_ok=True)
    for layer_name, feat in resnet_activations.items():
        # feat: [1, C, H, W]
        feat = feat.squeeze(0)         # [C, H, W]
        cam = feat.mean(dim=0)         # 对 C 个通道求平均，得到 [H, W]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)  # 归一化到 [0,1]
        cam_np = cam.numpy()

        # 把热力图插值到 224×224
        cam_img = Image.fromarray((cam_np * 255).astype("uint8")).resize(
            (224,224), resample=Image.BILINEAR
        )
        # 将原始图裁剪到 224×224
        orig_crop = transforms.CenterCrop(224)(img)

        fig, ax = plt.subplots(figsize=(4,4))
        ax.imshow(orig_crop)
        ax.imshow(cam_img, cmap="jet", alpha=0.5)
        ax.axis("off")
        save_path = os.path.join("exp1_resnet_vis", f"resnet_{layer_name}.png")
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    print("✅ ResNet50 中间特征可视化完成，图像保存在 exp1_resnet_vis/ 目录下。")

    # -------------------------
    # 9. 可视化 ViT Attention Rollout
    # -------------------------
    num_layers = len(vit_attentions)
    if num_layers == 0:
        raise RuntimeError("未捕获到任何 ViT attention 权重，请确认 Hook 已正确注册。")

    # 将每层 attn_weights 在 dim=1（num_heads）上取平均，再堆叠成 [L, N, N]
    attn_stack = []
    for attn in vit_attentions:
        # attn: [1, num_heads, N, N]
        attn_avg = attn.squeeze(0).mean(dim=0).numpy()  # [N, N]
        attn_stack.append(attn_avg)
    attn_stack = np.stack(attn_stack, axis=0)  # [L, N, N]

    # Attention Rollout：每层 attn 矩阵加 residual，再行归一化，逐层相乘
    result = np.eye(attn_stack.shape[-1], dtype=np.float32)
    for a in attn_stack:
        a_res = a + np.eye(a.shape[-1], dtype=np.float32)
        a_res = a_res / (a_res.sum(axis=-1, keepdims=True) + 1e-12)
        result = a_res @ result

    # 取 CLS token (index 0) 对所有 patch 的注意力分布
    cls_attn = result[0, 1:]  # 去掉 CLS 自身
    n_patch = int(np.sqrt(cls_attn.shape[0]))  # e.g. 14
    cls_map = cls_attn.reshape(n_patch, n_patch)  # [n_patch, n_patch]
    cls_map = cls_map - cls_map.min()
    cls_map = cls_map / (cls_map.max() + 1e-8)

    # 插值到完整 224×224
    cls_map_img = Image.fromarray((cls_map * 255).astype("uint8")).resize(
        (224,224), resample=Image.BILINEAR
    )
    orig_crop = transforms.CenterCrop(224)(img)

    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(orig_crop)
    ax.imshow(cls_map_img, cmap="jet", alpha=0.5)
    ax.axis("off")
    os.makedirs("exp1_vit_vis", exist_ok=True)
    fig.savefig("exp1_vit_vis/exp1_vit_rollout.png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print("✅ ViT Attention Rollout 可视化完成，图像保存在 exp1_vit_vis/exp1_vit_rollout.png。")

if __name__ == "__main__":
    main()

