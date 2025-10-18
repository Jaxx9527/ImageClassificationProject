# exp2_gradcam_vit.py

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from types import MethodType

# -----------------------------
# 说明：
#  1. 使用 Grad-CAM 可视化 ResNet50 在最终决策时的类激活图；
#  2. 使用前面“Attention Rollout”方法可视化 ViT-B/16 在最终决策时对各 patch 的注意力分布；
#  3. 运行示例：
#       python exp2_gradcam_vit.py --img_path test/1Bell_tower/202504231722011.jpg --device cpu
#  4. 结果会输出到 exp2_resnet_gradcam/ 与 exp2_vit_rollout/ 中。
# -----------------------------

import argparse

def compute_vit_rollout(vit, input_tensor):
    """
    计算 ViT Attention Rollout 热力图（numpy, [224,224], 归一化到 [0,1]）。
    前提：在 ViT 模型里，每个 EncoderBlock 的 self_attention 已做 monkey patch
    并注册了 hook（与 exp1 脚本相同）。此处假定 vit_attentions 已在外部被填充。
    """
    # vit_attentions 已被全局累积
    global vit_attentions
    num_layers = len(vit_attentions)
    if num_layers == 0:
        raise RuntimeError("未捕获到任何 ViT attention 权重，请确认在 exp1 中的 hook 已正确注册。")

    # 把每层 attn_weights ([1, num_heads, N, N]) 在 dim=1( num_heads ) 上取平均 → [N, N]
    attn_stack = []
    for attn in vit_attentions:
        attn_avg = attn.squeeze(0).mean(dim=0).numpy()  # [N, N]
        attn_stack.append(attn_avg)
    attn_stack = np.stack(attn_stack, axis=0)  # [L, N, N]

    # Attention Rollout: 每层 residual + row-normalize → 矩阵相乘
    result = np.eye(attn_stack.shape[-1], dtype=np.float32)
    for a in attn_stack:
        a_res = a + np.eye(a.shape[-1], dtype=np.float32)
        a_res = a_res / (a_res.sum(axis=-1, keepdims=True) + 1e-12)
        result = a_res @ result

    # 取 CLS token (index 0) → 所有 patch ([1:N]) 的注意力
    cls_attn = result[0, 1:]  # [N_patch]
    n_patch = int(np.sqrt(cls_attn.shape[0]))  # 14
    cls_map = cls_attn.reshape(n_patch, n_patch)  # [14,14]
    cls_map = cls_map - cls_map.min()
    cls_map = cls_map / (cls_map.max() + 1e-8)

    # 插值到 224×224
    cls_map_img = Image.fromarray((cls_map * 255).astype("uint8")).resize(
        (224, 224), resample=Image.BILINEAR
    )
    return np.array(cls_map_img, dtype=np.float32) / 255.0  # [224,224], 归一化到 [0,1]

def visualize_resnet_gradcam(resnet, img_rgb, preprocess, target_category, output_path, device):
    """
    用 GradCAM 可视化 ResNet50 在指定类别上的类激活图，并叠加到原始 RGB 图像上（[0,1]）。
    - img_rgb: numpy 数组，shape [224,224,3]，值范围 [0,1]
    - preprocess: 图像预处理（PIL → Tensor + Normalize）
    - target_category: 要可视化的类别索引
    - output_path: 保存叠加后的可视化图路径
    - device: torch.device
    """
    # 把原始 PIL 图像先转到 [0,1] 范围的 numpy，再让 GradCAM 在此基础上叠加
    input_tensor = preprocess(Image.fromarray((img_rgb * 255).astype("uint8"))).unsqueeze(0).to(device)

    # GradCAM：移除 use_cuda 参数
    # 指定 target_layers 为 resnet.layer4[-1]（最后一个 BottleneckBlock 的卷积层）
    target_layers = [resnet.layer4[-1]]
    cam = GradCAM(model=resnet, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_category)]
    grayscale_cam = cam(input_tensor, targets)[0]  # [224,224], [0,1]

    # 叠加到原始图像上
    visualization = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
    Image.fromarray(visualization).save(output_path)

def main():
    parser = argparse.ArgumentParser(description="实验二：Grad-CAM & ViT Attention Rollout 可视化")
    parser.add_argument("--img_path", type=str, required=True,
                        help="待可视化图像路径")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"],
                        help="运行设备 (cpu 或 cuda)")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("检测到 --device cuda，但 CUDA 不可用，将切换到 CPU。")
        device = torch.device("cpu")

    # -------------------------
    # 1. 读取并预处理图像
    # -------------------------
    pil_img = Image.open(args.img_path).convert("RGB")
    # 归一化到 [0,1] numpy，用于 show_cam_on_image 叠加
    orig_pil = pil_img.resize((224, 224))
    img_rgb = np.array(orig_pil).astype(np.float32) / 255.0

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # -------------------------
    # 2. 加载 ResNet50 并写入分类头，加载权重
    # -------------------------
    # 2.A ResNet50
    # 先用预训练权重初始化，然后替换 fc 层，使输出等于类别数
    num_classes = len(os.listdir("dataset/train_jpg"))  # 与训练时保持一致
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device).eval()
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes).to(device)
    resnet.load_state_dict(torch.load("best_model.pth", map_location=device))
    resnet.eval()

    # 得到模型预测，确定 target_category
    input_tensor = preprocess(orig_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = resnet(input_tensor)
    pred_idx = logits.argmax(dim=1).item()

    # 进行 GradCAM 可视化
    os.makedirs("exp2_resnet_gradcam", exist_ok=True)
    vis_path = os.path.join("exp2_resnet_gradcam", f"resnet_gradcam_{pred_idx}.png")
    visualize_resnet_gradcam(resnet, img_rgb, preprocess, pred_idx, vis_path, device)
    print("ResNet50 Grad-CAM 可视化已保存至：", vis_path)

    # -------------------------
    # 3. 加载 ViT-B/16 并写入分类头、加载权重
    # -------------------------
    # 3.A ViT-B/16
    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1).to(device).eval()
    vit.heads = nn.Sequential(nn.Linear(vit.heads.head.in_features, num_classes)).to(device)
    vit.load_state_dict(torch.load("vit_model.pth", map_location=device))
    vit.eval()

    # 3.B 为了捕获注意力权重，需要对每个 EncoderBlock 做 monkey patch（与 exp1 相同）
    global vit_attentions
    vit_attentions = []
    for idx, block in enumerate(vit.encoder.layers):
        if hasattr(block, "self_attention"):
            orig_forward = block.self_attention.forward
            def forward_with_weights(self, query, key, value, **kwargs):
                return orig_forward(query, key, value, need_weights=True, **{k:v for k,v in kwargs.items() if k!="need_weights"})
            block.self_attention.forward = MethodType(forward_with_weights, block.self_attention)
            block.self_attention.register_forward_hook(
                lambda module, inp, outp: vit_attentions.append(outp[1].detach().cpu())
            )
        else:
            raise RuntimeError(f"在第 {idx} 个 EncoderBlock 中未找到 'self_attention'，请确认 TorchVision 版本一致。")

    # 3.C 前向推理以捕获注意力权重
    vit_input = preprocess(orig_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        _ = vit(vit_input)

    # 3.D 生成 Attention Rollout 热力图
    os.makedirs("exp2_vit_rollout", exist_ok=True)
    cls_map_np = compute_vit_rollout(vit, vit_input)  # [224,224] 归一化到 [0,1]
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(orig_pil)
    ax.imshow(cls_map_np, cmap="jet", alpha=0.5)
    ax.axis("off")
    vit_vis_path = os.path.join("exp2_vit_rollout", "vit_gradcam.png")
    plt.savefig(vit_vis_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print("ViT Attention Rollout 热力图保存至：", vit_vis_path)

    print("实验二可视化完成。")

if __name__ == "__main__":
    main()

