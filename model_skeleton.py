import torch
import torch.nn as nn
import torchvision.models as models

def create_model(num_classes: int) -> nn.Module:
    """
    创建ResNet50模型（使用ImageNet预训练权重），替换最后的全连接层以适配新的类别数。
    参数:
        num_classes: 新的分类类别数量
    返回:
        微调后的 ResNet50 模型 (torch.nn.Module)
    """
    # 加载ResNet50预训练模型
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)  # 使用ImageNet预训练权重
    # 如果使用较旧版本的torchvision，可以改用:
    # model = models.resnet50(pretrained=True)
    # 替换最后的全连接层，使输出节点数等于类别数
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_vit_model(num_classes: int) -> nn.Module:
    """
    创建一个使用 ImageNet 预训练权重的 ViT-Base_16 模型，
    并将其分类头替换为输出 `num_classes` 个类别的线性层。
    """
    # 加载预训练的 ViT-Base_16 模型
    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    # 用新的线性层替换默认的分类头
    vit.heads = nn.Sequential(nn.Linear(vit.heads.head.in_features, num_classes))
    return vit