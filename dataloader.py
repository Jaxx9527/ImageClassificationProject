import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from custom_dataset import CustomDataset

def get_dataloaders(train_img_dir: str, train_json_dir: str,
                    val_img_dir: str, val_json_dir: str,
                    batch_size: int = 32, num_workers: int = 4) -> (DataLoader, DataLoader):
    """
    定义训练和验证集的DataLoader。
    参数:
        train_img_dir: 训练集图像文件夹根路径（按类分文件夹）
        train_json_dir: 训练集JSON标注文件夹根路径（按类分文件夹）
        val_img_dir: 验证集图像文件夹根路径
        val_json_dir: 验证集JSON标注文件夹根路径
        batch_size: 批量大小
        num_workers: DataLoader的工作进程数
    返回:
        (train_loader, val_loader)
    """
    # 图像Net数据集均值和标准差（用于归一化）
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    # 定义训练数据增强和预处理变换
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),            # 随机裁剪并缩放到224x224
        transforms.RandomHorizontalFlip(),            # 随机水平翻转
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # 颜色扰动
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    # 定义验证/测试数据的预处理（无随机变换）
    val_transforms = transforms.Compose([
        transforms.Resize(256),               # 将最短边缩放到256（保持长宽比）
        transforms.CenterCrop(224),           # 中心裁剪224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # 构建自定义数据集
    train_dataset = CustomDataset(train_img_dir, train_json_dir, transform=train_transforms)
    val_dataset   = CustomDataset(val_img_dir, val_json_dir, transform=val_transforms)
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

