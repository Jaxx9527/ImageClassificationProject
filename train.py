# train.py

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from model_skeleton import create_model, get_vit_model
from dataloader import get_dataloaders

def train():
    parser = argparse.ArgumentParser(description="一次运行训练 ResNet50 和 ViT 两套模型")
    parser.add_argument("--data_dir",    type=str,   default="dataset",
                        help="数据集根目录，包含 train_jpg/train_json 和 val_jpg/val_json")
    parser.add_argument("--epochs",      type=int,   default=20,   help="训练轮数")
    parser.add_argument("--batch_size",  type=int,   default=32,   help="批量大小")
    parser.add_argument("--lr",          type=float, default=1e-3, help="初始学习率")
    parser.add_argument("--weight_decay",type=float, default=1e-4, help="AdamW 权重衰减")
    parser.add_argument("--num_workers", type=int,   default=4,    help="DataLoader 的 num_workers")
    args = parser.parse_args()

    # 构建 DataLoader
    train_img_dir  = os.path.join(args.data_dir, "train_jpg")
    train_json_dir = os.path.join(args.data_dir, "train_json")
    val_img_dir    = os.path.join(args.data_dir, "validation_jpg")
    val_json_dir   = os.path.join(args.data_dir, "validation_json")
    train_loader, val_loader = get_dataloaders(
        train_img_dir, train_json_dir,
        val_img_dir,   val_json_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 自动推断类别数
    num_classes = len(train_loader.dataset.classes)
    print(f"检测到 {num_classes} 个类别: {train_loader.dataset.classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 要训练的模型列表
    for model_name in ["resnet", "vit"]:
        print(f"\n=== 开始训练 {model_name.upper()} 模型 ===")
        # 选择模型及保存文件名
        if model_name == "resnet":
            model = create_model(num_classes)
            save_name = "best_model.pth"
        else:
            model = get_vit_model(num_classes)
            save_name = "vit_model.pth"
        model = model.to(device)

        # 损失、优化器、学习率调度
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs
        )

        best_acc = 0.0
        best_epoch = 0

        # 训练 & 验证循环
        for epoch in range(1, args.epochs + 1):
            # 训练
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct_train += preds.eq(labels).sum().item()
                total_train += labels.size(0)

            train_loss = running_loss / total_train
            train_acc  = correct_train / total_train

            # 验证
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    preds = outputs.argmax(dim=1)
                    correct_val += preds.eq(labels).sum().item()
                    total_val += labels.size(0)

            val_loss = val_loss / total_val
            val_acc  = correct_val / total_val

            # 学习率更新
            scheduler.step()

            # 日志输出
            print(f"[{model_name.upper()}] Epoch {epoch}/{args.epochs}  "
                  f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc*100:.2f}%  "
                  f"Val   Loss: {val_loss:.4f}  Val   Acc: {val_acc*100:.2f}%")

            # 保存最佳权重
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(), save_name)
                print(f"** [{model_name.upper()}] 保存最佳模型 '{save_name}' (Epoch {epoch}, Val Acc={val_acc*100:.2f}%) **")

        print(f"=== {model_name.upper()} 训练结束：最佳 Val Acc={best_acc*100:.2f}% (Epoch {best_epoch})，权重保存在 {save_name} ===")

if __name__ == "__main__":
    train()
