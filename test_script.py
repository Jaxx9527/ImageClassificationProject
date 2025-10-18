import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# 集成模型类：ResNet50 + ViT
class EnsembleModel(nn.Module):
    def __init__(self, modelA, modelB):
        super(EnsembleModel, self).__init__()
        self.modelA = modelA
        self.modelB = modelB

    def forward(self, x):
        probsA = F.softmax(self.modelA(x), dim=1)
        probsB = F.softmax(self.modelB(x), dim=1)
        return (probsA + probsB) / 2

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 准备测试集 DataLoader，用来推断类别数和类名
    test_dir = "./test"
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    testset = torchvision.datasets.ImageFolder(test_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)

    num_classes = len(testset.classes)

    # 2. 加载 ResNet50 模型及权重
    resnet = torchvision.models.resnet50(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    resnet.load_state_dict(torch.load("best_model.pth", map_location=device))
    resnet.to(device).eval()

    # 3. 加载 ViT 模型及权重
    vit = torchvision.models.vit_b_16(weights=None)
    vit.heads = nn.Sequential(nn.Linear(vit.heads.head.in_features, num_classes))
    vit.load_state_dict(torch.load("vit_model.pth", map_location=device))
    vit.to(device).eval()

    # 4. 用两个模型构建集成网络
    ensemble_net = EnsembleModel(resnet, vit).to(device)
    ensemble_net.eval()

    return ensemble_net, test_loader

def test(net, test_loader):
    net.eval()
    correct = 0
    total = 0
    errors = []  # 存放预测错误的样本信息

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            # TTA：水平翻转
            inputs_flipped = torch.flip(inputs, dims=[3])

            # 两次推理
            probs_orig = net(inputs)
            probs_flip = net(inputs_flipped)

            # 概率平均
            avg_probs = (probs_orig + probs_flip) / 2
            _, predicted = torch.max(avg_probs, 1)

            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

            # 记录每个错误样本的路径和标签
            for i in range(targets.size(0)):
                if predicted[i] != targets[i]:
                    idx_global = batch_idx * test_loader.batch_size + i
                    img_path = test_loader.dataset.samples[idx_global][0]
                    pred_label = test_loader.dataset.classes[predicted[i].item()]
                    true_label = test_loader.dataset.classes[targets[i].item()]
                    errors.append((img_path, pred_label, true_label))

    accuracy = correct / total * 100.0
    print(f"Test accuracy: {accuracy:.2f}%")
    if errors:
        print("Misclassified samples (path, predicted → true):")
        for path, pred, true in errors:
            print(f"{path},  {pred} → {true}")
    return accuracy

def main():
    net, test_loader = load_model()
    test(net, test_loader)

if __name__ == "__main__":
    main()


