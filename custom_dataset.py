import os
import json
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    """
    自定义数据集类：从图像和对应的JSON分割掩码加载数据。
    假设目录结构：image_dir/类别名/xxx.jpg 与 json_dir/类别名/xxx.json 对应。
    JSON文件内包含 'segmentation' 字段 (多边形坐标点列表)，用于生成掩码去除背景。
    """
    def __init__(self, image_dir: str, json_dir: str, transform=None):
        """
        初始化数据集，读取所有图像路径和标签。
        参数:
            image_dir: 存放图像文件的根目录（按类分子文件夹）
            json_dir: 存放与图像对应的JSON标注的根目录（按类分子文件夹）
            transform: 要应用的图像变换(例如数据增强和预处理)
        """
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.transform = transform

        # 获取类别名列表（按文件夹名称），并建立类别到索引的映射
        self.classes = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
        self.classes.sort()  # 按名称排序保证一致
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # 收集所有 (图像路径, JSON路径, 类别索引) 三元组
        self.samples = []
        # 支持的图像扩展名集合
        IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
        for cls in self.classes:
            cls_img_dir = os.path.join(image_dir, cls)
            cls_json_dir = os.path.join(json_dir, cls)
            if not os.path.isdir(cls_json_dir):
                raise FileNotFoundError(f"对应的JSON目录不存在: {cls_json_dir}")
            # 遍历该类别文件夹下的所有图像文件
            for fname in os.listdir(cls_img_dir):
                fpath = os.path.join(cls_img_dir, fname)
                if os.path.splitext(fname)[1].lower() not in IMG_EXTENSIONS:
                    continue  # 跳过非图像文件
                json_path = os.path.join(cls_json_dir, os.path.splitext(fname)[0] + '.json')
                if not os.path.exists(json_path):
                    raise FileNotFoundError(f"未找到图像对应的JSON文件: {json_path}")
                # 保存样本路径和标签
                self.samples.append((fpath, json_path, self.class_to_idx[cls]))

    def __len__(self):
        # 数据集样本总数
        return len(self.samples)

    def __getitem__(self, index: int):
        """
        获取索引对应的图像及标签。
        返回:
            (image_tensor, label) 图像张量和类别索引
        """
        img_path, json_path, label = self.samples[index]
        # 打开图像并转换为RGB模式
        image = Image.open(img_path).convert('RGB')

        # 加载JSON文件，获取segmentation多边形坐标
        with open(json_path, 'r') as f:
            data = json.load(f)
        segmentation = data.get('segmentation')

        # 创建掩码图像（单通道），初始全0（背景为0）
        if segmentation is None:
            # 若无segmentation信息，则不做背景去除（掩码全为1）
            mask = Image.new('L', image.size, 255)
        else:
            mask = Image.new('L', image.size, 0)
            draw = ImageDraw.Draw(mask)
            # segmentation可能是嵌套列表（如果有多个多边形）
            if isinstance(segmentation, list) and len(segmentation) > 0 and isinstance(segmentation[0], list):
                polygons = segmentation
            else:
                polygons = [segmentation]
            # 在掩码上绘制多边形区域，填充值255（表示保留区域）
            for poly in polygons:
                # 将坐标转换为整数
                poly_points = [int(x) for x in poly]
                draw.polygon(poly_points, fill=255)
        # 将背景设置为白色，在掩码为0的区域应用白色背景
        background = Image.new('RGB', image.size, (255, 255, 255))
        # 利用掩码合成新图：掩码非零的区域保留原图像，其余区域替换为白色
        image = Image.composite(image, background, mask)

        # 应用图像变换（数据增强和预处理）
        if self.transform:
            image = self.transform(image)
        return image, label
