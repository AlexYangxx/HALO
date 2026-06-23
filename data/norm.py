import os
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# -------------------------- 配置参数 --------------------------
# 原始数据集根目录（请替换为你的数据集路径）
ORIGINAL_DATASET_PATH = "data_dir/eval15/low"
# 处理后新数据集根目录（会自动创建）
NEW_DATASET_PATH = "normalized_dataset"
# ImageNet的均值和标准差（RGB通道）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# 支持的图像格式
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# -------------------------- 核心函数 --------------------------
def normalize_image(image_path, save_path):
    """
    对单张图像进行ImageNet归一化，并保存处理后的图像
    :param image_path: 原始图像路径
    :param save_path: 处理后图像的保存路径
    """
    try:
        # 1. 读取图像并转换为RGB格式（避免灰度图或RGBA格式问题）
        image = Image.open(image_path).convert('RGB')
        
        # 2. 定义归一化变换：转张量→归一化到0-1→ImageNet标准化
        preprocess = transforms.Compose([
            transforms.ToTensor(),  # 将PIL图像转为张量，像素值从0-255转为0-1
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)  # ImageNet标准化
        ])
        
        # 3. 执行归一化
        normalized_tensor = preprocess(image)
        
        # 4. 反归一化（还原到0-255范围，以便保存为图像）
        # 公式：x = x*std + mean → 还原到0-1，再×255转0-255
        denormalize = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.], std=[1/s for s in IMAGENET_STD]),
            transforms.Normalize(mean=[-m for m in IMAGENET_MEAN], std=[1., 1., 1.]),
            transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),  # 限制范围在0-1，避免溢出
            transforms.ToPILImage()  # 转回PIL图像
        ])
        denormalized_image = denormalize(normalized_tensor)
        
        # 5. 保存图像
        denormalized_image.save(save_path)
        
        return True
    except Exception as e:
        print(f"处理图像失败 {image_path}：{str(e)}")
        return False

def process_dataset(original_path, new_path):
    """
    遍历并处理整个数据集
    :param original_path: 原始数据集根目录
    :param new_path: 新数据集根目录
    """
    # 获取所有图像文件的路径
    image_paths = []
    for root, dirs, files in os.walk(original_path):
        for file in files:
            if file.lower().endswith(SUPPORTED_FORMATS):
                image_paths.append(os.path.join(root, file))
    
    if not image_paths:
        print("未找到任何支持的图像文件！")
        return
    
    # 遍历处理每张图像
    success_count = 0
    for img_path in tqdm(image_paths, desc="处理图像"):
        # 计算新路径（保持原目录结构）
        relative_path = os.path.relpath(img_path, original_path)
        new_img_path = os.path.join(new_path, relative_path)
        
        # 创建保存目录（如果不存在）
        new_dir = os.path.dirname(new_img_path)
        os.makedirs(new_dir, exist_ok=True)
        
        # 处理并保存图像
        if normalize_image(img_path, new_img_path):
            success_count += 1
    
    # 输出处理结果
    print(f"\n处理完成！总计：{len(image_paths)} 张，成功：{success_count} 张，失败：{len(image_paths)-success_count} 张")
    print(f"新数据集保存路径：{os.path.abspath(new_path)}")

# -------------------------- 执行主程序 --------------------------
if __name__ == "__main__":
    # 检查原始数据集是否存在
    if not os.path.exists(ORIGINAL_DATASET_PATH):
        print(f"错误：原始数据集路径不存在 → {ORIGINAL_DATASET_PATH}")
    else:
        process_dataset(ORIGINAL_DATASET_PATH, NEW_DATASET_PATH)