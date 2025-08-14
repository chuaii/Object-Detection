import os
import cv2
import random
import shutil
import yaml
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import numpy as np


class VideoToYoloDataset:
    def __init__(self, video_path, output_dir, frame_interval=1, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """
        初始化视频到 YOLOv8 数据集转换器

        参数:
            video_path: 输入视频文件路径
            output_dir: 输出数据集目录
            frame_interval: 每隔多少帧提取一对
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
        """
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.frame_interval = frame_interval
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # 创建必要的目录
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.train_images_dir = self.images_dir / "train"
        self.train_labels_dir = self.labels_dir / "train"
        self.val_images_dir = self.images_dir / "val"
        self.val_labels_dir = self.labels_dir / "val"
        self.test_images_dir = self.images_dir / "test"
        self.test_labels_dir = self.labels_dir / "test"

        for dir_path in [
            self.train_images_dir, self.train_labels_dir,
            self.val_images_dir, self.val_labels_dir,
            self.test_images_dir, self.test_labels_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # 视频信息
        self.video = cv2.VideoCapture(video_path)
        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"视频信息: {self.width}x{self.height}, {self.fps:.2f} FPS, {self.frame_count} 帧")

    def extract_frame_pairs(self):
        """从视频中提取帧对并保存为图像"""
        print(f"开始提取帧对 (每隔 {self.frame_interval} 帧提取一对)...")

        # 计算可提取的帧对数量
        max_pairs = self.frame_count - self.frame_interval
        if max_pairs <= 0:
            raise ValueError("视频帧数太少，无法提取帧对")

        print(f"总共可提取 {max_pairs} 对帧")

        for i in tqdm(range(max_pairs)):
            # 读取第一帧
            self.video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret1, frame1 = self.video.read()

            # 读取第二帧
            self.video.set(cv2.CAP_PROP_POS_FRAMES, i + self.frame_interval)
            ret2, frame2 = self.video.read()

            if not ret1 or not ret2:
                print(f"警告: 无法读取帧对 {i}, {i + self.frame_interval}，跳过")
                continue

            # 保存帧对
            pair_name = f"pair_{i:06d}.jpg"
            pair_path = self.images_dir / pair_name

            # 合并两帧为六通道（这里先保存为RGB-RGB水平拼接，训练时再处理为六通道）
            combined_frame = np.hstack([frame1, frame2])
            cv2.imwrite(str(pair_path), combined_frame)

        self.video.release()
        print(f"成功提取 {max_pairs} 对帧")
        return max_pairs

    def split_dataset(self, total_pairs=None):
        """将数据集分为训练集、验证集和测试集"""
        if total_pairs is None:
            total_pairs = len(list(self.images_dir.glob("*.jpg")))

        # 获取所有图像文件
        image_files = sorted(list(self.images_dir.glob("*.jpg")))

        # 打乱文件顺序
        random.seed(42)  # 设置随机种子，确保结果可重现
        random.shuffle(image_files)

        # 计算分割点
        train_split = int(total_pairs * self.train_ratio)
        val_split = int(total_pairs * (self.train_ratio + self.val_ratio))

        # 分割数据集
        train_files = image_files[:train_split]
        val_files = image_files[train_split:val_split]
        test_files = image_files[val_split:]

        # 移动文件到相应目录
        print("正在分割数据集...")
        self._move_files_to_dir(train_files, self.train_images_dir)
        self._move_files_to_dir(val_files, self.val_images_dir)
        self._move_files_to_dir(test_files, self.test_images_dir)

        print(f"数据集分割完成:")
        print(f"  训练集: {len(train_files)} 图像对")
        print(f"  验证集: {len(val_files)} 图像对")
        print(f"  测试集: {len(test_files)} 图像对")

        return len(train_files), len(val_files), len(test_files)

    def _move_files_to_dir(self, files, target_dir):
        """将文件移动到目标目录"""
        for file in files:
            shutil.move(str(file), str(target_dir / file.name))

    def create_data_yaml(self, classes):
        """创建 YOLOv8 配置文件 (taozi.yaml)"""
        data_yaml_path = self.output_dir / "taozi.yaml"

        data_config = {
            "train": str(self.train_images_dir.relative_to(self.output_dir)),
            "val": str(self.val_images_dir.relative_to(self.output_dir)),
            "test": str(self.test_images_dir.relative_to(self.output_dir)),
            "nc": len(classes),
            "names": classes
        }

        with open(data_yaml_path, "w") as f:
            yaml.dump(data_config, f, default_flow_style=False)

        print(f"已创建配置文件: {data_yaml_path}")
        return data_yaml_path

    def create_sample_labels(self, classes):
        """为演示创建样本标签文件（所有标签初始化为空）"""
        print("创建样本标签文件...")

        for split in ["train", "val", "test"]:
            images_dir = self.images_dir / split
            labels_dir = self.labels_dir / split

            for image_path in images_dir.glob("*.jpg"):
                label_path = labels_dir / f"{image_path.stem}.txt"
                # 创建空标签文件（表示没有检测到对象）
                with open(label_path, "w") as f:
                    pass

    def generate_dataset(self, classes):
        """生成完整的 YOLOv8 数据集"""
        print(f"开始生成 YOLOv8 数据集: {self.output_dir}")

        # 1. 提取帧对
        total_pairs = self.extract_frame_pairs()

        # 2. 分割数据集
        train_size, val_size, test_size = self.split_dataset(total_pairs)

        # 3. 创建配置文件
        data_yaml_path = self.create_data_yaml(classes)

        # 4. 创建样本标签文件
        self.create_sample_labels(classes)

        print("\n数据集生成完成！")
        print(f"数据集位置: {self.output_dir}")
        print(f"配置文件: {data_yaml_path}")
        print("\n下一步:")
        print("1. 使用标注工具（如 Roboflow、LabelImg）标注生成的图像对")
        print("2. 将标注文件（.txt 格式）放入对应的 labels/train, labels/val, labels/test 目录")
        print("3. 使用以下命令训练 YOLOv8 模型:")
        print(f"   yolo train model=yolov8n_6channel.pt data={data_yaml_path} epochs=100 imgsz={self.width * 2}")

        return {
            "output_dir": self.output_dir,
            "data_yaml": data_yaml_path,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size
        }


if __name__ == "__main__":
    # 配置参数
    VIDEO_PATH = "img_video_source/taozi4.mp4"  # 替换为你的视频路径
    OUTPUT_DIR = "datasets/taozi"  # 输出数据集目录
    FRAME_INTERVAL = 1  # 提取相邻帧对
    # 配置参数
    # CLASSES = ["bed", "dog", "chair", "bottle", "gloves", "desk"]  # 你的类别列表
    CLASSES = ["dog"]  # 你的类别列表

    # 创建数据集生成器
    dataset_creator = VideoToYoloDataset(
        video_path=VIDEO_PATH,
        output_dir=OUTPUT_DIR,
        frame_interval=FRAME_INTERVAL
    )

    # 生成数据集
    dataset_info = dataset_creator.generate_dataset(classes=CLASSES)
