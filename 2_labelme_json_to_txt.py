import os
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


class LabelmeToYoloConverter:
    def __init__(self, input_dir, output_dir, class_map=None):
        """
        初始化 Labelme 到 YOLO 格式转换器

        参数:
            input_dir: 包含 Labelme JSON 文件的目录
            output_dir: 输出 YOLO 格式文件的目录
            class_map: 类别名称到 ID 的映射字典，默认为空（自动生成）
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 类别映射
        self.class_map = class_map or {}
        self.class_id = {}  # 类别名称到 ID 的映射

    def convert(self):
        """执行转换"""
        # 获取所有 JSON 文件
        json_files = list(self.input_dir.glob("*.json"))
        print(f"找到 {len(json_files)} 个 JSON 文件")

        # 处理每个 JSON 文件
        for json_file in tqdm(json_files, desc="转换中"):
            self._convert_json(json_file)

        # 保存类别映射
        if not self.class_map and self.class_id:
            classes_path = self.output_dir.parent / "classes.txt"
            with open(classes_path, "w") as f:
                for name, id in sorted(self.class_id.items(), key=lambda x: x[1]):
                    f.write(f"{name}\n")
            print(f"已生成类别映射文件: {classes_path}")

        print(f"转换完成，YOLO 格式文件保存在: {self.output_dir}")

    def _convert_json(self, json_file):
        """转换单个 JSON 文件"""
        try:
            # 读取 JSON 文件
            with open(json_file, "r") as f:
                data = json.load(f)

            # 获取图像尺寸
            image_width = data.get("imageWidth", 0)
            image_height = data.get("imageHeight", 0)

            # 如果没有图像尺寸信息，尝试从图像文件获取
            if image_width == 0 or image_height == 0:
                image_path = self.input_dir / data.get("imagePath", "")
                if image_path.exists():
                    img = cv2.imread(str(image_path))
                    image_height, image_width = img.shape[:2]
                else:
                    print(f"警告: 无法获取图像尺寸，跳过 {json_file}")
                    return

            # 处理标注
            yolo_lines = []
            for shape in data.get("shapes", []):
                label = shape.get("label", "")
                if not label:
                    continue

                # 获取类别 ID
                if label in self.class_map:
                    class_id = self.class_map[label]
                else:
                    if label not in self.class_id:
                        self.class_id[label] = len(self.class_id)
                    class_id = self.class_id[label]

                # 处理边界框
                points = shape.get("points", [])
                if len(points) < 2:
                    continue

                # 检测类型（矩形或多边形）
                shape_type = shape.get("shape_type", "polygon")

                if shape_type == "rectangle" and len(points) == 2:
                    # 矩形标注
                    x1, y1 = points[0]
                    x2, y2 = points[1]

                    # 计算边界框中心和宽高
                    x_center = (x1 + x2) / 2 / image_width
                    y_center = (y1 + y2) / 2 / image_height
                    width = abs(x2 - x1) / image_width
                    height = abs(y2 - y1) / image_height

                    # 确保数值在有效范围内
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))

                    yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                elif shape_type == "polygon" and len(points) >= 3:
                    # 多边形标注（转换为边界框）
                    points = np.array(points, dtype=np.float32)
                    x_min, y_min = np.min(points, axis=0)
                    x_max, y_max = np.max(points, axis=0)

                    # 计算边界框中心和宽高
                    x_center = (x_min + x_max) / 2 / image_width
                    y_center = (y_min + y_max) / 2 / image_height
                    width = (x_max - x_min) / image_width
                    height = (y_max - y_min) / image_height

                    # 确保数值在有效范围内
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))

                    yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # 保存 YOLO 格式文件
            if yolo_lines:
                output_file = self.output_dir / f"{json_file.stem}.txt"
                with open(output_file, "w") as f:
                    f.write("\n".join(yolo_lines))

        except Exception as e:
            print(f"处理 {json_file} 时出错: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="将 Labelme JSON 标注转换为 YOLO 格式")
    parser.add_argument("--input", "-i", required=True, help="包含 Labelme JSON 文件的目录")
    parser.add_argument("--output", "-o", required=True, help="输出 YOLO 格式文件的目录")
    parser.add_argument("--classes", "-c", help="类别映射文件路径")
    args = parser.parse_args()

    # 加载类别映射（如果提供）
    class_map = {}
    if args.classes:
        with open(args.classes, "r") as f:
            classes = [line.strip() for line in f.readlines()]
            class_map = {name: i for i, name in enumerate(classes)}

    # 创建转换器并执行转换
    converter = LabelmeToYoloConverter(args.input, args.output, class_map)
    converter.convert()


# python 2_labelme_json_to_txt.py --input ???/labels/test --output datasets/taozi/labels/test
# python labelme_to_yolo.py --input ???/labels/train --output datasets/taozi/labels/train
# python labelme_to_yolo.py --input ???/labels/val --output datasets/taozi/labels/val
if __name__ == "__main__":
    main()
