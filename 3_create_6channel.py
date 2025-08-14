from ultralytics import YOLO
import torch.nn as nn


def create_six_channel_model(pretrained_weights, save_path):
    """
    创建支持六通道输入的 YOLOv8 模型
    参数:
        pretrained_weights: 预训练模型路径
        save_path: 修改后的模型保存路径
    """
    # 加载预训练模型
    model = YOLO(pretrained_weights)

    # 获取模型的骨干网络
    backbone = model.model.model[:10]  # 注意这里的索引可能因版本而异

    # 修改第一层卷积以接受6通道输入
    first_conv = backbone[0].conv
    new_first_conv = nn.Conv2d(
        in_channels=6,  # 六通道输入
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias
    )

    # 复制原始权重（前3通道）
    new_first_conv.weight.data[:, :3, :, :] = first_conv.weight.data.clone()
    # 复制前3通道的权重到后3通道
    new_first_conv.weight.data[:, 3:, :, :] = first_conv.weight.data.clone()

    # 替换模型中的第一层
    backbone[0] = new_first_conv

    # 保存修改后的模型
    model.save(save_path)
    print(f"6 channels模型已保存至: {save_path}")

    return model


if __name__ == "__main__":
    create_six_channel_model('yolov8n.pt', 'yolov8n_6channel.pt')
