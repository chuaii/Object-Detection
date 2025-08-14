from ultralytics import YOLO


def main():
    # 创建基于配置文件的模型
    # model = YOLO("yolov8n.pt")
    model = YOLO("runs/detect/train31/weights/best.pt")  # load my model

    # yolo task=detect mode=predict model=runs/detect/train31/weights/best.pt show=True source=img_video_source/taozi.mp4
    model.predict(
        source="img_video_source/taozi7.mp4",
        show=True
    )

if __name__ == '__main__':
    main()
