from ultralytics import YOLO


def main():
    # model = YOLO("yolov8n.pt")  # 官方默认的
    model = YOLO("yolov8n_6channel.pt")
    # model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

    # yolo train model=yolov8n_6channel.pt data=ultralytics/cfg/datasets/taozi.yaml device=0 epochs=100 imgsz=1088
    model.train(
        # data="ultralytics/cfg/datasets/VOC.yaml",
        data="ultralytics/cfg/datasets/taozi.yaml",
        epochs=100,
        imgsz=1088,
        batch=4,
        device=0,
        # cache=True,
        # pretrained=False
    )

if __name__ == '__main__':
    main()
