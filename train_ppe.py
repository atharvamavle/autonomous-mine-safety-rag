from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  # base COCO model

    results = model.train(
        data="data/mining_ppe/mining_ppe.yaml",
        imgsz=640,
        epochs=50,
        batch=8,
        name="mining_ppe_yolov8n",
        project="runs/mining_ppe",
    )

    print("Training finished. Best weights folder:", results.save_dir)

if __name__ == "__main__":
    main()
