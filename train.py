from ultralytics import YOLO


model = YOLO("yolo11m.pt")
model.train(
    data="MS_Shift.yaml",  # Path to the dataset configuration file
    epochs=150,
    imgsz=224,
    batch=128,
    name="",  # Name of the training run (subdirectory within project).
)
