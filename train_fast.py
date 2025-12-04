from ultralytics import YOLO
model = YOLO('/home/pablodlx/TacticEYE/runs/soccernet_800_b8_OK/weights/last.pt')
model.train(
    data='/home/pablodlx/TacticEYE/datasets/yolo_soccernet/dataset.yaml',
    epochs=200,
    batch=8,
    workers=4,
    imgsz=800,
    amp=False,
    project='/home/pablodlx/TacticEYE/runs',
    name='soccernet_800_compile',
    exist_ok=True,
    compile=True  # ← esto sí funciona en Python API
)
