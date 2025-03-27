from ultralytics import YOLO

# path
DATASET_PATH = r"E:\FAU\Nur Stillstand\merged_dataset\dataset.yaml"

# YOLOv8n initial
print("start ...")
model = YOLO('yolov8n.pt')  

# CPU
results = model.train(
    data=DATASET_PATH,
    epochs=15,          
    imgsz=320,           
    batch=8,            
    device='cpu',        
    workers=1,           
    patience=50,         
    save=True,           
    project="yolov8_train",  
    name="run3"          
)

print("finish！")
print(f"model: {model.export()}")

# validation
print("process validation...")
metrics = model.val()
print(f"results: {metrics}")
