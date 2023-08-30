from ultralytics import YOLO
import os
from tqdm import tqdm

# Load a model
model = YOLO(r"E:\API\Yolov8\ultralytics\runs\detect\Homeec_Epoch-10\weights\last.pt") 

# predict on an image-folder
directory  = r"E:\API\Yolov8\ultralytics\inference" 

dir = [os.path.abspath(os.path.join(directory, p)) for p in os.listdir(directory)]
print("Total Images:",len(dir))

for i in tqdm((dir)):
    results = model(source = f"{i}", save=True ) 
    

print("COMPLETED...")