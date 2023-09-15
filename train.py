from ultralytics import YOLO
import torch
from multiprocessing import shared_memory

shared_memory.__loader__

if torch.cuda.is_available(): 
    print("Gpu is Ready and Training on GPU")
else: 
    print('Training On CPU...>>>')

model = YOLO("yolov8n.pt")  

# for CUDA Training
# model.train(data=r"E:\API\Yolov8\ultralytics\data\data.yaml", epochs=10,imgsz=832) # for CPU Training



model.train(data=r"E:\API\Yolov8\ultralytics\data\data.yaml", epochs=10,imgsz=832, device=0)