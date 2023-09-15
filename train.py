from ultralytics import YOLO
import torch


if torch.cuda.is_available(): 
    print("Gpu is Ready and Training on GPU")
else: 
    print('Training On CPU...>>>')

model = YOLO("yolov8n.pt")  

# for CUDA Training
# model.train(data=r"E:\API\Yolov8\ultralytics\data\data.yaml", epochs=10,imgsz=832) # for CPU Training

if __name__ == '__main__':    
    model.train(data=r"E:\API\Yolov8\ultralytics\data\data.yaml", epochs=100,imgsz=832, device=0)