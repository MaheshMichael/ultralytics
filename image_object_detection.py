import cv2
import os
from ultralytics import YOLO
# from imread_from_url import imread_from_url


# Initialize yolov8 object detector
model_path = r"E:\API\Yolov8\ultralytics\runs\detect\train2\weights\best.pt"
yolov8_detector = YOLO(model_path)

# Read image
# img_url = r"E:\API\Yolov8\ultralytics\dog.jpeg"
# img = imread_from_url(img_url)
image_list  = os.listdir(r"E:\API\Yolov8\datasets\coco128\images\train2017")
for i in image_list:
    img = cv2.imread(f'r{i}')

# Detect Objects
    results = yolov8_detector(img)
    # results.save(f'{i.filename}')

# Draw detections
# combined_img = yolov8_detector.draw_detections(img)
# cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
# cv2.imshow("Detected Objects", combined_img)
# cv2.imwrite("doc/img/detected_objects.jpg", combined_img)
# cv2.waitKey(0)
