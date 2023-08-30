import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

# Load the model checkpoint
model_path = r"E:\API\Yolov8\ultralytics\runs\detect\train2\weights\best.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(model_path, map_location=device)
model = checkpoint['model'].to(device, dtype=torch.float32)  # Set model dtype to float32
model.eval()

# Preprocess the image
image_path = r"E:\API\Yolov8\ultralytics\dog.jpeg"
output_image_path = r"E:\API\Yolov8\ultralytics\output.jpeg"  # Replace with the desired output path
image = Image.open(image_path)
image = Image.open(image_path)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_image = transform(image).unsqueeze(0).to(device, dtype=torch.float16)  # Set input dtype to float16

# Perform inference
with torch.no_grad():
    output = model(input_image)

# Process the output
predicted_class = torch.argmax(output).item()
print(f'Predicted class index: {predicted_class}')

# Save the processed image with predicted class label
image_with_label = image.copy()
draw = ImageDraw.Draw(image_with_label)
label = f'Predicted Class: {predicted_class}'
draw.text((10, 10), label, fill=(255, 0, 0))
image_with_label.save(output_image_path)
