from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 segmentation model
model = YOLO('yolov8m-seg.pt')  # or yolov8n-seg.pt for lightweight

# Load image
image_path = "D:\overlapping object cropping\sample.png"
image = cv2.imread(image_path)

# Run detection
results = model(image)

# Display detected classes
for r in results:
    for c in r.boxes.cls:
        print(model.names[int(c)])
# Extract masks and boxes
masks = results[0].masks.data.cpu().numpy()
boxes = results[0].boxes
names = model.names

# Identify masks for face and hand
primary_label = 'face'
occluder_label = 'hand'

primary_mask = None
occluder_mask = None

for i, c in enumerate(boxes.cls):
    label = names[int(c)]
    if label == primary_label:
        primary_mask = masks[i]
    elif label == occluder_label:
        occluder_mask = masks[i]

if primary_mask is None:
    raise Exception("Primary object not detected!")

# Handle overlap
if occluder_mask is not None:
    final_mask = cv2.subtract(primary_mask, occluder_mask)
else:
    final_mask = primary_mask

# Convert mask to crop area
y, x = np.where(final_mask > 0)
cropped = image[np.min(y):np.max(y), np.min(x):np.max(x)]

cv2.imwrite("cropped_primary.png", cropped)
print("✅ Cropped image saved: D:\overlapping object cropping\cropped_primary.png")
