from ultralytics import YOLO

# Load a pretrained YOLOv11 model
model = YOLO('yolo11n.pt')  # You can choose other model sizes like 'yolov11s.pt' or 'yolov11m.pt'

# Run inference on an image
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
results = model.predict('rocket+launch.jpg')  # Replace 'rocke+launch.jpg' with the path to your image

# Access the results
for result in results:
    boxes = result.boxes  # Bounding boxes
    classes = result.boxes.cls  # Class IDs
    confidences = result.boxes.conf  # Confidence scores

    # Filter for rockets (assuming the class ID for rocket is 5)
    rocket_indices = [i for i, cls in enumerate(classes) if cls == 5]
    rocket_boxes = boxes[rocket_indices]
    rocket_confidences = confidences[rocket_indices]

    # Print the bounding boxes and confidence scores for rockets
    for box, conf in zip(rocket_boxes, rocket_confidences):
        print(f'Rocket detected at {box.xyxy[0]} with confidence {conf:.2f}')
