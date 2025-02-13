import cv2
from ultralytics import YOLO
from inference import get_model
from inference import InferencePipeline
import supervision as sv

# Define a GStreamer pipeline for a CSI camera on NVIDIA Jetson
def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=(int){}, height=(int){}, framerate=(fraction){}/1 ! "
        "nvvidconv flip-method={} ! "
        "video/x-raw, width=(int){}, height=(int){} ! "
        "videoconvert ! appsink".format(
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# Open the camera stream using the GStreamer pipeline.
# If you are using a USB camera instead, you might just call:
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(
                "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1920 height=1080 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink",
                cv2.CAP_GSTREAMER,
            )
if not cap.isOpened():
    print("Failed to open camera.")
    exit(1)

# Load your pre-trained YOLO model
model = get_model(model_id="rocket-wgmja-277le/3", api_key="API_KEY")

# Optionally, you can initialize your annotators once if you wish.
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run inference on the current frame.
    results = model.infer(frame)[0]

    # Convert inference results to detections using the supervision library.
    detections = sv.Detections.from_inference(results)

    # Annotate the frame with bounding boxes and labels.
    annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

    # Display the annotated frame.
    cv2.imshow("Detection", annotated_frame)

    # Press 'q' to exit.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup resources.
cap.release()
cv2.destroyAllWindows()