import cv2
from ultralytics import YOLO
from inference import get_model
from inference import InferencePipeline
import supervision as sv
import time
import Jetson.GPIO as GPIO

PAN_SERVO_PIN = 17  # change to actual PWM-capable GPIO pin
TILT_SERVO_PIN = 27 # change to actual PWM-capable GPIO pin

GPIO.setmode(GPIO.BOARD) # or BCM? adjust pin numbers as needed, not sure what they are

GPIO.setup(PAN_SERVO_PIN, GPIO.OUT)
GPIO.setup(TILT_SERVO_PIN, GPIO.OUT)

pan_pwm = GPIO.PWM(PAN_SERVO_PIN, 50)
tilt_pwm = GPIO.PWM(TILT_SERVO_PIN, 50)

pan_angle = 90
tilt_angle = 90

def angle_to_duty_cycle(angle):
    return 2.5 + (angle / 180.0) * 10 # converting angle to 0-180 range, example

pan_pwm.start(angle_to_duty_cycle(pan_angle))
tilt_pwm.start(angle_to_duty_cycle(tilt_angle))
time.sleep(1)


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
model = get_model(model_id="rocket-wgmja-277le/3", api_key="GGbWyL8M3YlR5nLJGdIn")

# Optionally, you can initialize your annotators once if you wish.
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# control parameters for camera tracking
cp_pan = 0.05       # dependent on calibration
cp_tilt = 0.05      # dependent on calibration

ret, frame = cap.read()
if not ret:
    print("Failed to grab initial frame")
    exit(1)
frame_height, frame_width = frame.shape[:2]
center_x, center_y = frame_width / 2, frame_height / 2

last_pan_adjust = 0
last_tilt_adjust = 0

# speed in degrees per frame
search_pan_speed = 0.5
search_tilt_speed = 0.2

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Run inference on the current frame.
        results = model.infer(frame)[0]

        # Convert inference results to detections using the supervision library.
        detections = sv.Detections.from_inference(results)

        if len(detections.xyxy) > 0:
            # if rocket is detected, resets search mode
            # if multiple are detected, tracks one with highest confidence
            xmin, ymin, xmax, ymax = detections.xyxy[0][:4]

            rocket_center_x = (xmin + xmax) / 2
            rocket_center_y = (ymin + ymax) / 2

            error_x = rocket_center_x - center_x
            error_y = rocket_center_y - center_y

            pan_adjust = cp_pan * error_x
            tilt_adjust = cp_tilt * error_y

            last_pan_adjust = pan_adjust
            last_tilt_adjust = tilt_adjust

            pan_angle -= pan_adjust
            tilt_angle -= tilt_adjust

        else:
            # if it loses the rocket, it will continue to search in the last known direction
            if last_pan_adjust == 0:
                last_pan_adjust = search_pan_speed
            if last_tilt_adjust == 0:
                last_tilt_adjust = search_tilt_speed
            
            pan_angle -= last_pan_adjust
            tilt_angle -= last_tilt_adjust

        pan_angle = max(0, min(180, pan_angle))
        tilt_angle = max(0, min(180, tilt_angle))

        pan_pwm.ChangeDutyCycle(angle_to_duty_cycle(pan_angle))
        tilt_pwm.ChangeDutyCycle(angle_to_duty_cycle(tilt_angle))
        
        # would print to check current position
        # print(f"Pan: {pan_angle:.2f}, Tilt: {tilt_angle:.2f}")

        # Annotate the frame with bounding boxes and labels.
        annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

        # Display the annotated frame.
        cv2.imshow("Detection", annotated_frame)

        # Press 'q' to exit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# cleanup
finally:
    cap.release()
    cv2.destroyAllWindows()
    pan_pwm.stop()
    tilt_pwm.stop()
    GPIO.cleanup()