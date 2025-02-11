
from ultralytics import YOLO
import roboflow

rf = roboflow.Roboflow(api_key="wLHoSeXIZwBKYqMmpU6t")
project = rf.workspace().project("123-mdbcu-8dhfr")

#can specify weights_filename, default is "weights/best.pt"
version = project.version("123-mdbcu-8dhfr/1")
# version.deploy("model-type", "tinyyolo-custom-weights/", "rocket-weights.pt")

#example1 - directory path is "training1/model1.pt" for yolov8 model
version.deploy("yolov5", "tinyyolo-custom-weights/", "rocket-weights.pt")

#example2 - directory path is "training1/weights/best.pt" for yolov8 model
# version.deploy("yolov5", "training1")