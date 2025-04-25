import cv2
from ultralytics import YOLO

from .object_tracker import ObjectTracker
from .tracked_object import BoundingBox, DetectedObject, Point2D, TrackerObject
from .vision_model import VisionModel


def yolo_output_to_detected_objects(results, class_names) -> list[DetectedObject]:
    detected_objects = []

    for result in results:
        for box in result.boxes:
            # Extract box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            bbox = BoundingBox(Point2D(x1, y1), Point2D(x2, y2))
            detected_objects.append(
                DetectedObject(
                    class_name=class_names[class_id],
                    confidence=conf,
                    center=Point2D(center_x, center_y),
                    bbox=bbox,
                )
            )

    return detected_objects


class YoloModel(VisionModel):

    def __init__(self, object_tracker: ObjectTracker, yolo_file: str = "yolo11m.pt"):
        super().__init__(object_tracker)
        self.model = YOLO(yolo_file)
        self.class_names = self.model.names

    def _process_frame(self, frame: cv2.typing.MatLike) -> list[DetectedObject]:
        results = self.model(frame, conf=0.1)
        return yolo_output_to_detected_objects(results, self.class_names)
