from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes

pipeline = InferencePipeline.init(
    model_id="rocket-wgmja-277le/3", # Roboflow model to use | rockets-rockets-flying/16
    api_key="API_KEY",
    video_reference="rocket1.MOV", # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
                                   # "rtsp://username:password"
    on_prediction=render_boxes, # Function to run after each prediction
)
pipeline.start()
pipeline.join()