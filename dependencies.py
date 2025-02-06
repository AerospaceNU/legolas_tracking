from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="wLHoSeXIZwBKYqMmpU6t"
)

result = CLIENT.infer('rocketlaunch.jpg', model_id="rockets-rockets-flying/16")