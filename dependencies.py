from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="API_KEY"
)

result = CLIENT.infer('rocketlaunch.jpg', model_id="rocket-wgmja-277le/3")  # rockets-rockets-flying/16