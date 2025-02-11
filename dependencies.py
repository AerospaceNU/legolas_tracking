from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="GGbWyL8M3YlR5nLJGdIn"
)

result = CLIENT.infer('rocketlaunch.jpg', model_id="rocket-wgmja-277le/3")  # rockets-rockets-flying/16