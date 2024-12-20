import cv2
from functools import cmp_to_key
from inference_sdk import InferenceHTTPClient
import streamlit as st

CLIENT = InferenceHTTPClient(
    api_url = "https://detect.roboflow.com",
    api_key = "ZEWEc1ZPB3XNtTHuMvf2"
)

MODEL_ID = "kertas-suara-titik/1"
IMAGE_SIZE = (40, 240)

def compare(a, b):
    return 1 if (a['x'] + a['y']) > (b['x'] + b['y']) else -1

def cropping(path):
    img = cv2.imread(path)
    image_path = path
    result = CLIENT.infer(image_path, model_id=MODEL_ID)
    
    if len(result['predictions']) != 9:
        st.error("Please provide a better photo. Expected 9 regions, found " + str(len(result['predictions'])))
        return []
        
    pred = result['predictions']
    pred = sorted(pred, key=cmp_to_key(compare))
    cropped_imgs = []
    
    for res in pred:
        cropped_img = img[int(res['y'] - (res['height'] / 2)): int(res['y'] + (res['height'] / 2)),
                         int(res['x'] - (res['width'] / 2)): int(res['x'] + (res['width'] / 2))]
        cropped_img = cv2.resize(cropped_img, IMAGE_SIZE)
        gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        th = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 2)
        th = th / 255.0
        cropped_imgs.append(th)
    return cropped_imgs