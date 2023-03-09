import cv2
import base64
import numpy as np
from scripts.processor import midas_depth
from numpy.typing import NDArray

def crop_and_resize(img: cv2.Mat):
    # Get the dimensions of the image
    h, w = img.shape[:2]

    # Find the smaller dimension and crop to a square
    if h < w:
        square_size = h
        y_start = 0
        x_start = (w - square_size) // 2
    else:
        square_size = w
        x_start = 0
        y_start = (h - square_size) // 2

    cropped_img = img[y_start:y_start+square_size, x_start:x_start+square_size]

    # Resize the image to 512x512 pixels
    resized_img = cv2.resize(cropped_img, (512, 512))

    return resized_img

def midas_embeddings(b64: str) -> list:
    nparr = np.frombuffer(base64.b64decode(b64), np.uint8)
    img: cv2.Mat = cv2.imdecode(nparr, flags=cv2.IMREAD_COLOR)
    cropped = crop_and_resize(img)
    _, _, depth = midas_depth(cropped, 64)
    vec: NDArray = depth.squeeze().cpu().numpy()
    return vec.tolist()