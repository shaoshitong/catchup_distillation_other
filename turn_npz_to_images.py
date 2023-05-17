import os

import numpy as np
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='Turn NPZ to IMAGE')
parser.add_argument('--npz_path', default='/tmp/openai-2023-05-17-19-17-04-324341/samples_500x64x64x3.npz', type=str)
parser.add_argument('--save_path', default='./images/', type=str)
arg = parser.parse_args()

def turn_npz_to_image(npz_path,save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    arr = np.load(npz_path)
    arr = arr["arr_0"]
    print(arr.max(),arr.min())
    arr = arr.astype(np.uint8)
    for i in range(arr.shape[0]):
        image = arr[i]
        if image.shape[2] == 1:
            si = Image.fromarray(image.squeeze(),"L")
        else:
            si = Image.fromarray(image,"RGB")
        save_image_path = os.path.join(save_path,f"{i}.png")
        si.save(save_image_path)

if __name__ == "__main__":

    turn_npz_to_image(arg.npz_path,arg.save_path)
