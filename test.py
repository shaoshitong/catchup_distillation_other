import numpy as np
import os,sys

if __name__ == "__main__":
    file = np.load("./checkpoints/admnet_imagenet64.npz")
    print(file["arr_0"].shape)