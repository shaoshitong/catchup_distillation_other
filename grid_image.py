from PIL import Image
import os
from numpy import random
def merge_images(folder_path, output_path, n, m):
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
    indices = list(range(len(image_files))) # indices = the number of images in the source data set
    random.shuffle(indices)
    image_files = [image_files[i] for i in indices]
    image_files = image_files[:n * m]

    if len(image_files) < n * m:
        print("Warning: Not enough images found. Using available images.")

    images = [Image.open(os.path.join(folder_path, img_file)) for img_file in image_files]

    # 获取最大宽度和高度
    max_width = max([img.width for img in images])
    max_height = max([img.height for img in images])

    # 创建一个空白图片作为拼接结果
    result = Image.new('RGB', ((max_width+2) * n, (max_height+2) * m))

    # 按行列拼接图片
    for i in range(m):
        for j in range(n):
            index = i * n + j
            if index < len(images):
                img = images[index]
                result.paste(img, (j * (max_width+2), i * (max_height+2)))

    # 保存拼接后的图片
    result.save(output_path)

if __name__ == "__main__":
    merge_images("./images", "./rk45.jpg", 18, 6)