# import os

# # 你需要修改这个路径为你想要修改的文件夹的路径
# folder_path = "/home/project/fast-ode/runs/cifar10-onlineslim-predstep-2-uniform-shakedrop0.75-beta20/test_8/samples"

# for filename in os.listdir(folder_path):
#     # 分离文件名和扩展名
#     file_base_name, file_extension = os.path.splitext(filename)
    
#     # 检查文件名是否为数字
#     if file_base_name.isdigit():
#         new_file_base_name = f"f{int(file_base_name)%1000}_f{int(file_base_name)//1000}"
#         new_file_name = new_file_base_name + file_extension

#         # 创建原文件和新文件的完整路径
#         old_file_path = os.path.join(folder_path, filename)
#         new_file_path = os.path.join(folder_path, new_file_name)

#         # 重命名文件
#         os.rename(old_file_path, new_file_path)


import os
import shutil

# 定义你的目标文件夹
parent_dir = '/home/imagenet/train'

# 遍历每一个子文件夹
for folder_name in os.listdir(parent_dir):
    # 检查是否为文件夹
    if os.path.isdir(os.path.join(parent_dir, folder_name)):
        # 获取子文件夹内所有图片文件
        for image_name in os.listdir(os.path.join(parent_dir, folder_name)):
            # 确保只处理图片文件
            if image_name.endswith('.JPEG') or image_name.endswith('.jpg'):
                # 构建新的图片名
                new_image_name = f"{image_name}"
                # 创建原图片文件的完整路径
                old_image_path = os.path.join(parent_dir, folder_name, image_name)
                # 创建新图片文件的完整路径
                new_image_path = os.path.join(parent_dir, new_image_name)
                # 移动（同时也是重命名）图片文件
                shutil.move(old_image_path, new_image_path)

print("All images have been renamed successfully.")
