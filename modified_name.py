import os

# 你需要修改这个路径为你想要修改的文件夹的路径
folder_path = "/home/project/fast-ode/runs/cifar10-onlineslim-predstep-2-uniform-shakedrop0.75-beta20/test_8/samples"

for filename in os.listdir(folder_path):
    # 分离文件名和扩展名
    file_base_name, file_extension = os.path.splitext(filename)
    
    # 检查文件名是否为数字
    if file_base_name.isdigit():
        new_file_base_name = f"f{int(file_base_name)%1000}_f{int(file_base_name)//1000}"
        new_file_name = new_file_base_name + file_extension

        # 创建原文件和新文件的完整路径
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, new_file_name)

        # 重命名文件
        os.rename(old_file_path, new_file_path)
