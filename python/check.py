import os

def check_dataset_files(data_dir):
    # 使用 os.path.join 来确保路径正确
    image_file = os.path.join(data_dir, "train-images.idx3-ubyte")
    label_file = os.path.join(data_dir, "train-labels.idx1-ubyte")
    validation_image_file = os.path.join(data_dir, "t10k-images.idx3-ubyte")
    validation_label_file = os.path.join(data_dir, "t10k-labels.idx1-ubyte")

    # 检查每个文件是否存在
    if not os.path.exists(image_file):
        print(f"Error: Image file not found: {image_file}")
    else:
        print(f"Image file found: {image_file}")

    if not os.path.exists(label_file):
        print(f"Error: Label file not found: {label_file}")
    else:
        print(f"Label file found: {label_file}")

    if not os.path.exists(validation_image_file):
        print(f"Error: Validation image file not found: {validation_image_file}")
    else:
        print(f"Validation image file found: {validation_image_file}")

    if not os.path.exists(validation_label_file):
        print(f"Error: Validation label file not found: {validation_label_file}")
    else:
        print(f"Validation label file found: {validation_label_file}")

if __name__ == "__main__":
    # 使用绝对路径来检查文件
    data_dir = "D:/app/pycharm/python project/mathematical modeling/BMCNNwHFCs/data"
    check_dataset_files(data_dir)
