import gzip
import shutil
import os

def decompress_gz_files():
    # 当前脚本所在目录
    data_dir = os.path.dirname(os.path.abspath(__file__))

    # 遍历当前目录下所有 .gz 文件
    for filename in os.listdir(data_dir):
        if filename.endswith('.gz'):
            gz_path = os.path.join(data_dir, filename)
            output_path = os.path.join(data_dir, filename[:-3])  # 去掉 .gz 后缀

            # 如果已经解压过，就跳过
            if os.path.exists(output_path):
                print(f"File {output_path} already exists, skipping...")
                continue

            print(f"Decompressing {gz_path} to {output_path}...")
            with gzip.open(gz_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

    print("✅ All files decompressed successfully!")

if __name__ == "__main__":
    decompress_gz_files()
