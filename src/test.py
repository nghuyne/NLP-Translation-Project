import tarfile

file_path = "training-parallel-un.tgz"
output_dir = "./"  # Thư mục giải nén

with tarfile.open(file_path, "r:gz") as tar:
    tar.extractall(path=output_dir)
    print("Đã giải nén thành công")
