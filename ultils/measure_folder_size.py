import os

# Đường dẫn đến thư mục cha
base_dir = "/home/tuanvu17/mydocuments/ths/luanvan/Drone_Detection_Project"

# Hàm tính kích thước của một thư mục (bao gồm tất cả file và thư mục con)
def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += os.path.getsize(file_path)
    return total_size

# Lấy danh sách tất cả các thư mục con trong thư mục cha
subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir()]

# Đo kích thước và in kết quả
print(f"\nKích thước của các thư mục trong: {base_dir}")
print("-" * 50)
for folder in subfolders:
    size_bytes = get_folder_size(folder)
    size_mb = size_bytes / (1024 * 1024)  # Chuyển sang MB
    folder_name = os.path.basename(folder)
    print(f"Thư mục: {folder_name:20} | Kích thước: {size_mb:.2f} MB | ({size_bytes:,} bytes)")

print("-" * 50)
print(f"Thời gian đo: {os.path.basename(__file__)} chạy vào {time.strftime('%H:%M:%S %d/%m/%Y', time.localtime())}")