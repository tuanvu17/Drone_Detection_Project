# Mô Tả Các Bộ Dữ Liệu Âm Thanh

## 1. Bộ Dữ Liệu Gốc (Original Audio Dataset)

### 1.1. Nguồn Gốc
- **Nguồn**: Svanström et al. dataset
- **Vị trí**: `data/raw/Audio_Original/`
- **Định dạng**: File audio WAV

### 1.2. Cấu Trúc và Phân Bố
Bộ dữ liệu gồm **90 clip audio** được tổ chức thành 3 lớp:

| Lớp | Số lượng file | Thư mục |
|-----|---------------|---------|
| **DRONE** | 30 files | `data/raw/Audio_Original/DRONE/` |
| **HELICOPTER** | 30 files | `data/raw/Audio_Original/HELICOPTER/` |
| **BACKGROUND** | 30 files | `data/raw/Audio_Original/BACKGROUND/` |
| **TỔNG CỘNG** | **90 files** | |

### 1.3. Đặc Điểm Kỹ Thuật
- **Định dạng file**: `.wav`
- **Kích thước file**: Khoảng 1.7MB mỗi file (tương đương khoảng 10 giây audio ở tần số lấy mẫu 44.1kHz)
- **Tần số lấy mẫu (Sample Rate)**: 44,100 Hz (theo cấu hình `AUDIO_SAMPLE_RATE = 44100`)
- **Quy ước đặt tên**: 
  - DRONE: `DRONE_001.wav`, `DRONE_002.wav`, ..., `DRONE_030.wav`
  - HELICOPTER: `HELICOPTER_001.wav`, `HELICOPTER_002.wav`, ..., `HELICOPTER_030.wav`
  - BACKGROUND: `BACKGROUND_001.wav`, `BACKGROUND_002.wav`, ..., `BACKGROUND_030.wav`

### 1.4. Cách Sử Dụng
- **Mục đích**: Huấn luyện mô hình audio cơ bản (baseline model)
- **Chia dữ liệu**:
  - Test set: 30% (5 file mỗi lớp = 15 files)
  - Train/Validation: 70% còn lại (25 file mỗi lớp = 75 files)
    - Validation: 20% của 70% (5 file mỗi lớp = 15 files)
    - Train: 80% của 70% (20 file mỗi lớp = 60 files)
- **Xử lý**: Mỗi file audio được chia thành các segment 1 giây để trích xuất MFCC features

---

## 2. Bộ Dữ Liệu Thu Thập từ YouTube ("YT-InDomain")

### 2.1. Quy Trình Thu Thập

#### Bước 1: Thu thập Video
- **Nguồn**: Video từ YouTube
- **Vị trí lưu trữ**: `data/raw/YouTube_Videos_Raw/`
- **Định dạng**: MP4, AVI, MOV, MKV
- **Tổ chức**: Video được phân loại theo lớp trong các thư mục con

#### Bước 2: Trích Xuất Audio
- Sử dụng thư viện `moviepy` để trích xuất track âm thanh từ video
- Lưu tạm vào `data/interim/YouTube_Audio_Extracted/`
- Tần số lấy mẫu chuẩn hóa: **44,100 Hz**

#### Bước 3: Chia Thành Segments
- Mỗi video được chia thành các **segment 1 giây**
- Mỗi segment được lưu dưới dạng file `.wav` riêng biệt
- Vị trí lưu: `data/processed/Audio_YouTube_FineTune_Segments/`
- Quy ước đặt tên: `{video_name}_seg{000000}.wav`

#### Bước 4: Tạo Metadata
- File CSV metadata: `audio_youtube_finetune_metadata.csv`
- Chứa thông tin: `segment_id`, `original_video`, `audio_segment_file`, `label`

### 2.2. Đặc Điểm Bộ Dữ Liệu

#### 2.2.1. Phân Bố Theo Lớp

| Lớp | Số segments | Tỷ lệ (%) | Số video gốc |
|-----|------------|-----------|--------------|
| **HELICOPTER** | 2,352 | 58.43% | 9 videos |
| **BACKGROUND** | 1,200 | 29.81% | 1 video |
| **DRONE** | 473 | 11.76% | 10 videos |
| **TỔNG CỘNG** | **4,025 segments** | **100%** | **20 videos** |

#### 2.2.2. Phân Bố Chi Tiết Theo Video

**DRONE (10 videos, 473 segments):**
- Drone 09: 116 segments
- Drone 10: 102 segments
- Drone 11: 96 segments
- Drone 12: 42 segments
- Drone 6: 65 segments
- Drone 7: 52 segments
- (và 4 video khác)

**HELICOPTER (9 videos, 2,352 segments):**
- Helicopter_3: 1,166 segments (video dài nhất)
- Helicopter_3_1: 416 segments
- Helicopter_3_2: 224 segments
- Helicopter 6 - 1: 353 segments
- Helicopter 6 - 2: 102 segments
- (và 4 video khác)

**BACKGROUND (1 video, 1,200 segments):**
- Background_1: 1,200 segments (khoảng 20 phút audio)

### 2.3. Đặc Điểm Kỹ Thuật

- **Độ dài mỗi segment**: 1.0 giây (cố định)
- **Tần số lấy mẫu**: 44,100 Hz
- **Định dạng**: WAV (PCM, 16-bit)
- **Tổng thời lượng audio**: 
  - HELICOPTER: ~39.2 phút (2,352 giây)
  - BACKGROUND: ~20 phút (1,200 giây)
  - DRONE: ~7.9 phút (473 giây)
  - **TỔNG**: ~**67.1 phút** (~1.12 giờ)

### 2.4. Cách Sử Dụng

- **Mục đích**: Fine-tuning mô hình audio đã được huấn luyện trên dữ liệu gốc
- **Chia dữ liệu**:
  - Test In-Domain: 30% (1,207 segments)
  - Train/Validation: 70% còn lại (2,818 segments)
    - Validation: 20% của 70% (~564 segments)
    - Train: 80% của 70% (~2,254 segments)
- **Xử lý**: 
  - Trích xuất MFCC features (13 coefficients)
  - Padding và scaling sử dụng scaler và max_len được tính từ tập train/val
  - Fine-tuning 2 giai đoạn:
    - Stage 1: Đóng băng các lớp gốc, chỉ train lớp output mới
    - Stage 2: Mở băng và fine-tune toàn bộ mô hình

### 2.5. Lưu Ý Quan Trọng

⚠️ **Vấn đề về Phân Bố Dữ Liệu:**
- Bộ dữ liệu YouTube có phân bố không đều:
  - HELICOPTER chiếm hơn 58% dữ liệu
  - DRONE chỉ chiếm ~12% dữ liệu
- Điều này có thể gây ra class imbalance trong quá trình fine-tuning

⚠️ **Thiếu Dữ Liệu cho Một Số Lớp:**
- Trong config định nghĩa 5 lớp: `['AIRPLANE', 'BIRD', 'DRONE', 'HELICOPTER', 'BACKGROUND']`
- Nhưng dữ liệu YouTube chỉ có 3 lớp: DRONE, HELICOPTER, BACKGROUND
- AIRPLANE và BIRD không có dữ liệu (0 segments)

---

## 3. So Sánh Hai Bộ Dữ Liệu

| Đặc điểm | Audio Gốc (Svanström) | YouTube Audio |
|----------|------------------------|---------------|
| **Số lượng** | 90 files | 4,025 segments |
| **Lớp** | 3 lớp (DRONE, HELICOPTER, BACKGROUND) | 3 lớp (DRONE, HELICOPTER, BACKGROUND) |
| **Độ dài trung bình** | ~10 giây/file | 1 giây/segment |
| **Tổng thời lượng** | ~15 phút | ~67 phút |
| **Nguồn** | Dataset công bố (Svanström et al.) | Thu thập từ YouTube |
| **Mục đích** | Huấn luyện mô hình baseline | Fine-tuning mô hình |
| **Phân bố** | Cân bằng (30 files/lớp) | Không cân bằng (58% HELICOPTER) |

---

## 4. Thống Kê Tổng Hợp

### 4.1. Tổng Số Lượng Dữ Liệu
- **Audio gốc**: 90 files (~15 phút)
- **YouTube audio**: 4,025 segments (~67 phút)
- **TỔNG**: ~82 phút audio

### 4.2. Phân Bố Tổng Theo Lớp

| Lớp | Audio Gốc | YouTube | Tổng |
|-----|-----------|--------|------|
| DRONE | 30 files | 473 segments | 30 files + 473 segments |
| HELICOPTER | 30 files | 2,352 segments | 30 files + 2,352 segments |
| BACKGROUND | 30 files | 1,200 segments | 30 files + 1,200 segments |

---

*Tài liệu được tạo tự động từ phân tích dữ liệu trong dự án Drone Detection Project*


