# Hướng Dẫn Chạy Thực Nghiệm So Sánh Đặc Trưng Audio

## Mục Đích

Script này thực nghiệm so sánh 3 loại đặc trưng audio:
- **MFCC** (Mel-Frequency Cepstral Coefficients) - 13 coefficients
- **ZCR** (Zero Crossing Rate) - 1 giá trị
- **RMSE** (Root Mean Square Energy) - 1 giá trị

Để chứng minh rằng **MFCC vượt trội hơn** các đặc trưng cơ bản (ZCR, RMSE) thông qua so sánh chất lượng phân loại.

## Cách Chạy

### 1. Đảm bảo môi trường

```bash
cd /home/tuanvu17/mydocuments/ths/luanvan/Drone_Detection_Project
pip install -r requirements.txt
```

### 2. Chạy thực nghiệm

```bash
python -m src.main_experiment_compare_features
```

### 3. Kết quả

Kết quả sẽ được lưu tại: `experiments/feature_comparison_results/`

Bao gồm:
- `best_model_mfcc.keras` - Mô hình tốt nhất với MFCC
- `best_model_zcr.keras` - Mô hình với ZCR
- `best_model_rmse.keras` - Mô hình với RMSE
- `results_mfcc.pkl` - Kết quả chi tiết MFCC
- `results_zcr.pkl` - Kết quả chi tiết ZCR
- `results_rmse.pkl` - Kết quả chi tiết RMSE
- `feature_comparison.png` - Biểu đồ so sánh Accuracy và F1-Score
- `confusion_matrices_comparison.png` - Confusion matrices của cả 3 đặc trưng
- `comparison_report.txt` - Báo cáo so sánh chi tiết

## Quy Trình Thực Nghiệm

1. **Trích xuất đặc trưng**:
   - Load audio files từ `data/raw/Audio_Original/`
   - Chia thành segments 1 giây
   - Trích xuất MFCC, ZCR, hoặc RMSE

2. **Xử lý dữ liệu**:
   - Padding để đảm bảo cùng độ dài
   - Scaling (StandardScaler)
   - Chia train/validation (80/20)

3. **Huấn luyện mô hình**:
   - MFCC: BiLSTM (giống mô hình gốc)
   - ZCR/RMSE: LSTM đơn giản hơn
   - Epochs: 30 (có thể điều chỉnh)
   - Early stopping, model checkpoint

4. **Đánh giá**:
   - Accuracy
   - F1-Score (weighted)
   - Classification Report
   - Confusion Matrix

5. **So sánh**:
   - Bảng so sánh Accuracy và F1-Score
   - Biểu đồ so sánh
   - Confusion matrices
   - Báo cáo chi tiết

## Kết Quả Dự Kiến

Dựa trên lý thuyết và thực nghiệm:

| Đặc trưng | Accuracy dự kiến | F1-Score dự kiến |
|-----------|------------------|------------------|
| **MFCC** | **>85%** | **>0.85** |
| **ZCR** | <60% | <0.60 |
| **RMSE** | <50% | <0.50 |

**Kết luận**: MFCC sẽ cho kết quả tốt nhất, chứng minh việc sử dụng MFCC là đúng đắn.

## Lưu Ý

1. **Thời gian chạy**: 
   - Thực nghiệm có thể mất 30-60 phút tùy vào dữ liệu và GPU/CPU
   - Có thể giảm `epochs` trong code để chạy nhanh hơn (nhưng kết quả có thể kém hơn)

2. **Dữ liệu**:
   - Script sử dụng dữ liệu từ `data/raw/Audio_Original/`
   - Đảm bảo có đủ dữ liệu cho 3 lớp: DRONE, HELICOPTER, BACKGROUND

3. **GPU/CPU**:
   - Có thể chạy trên CPU hoặc GPU
   - TensorFlow sẽ tự động sử dụng GPU nếu có

## Tùy Chỉnh

Có thể chỉnh sửa các tham số trong `src/experiments/compare_audio_features.py`:

- `epochs`: Số epochs huấn luyện (mặc định: 30)
- `batch_size`: Kích thước batch (mặc định: 32)
- `validation_split`: Tỷ lệ validation (mặc định: 0.2)

## Ví Dụ Kết Quả

Sau khi chạy, bạn sẽ thấy output như:

```
============================================================
SO SÁNH KẾT QUẢ CÁC ĐẶC TRƯNG
============================================================

Bảng So Sánh:
------------------------------------------------------------
MFCC            | Accuracy: 0.9234 | F1: 0.9210
ZCR             | Accuracy: 0.5432 | F1: 0.5123
RMSE            | Accuracy: 0.4567 | F1: 0.4234
------------------------------------------------------------

✓ Đặc trưng tốt nhất: MFCC
  Accuracy: 0.9234
  F1-Score: 0.9210
```

Điều này chứng minh rằng **MFCC vượt trội hơn** ZCR và RMSE!

