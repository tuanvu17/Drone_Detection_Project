# Quy Trình Triển Khai Trích Xuất Đặc Trưng MFCC

## 4.2. Quy Trình Triển Khai Trích Xuất Đặc Trưng MFCC

### 4.2.1. Công Cụ và Thư Viện Sử Dụng

#### Ngôn Ngữ Lập Trình
- **Python 3.x**: Ngôn ngữ lập trình chính được sử dụng trong toàn bộ dự án

#### Thư Viện Chính

1. **Librosa** (`librosa`)
   - **Mục đích**: Xử lý và phân tích tín hiệu âm thanh
   - **Chức năng chính**: 
     - Trích xuất đặc trưng MFCC (Mel-Frequency Cepstral Coefficients)
     - Tải và xử lý file audio
     - Resampling audio về tần số lấy mẫu mong muốn
   - **Phiên bản**: Sử dụng `librosa.feature.mfcc()` và `librosa.load()`

2. **NumPy** (`numpy`)
   - **Mục đích**: Xử lý mảng đa chiều và các phép toán số học
   - **Chức năng**: 
     - Lưu trữ và thao tác với dữ liệu MFCC (dạng mảng numpy)
     - Thực hiện padding (đệm) cho các feature có độ dài khác nhau
     - Reshape dữ liệu cho quá trình scaling

3. **Scikit-learn** (`sklearn.preprocessing`)
   - **Mục đích**: Tiền xử lý dữ liệu
   - **Chức năng**: 
     - `StandardScaler`: Chuẩn hóa dữ liệu (zero mean, unit variance)
     - `LabelEncoder`: Mã hóa nhãn văn bản thành số

4. **SoundFile** (`soundfile`)
   - **Mục đích**: Đọc/ghi file audio (hỗ trợ cho librosa trong một số trường hợp)

---

### 4.2.2. Các Tham Số Cụ Thể Đã Chọn

Các tham số được định nghĩa trong `config/project_config.py` và được sử dụng nhất quán trong toàn bộ pipeline:

#### Tham Số Cơ Bản

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| **Sample Rate (sr)** | **44,100 Hz** | Tần số lấy mẫu chuẩn cho tất cả audio. Đây là tần số chuẩn cho audio chất lượng CD, đảm bảo chất lượng âm thanh tốt và tương thích với hầu hết các thiết bị. |
| **Segment Duration** | **1.0 giây** | Độ dài mỗi segment audio được chia nhỏ để trích xuất đặc trưng. Mỗi segment 1 giây cung cấp đủ thông tin để phân loại trong khi vẫn giữ được tính nhất quán. |

#### Tham Số Trích Xuất MFCC

| Tham số | Giá trị | Công thức | Mô tả |
|---------|---------|-----------|-------|
| **n_mfcc** | **13** | - | Số lượng hệ số MFCC được trích xuất. 13 là giá trị phổ biến trong xử lý tiếng nói và âm thanh, cung cấp đủ thông tin về phổ tần số âm thanh mà không quá phức tạp. |
| **n_fft** | **1,102** | `int(sample_rate * 0.025)` = `int(44100 * 0.025)` | Kích thước cửa sổ FFT (Fast Fourier Transform). Tương đương với 25ms của tín hiệu audio, đây là độ dài cửa sổ phù hợp để phân tích tần số trong miền thời gian ngắn. |
| **hop_length** | **441** | `int(sample_rate * 0.010)` = `int(44100 * 0.010)` | Khoảng cách giữa các cửa sổ FFT liên tiếp. Tương đương với 10ms, tạo ra độ phân giải thời gian tốt và đảm bảo có đủ overlap giữa các cửa sổ (overlap = 60%). |
| **window** | **'hamming'** | - | Loại cửa sổ được sử dụng trong FFT. Cửa sổ Hamming giúp giảm hiện tượng rò rỉ phổ (spectral leakage) và tạo ra phổ tần số mượt mà hơn so với cửa sổ hình chữ nhật. |
| **center** | **False** | - | Không căn giữa cửa sổ. Điều này phù hợp với việc xử lý các segment audio đã được chia sẵn, tránh việc thêm padding không cần thiết. |

#### Giải Thích Lý Do Chọn Các Tham Số

1. **Sample Rate 44,100 Hz**:
   - Đây là tần số lấy mẫu chuẩn cho audio chất lượng cao
   - Đảm bảo có thể capture được tần số lên đến 22,050 Hz (theo định lý Nyquist)
   - Tương thích với hầu hết các nguồn audio hiện đại

2. **n_mfcc = 13**:
   - 13 hệ số MFCC là đủ để biểu diễn đặc trưng phổ tần số âm thanh
   - Giảm chiều dữ liệu so với việc sử dụng toàn bộ phổ tần số
   - Được chứng minh hiệu quả trong nhiều bài toán phân loại âm thanh

3. **n_fft = 1,102 (25ms)**:
   - 25ms là độ dài cửa sổ phù hợp để phân tích tần số
   - Đủ ngắn để capture các thay đổi nhanh trong tín hiệu
   - Đủ dài để có độ phân giải tần số tốt

4. **hop_length = 441 (10ms)**:
   - Tạo overlap 60% giữa các cửa sổ (overlap = (n_fft - hop_length) / n_fft)
   - Đảm bảo không bỏ sót thông tin quan trọng
   - Tạo ra số lượng frame MFCC phù hợp cho mỗi segment 1 giây

5. **Window 'hamming'**:
   - Giảm hiện tượng rò rỉ phổ (spectral leakage) so với cửa sổ hình chữ nhật
   - Tạo ra phổ tần số mượt mà và chính xác hơn
   - Phổ biến trong xử lý tín hiệu âm thanh

---

### 4.2.3. Quy Trình Tiền Xử Lý Sau Khi Trích Xuất

Sau khi trích xuất MFCC, dữ liệu cần được xử lý thêm để phù hợp với đầu vào của mô hình LSTM. Quy trình bao gồm 2 bước chính: **Padding** và **Scaling**.

#### 4.2.3.1. Đệm (Padding)

##### Tại Sao Cần Padding?

1. **Vấn đề về độ dài khác nhau**:
   - Mỗi segment audio 1 giây có thể tạo ra số lượng frame MFCC khác nhau do:
     - Sự khác biệt nhỏ trong độ dài thực tế của segment
     - Cách tính toán frame dựa trên `hop_length` và `n_fft`
   - Ví dụ: Một segment có thể tạo ra 87 frame, segment khác có thể có 88 hoặc 89 frame

2. **Yêu cầu của mô hình LSTM**:
   - Mô hình LSTM yêu cầu đầu vào có cùng kích thước (shape) cho tất cả các mẫu trong một batch
   - Input shape phải là: `(batch_size, timesteps, features)` = `(N, max_len, 13)`
   - Không thể xử lý các mảng có độ dài khác nhau trong cùng một batch

3. **Giải pháp Padding**:
   - Đệm các feature ngắn hơn bằng giá trị 0 (zero-padding)
   - Cắt bớt các feature dài hơn về `max_len`
   - Đảm bảo tất cả features có cùng độ dài `max_len`

##### Cách Xác Định max_len

Quy trình xác định `max_len` được thực hiện trong hàm `pad_features()`:

```python
def pad_features(features, max_len=None):
    # 1. Lọc các feature hợp lệ
    valid_features = [feat for feat in features 
                     if isinstance(feat, np.ndarray) 
                     and feat.ndim > 0 
                     and feat.shape[0] > 0]
    
    # 2. Tính max_len nếu chưa có
    if max_len is None:
        max_len = max(feat.shape[0] for feat in valid_features)
    
    # 3. Padding hoặc cắt bớt mỗi feature
    for feat in features:
        if feat.shape[0] < max_len:
            # Đệm bằng 0 ở cuối
            padded = np.pad(feat, ((0, max_len - feat.shape[0]), (0, 0)), 
                          mode='constant')
        else:
            # Cắt bớt về max_len
            padded = feat[:max_len, :]
```

**Quy trình xác định max_len**:

1. **Trong quá trình training (Audio gốc)**:
   - `max_len` được tính từ tập **train/validation**
   - Tìm độ dài lớn nhất trong tất cả các MFCC features của tập train/val
   - Giá trị này được lưu vào file `max_len_audio_original.pkl`
   - **Lý do**: Chỉ sử dụng dữ liệu training để tính max_len, tránh data leakage từ test set

2. **Trong quá trình fine-tuning (YouTube)**:
   - `max_len` được tính từ tập **train/validation** của dữ liệu YouTube
   - Lưu vào file `max_len_audio_youtube.pkl`
   - Có thể khác với `max_len` của audio gốc do đặc điểm dữ liệu khác nhau

3. **Trong quá trình inference/evaluation**:
   - Sử dụng `max_len` đã được lưu từ quá trình training
   - Đảm bảo tính nhất quán với dữ liệu đã huấn luyện

**Ví dụ thực tế**:
- Với segment 1 giây, sample rate 44,100 Hz, hop_length 441:
  - Số frame MFCC ≈ `(44100 - 1102) / 441 + 1 ≈ 98 frames`
  - `max_len` thường nằm trong khoảng 95-100 frames

##### Cách Thực Hiện Padding

1. **Padding cho feature ngắn hơn**:
   ```python
   # Feature có shape (87, 13), max_len = 98
   # Đệm 11 frame bằng 0 ở cuối
   padded = np.pad(feat, ((0, 98 - 87), (0, 0)), mode='constant')
   # Kết quả: shape (98, 13)
   ```

2. **Cắt bớt cho feature dài hơn**:
   ```python
   # Feature có shape (102, 13), max_len = 98
   # Cắt bớt 4 frame ở cuối
   padded = feat[:98, :]
   # Kết quả: shape (98, 13)
   ```

3. **Kết quả**:
   - Tất cả features có cùng shape: `(max_len, 13)`
   - Có thể stack thành mảng 3D: `(n_samples, max_len, 13)`

---

#### 4.2.3.2. Chuẩn Hóa (Scaling)

##### Tại Sao Cần StandardScaler?

1. **Vấn đề về phạm vi giá trị khác nhau**:
   - Các hệ số MFCC có thể có phạm vi giá trị rất khác nhau
   - Ví dụ: MFCC[0] (năng lượng tổng) có thể có giá trị lớn hơn nhiều so với MFCC[12]
   - Điều này có thể gây ra:
     - Một số features "thống trị" quá trình học
     - Gradient descent không ổn định
     - Mô hình khó hội tụ

2. **Yêu cầu của mô hình Neural Network**:
   - Các mô hình neural network hoạt động tốt hơn khi dữ liệu đầu vào được chuẩn hóa
   - Giúp gradient descent hoạt động hiệu quả hơn
   - Tăng tốc độ hội tụ và cải thiện độ chính xác

3. **Giải pháp StandardScaler**:
   - Chuẩn hóa mỗi feature về phân phối chuẩn với:
     - Mean (trung bình) = 0
     - Standard deviation (độ lệch chuẩn) = 1
   - Công thức: `z = (x - μ) / σ`
     - `x`: giá trị gốc
     - `μ`: mean của feature
     - `σ`: standard deviation của feature
     - `z`: giá trị sau khi chuẩn hóa

##### Cách StandardScaler Được Fit Trên Tập Huấn Luyện

Quy trình được thực hiện trong hàm `scale_features()`:

```python
def scale_features(features_padded, scaler=None):
    # 1. Reshape từ 3D về 2D để fit scaler
    # features_padded shape: (n_samples, max_len, 13)
    n_samples, n_timesteps, n_features = features_padded.shape
    features_reshaped = features_padded.reshape(-1, n_features)
    # Shape sau reshape: (n_samples * max_len, 13)
    
    # 2. Fit scaler trên tập training (nếu chưa có)
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(features_reshaped)  # Tính mean và std
        # scaler.mean_: shape (13,) - mean của mỗi MFCC coefficient
        # scaler.scale_: shape (13,) - std của mỗi MFCC coefficient
    
    # 3. Transform dữ liệu
    features_scaled_reshaped = scaler.transform(features_reshaped)
    features_scaled = features_scaled_reshaped.reshape(n_samples, n_timesteps, n_features)
    # Shape cuối cùng: (n_samples, max_len, 13)
```

**Quy trình chi tiết**:

1. **Trong quá trình training (Audio gốc)**:
   - **Bước 1**: Fit scaler trên tập **training**:
     ```python
     X_train_scaled, scaler = scale_features(X_train)  # scaler được fit
     ```
     - Scaler tính toán `mean` và `std` cho từng MFCC coefficient (13 giá trị)
     - Chỉ sử dụng dữ liệu training, không dùng validation hoặc test
   
   - **Bước 2**: Transform tập validation bằng scaler đã fit:
     ```python
     X_val_scaled, _ = scale_features(X_val, scaler=scaler)  # Sử dụng scaler đã fit
     ```
     - Sử dụng cùng `mean` và `std` từ tập training
     - Đảm bảo tính nhất quán
   
   - **Bước 3**: Lưu scaler:
     ```python
     with open(scaler_path, 'wb') as f:
         pickle.dump(scaler, f)
     ```
     - Lưu vào file `scaler_audio_original.pkl`
     - Sử dụng lại trong quá trình evaluation và inference

2. **Trong quá trình fine-tuning (YouTube)**:
   - Fit scaler mới trên tập train/val của dữ liệu YouTube
   - Lưu vào file `scaler_audio_youtube.pkl`
   - Có thể có `mean` và `std` khác với audio gốc

3. **Trong quá trình evaluation/inference**:
   - Load scaler đã được lưu từ quá trình training
   - Sử dụng để transform dữ liệu mới
   - **Quan trọng**: Không được fit lại scaler trên dữ liệu test/inference

##### Tại Sao Chỉ Fit Trên Tập Training?

1. **Tránh Data Leakage**:
   - Nếu fit scaler trên toàn bộ dữ liệu (bao gồm test), thông tin từ test set sẽ "rò rỉ" vào quá trình training
   - Mô hình có thể học được thông tin về test set, dẫn đến đánh giá không công bằng

2. **Mô phỏng thực tế**:
   - Trong thực tế, ta chỉ có dữ liệu training để fit scaler
   - Dữ liệu mới (test/inference) sẽ được transform bằng scaler đã học từ training
   - Đảm bảo mô hình hoạt động đúng trong môi trường thực tế

3. **Tính nhất quán**:
   - Tất cả dữ liệu (train, val, test, inference) đều được chuẩn hóa bằng cùng một scaler
   - Đảm bảo phân phối dữ liệu nhất quán

##### Cách Thực Hiện Scaling

1. **Reshape dữ liệu**:
   ```python
   # Input: (n_samples, max_len, 13)
   # Reshape: (n_samples * max_len, 13)
   # Lý do: StandardScaler làm việc với 2D array
   ```

2. **Fit scaler** (chỉ trên training):
   ```python
   scaler.fit(features_reshaped)
   # Tính mean và std cho mỗi MFCC coefficient
   # scaler.mean_[i] = mean của MFCC coefficient thứ i
   # scaler.scale_[i] = std của MFCC coefficient thứ i
   ```

3. **Transform**:
   ```python
   features_scaled = scaler.transform(features_reshaped)
   # Áp dụng công thức: (x - mean) / std cho từng feature
   ```

4. **Reshape lại**:
   ```python
   # Reshape về shape ban đầu: (n_samples, max_len, 13)
   features_scaled = features_scaled.reshape(n_samples, max_len, 13)
   ```

##### Kết Quả Sau Scaling

- Mỗi MFCC coefficient có:
  - Mean ≈ 0
  - Standard deviation ≈ 1
- Tất cả features có cùng phạm vi giá trị
- Dữ liệu sẵn sàng cho mô hình LSTM

---

### 4.2.4. Tóm Tắt Quy Trình Hoàn Chỉnh

```
Audio File (.wav)
    ↓
1. Load audio với librosa (sr=44100 Hz)
    ↓
2. Chia thành segments 1 giây
    ↓
3. Trích xuất MFCC cho mỗi segment
   - n_mfcc=13, n_fft=1102, hop_length=441
   - window='hamming', center=False
   - Kết quả: (n_frames, 13)
    ↓
4. Padding
   - Tính max_len từ tập training
   - Đệm hoặc cắt về max_len
   - Kết quả: (max_len, 13)
    ↓
5. Scaling
   - Fit StandardScaler trên tập training
   - Transform bằng scaler đã fit
   - Kết quả: (max_len, 13) với mean=0, std=1
    ↓
6. Stack thành batch
   - Kết quả: (batch_size, max_len, 13)
    ↓
7. Đầu vào cho mô hình LSTM
```

---

### 4.2.5. Lưu Ý Quan Trọng

1. **Tính nhất quán**:
   - Cùng một bộ tham số (n_mfcc, n_fft, hop_length) phải được sử dụng trong toàn bộ pipeline
   - Cùng một `max_len` và `scaler` phải được sử dụng cho training, validation, test và inference

2. **Lưu trữ**:
   - `max_len` được lưu trong file `.pkl`
   - `scaler` được lưu trong file `.pkl`
   - Cần load lại khi sử dụng mô hình

3. **Xử lý lỗi**:
   - Kiểm tra độ dài segment trước khi trích xuất MFCC
   - Xử lý các segment quá ngắn (< 0.1 giây)
   - Xử lý các feature rỗng hoặc không hợp lệ

---

*Tài liệu được tạo từ phân tích code trong dự án Drone Detection Project*


