"""
Benchmark script để đo thời gian suy luận (inference time) của các Fusion Models.
"""
import os
import sys
import time
import numpy as np
import tensorflow as tf

# Thêm đường dẫn project vào sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.model_architectures.fusion_model import (
    AttentionFusionLayer, GatedFusionLayer, HybridFusionLayer
)


def benchmark_model(model, dummy_audio, dummy_vcam, num_warmup=10, num_samples=100):
    """
    Đo thời gian suy luận trung bình của model.

    Args:
        model: Keras model đã load
        dummy_audio: Dữ liệu audio giả lập
        dummy_vcam: Dữ liệu vcam giả lập
        num_warmup: Số lần warm-up
        num_samples: Số lần đo

    Returns:
        avg_time_ms: Thời gian trung bình (ms)
        std_time_ms: Độ lệch chuẩn (ms)
    """
    # Warm-up (chạy nháp để GPU/CPU nóng máy)
    for _ in range(num_warmup):
        _ = model.predict([dummy_audio, dummy_vcam], verbose=0)

    # Đo thời gian
    times = []
    for _ in range(num_samples):
        start_time = time.perf_counter()
        _ = model.predict([dummy_audio, dummy_vcam], verbose=0)
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # Convert to ms

    avg_time_ms = np.mean(times)
    std_time_ms = np.std(times)

    return avg_time_ms, std_time_ms


def main():
    print("=" * 60)
    print("BENCHMARK FUSION MODELS - INFERENCE TIME")
    print("=" * 60)

    # Kiểm tra GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU detected: {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU detected, using CPU")

    # Custom objects cho load model
    custom_objects = {
        'AttentionFusionLayer': AttentionFusionLayer,
        'GatedFusionLayer': GatedFusionLayer,
        'HybridFusionLayer': HybridFusionLayer,
    }

    # Định nghĩa các model paths
    model_paths = {
        'Concatenate': os.path.join(PROJECT_ROOT, 'models/fine_tuned_fusion_model/best_vcam_audio_fusion_model.keras'),
        'Attention': os.path.join(PROJECT_ROOT, 'models/fine_tuned_fusion_model -- attendtion/best_vcam_audio_fusion_model.keras'),
        'Gated': os.path.join(PROJECT_ROOT, 'models/fine_tuned_fusion_model -- gated/best_vcam_audio_fusion_model.keras'),
        'Hybrid': os.path.join(PROJECT_ROOT, 'models/fine_tuned_fusion_model -- hybrid/best_vcam_audio_fusion_model.keras'),
    }

    # Tạo dữ liệu giả lập (Dummy data)
    # Audio: (batch_size, time_steps, mfcc_features) = (1, 98, 13)
    # VCam: (batch_size, feature_dim) = (1, 256)
    dummy_audio = np.random.rand(1, 98, 13).astype(np.float32)
    dummy_vcam = np.random.rand(1, 256).astype(np.float32)

    print(f"\nInput shapes:")
    print(f"  Audio: {dummy_audio.shape}")
    print(f"  VCam:  {dummy_vcam.shape}")
    print(f"\nBenchmark settings: warmup=10, samples=100")
    print("-" * 60)

    results = {}

    for name, path in model_paths.items():
        if not os.path.exists(path):
            print(f"\n[{name}] Model not found: {path}")
            continue

        print(f"\n[{name}] Loading model...")
        try:
            model = tf.keras.models.load_model(path, custom_objects=custom_objects)

            # Đo thời gian
            avg_time, std_time = benchmark_model(model, dummy_audio, dummy_vcam)
            results[name] = {'avg': avg_time, 'std': std_time}

            print(f"[{name}] Inference time: {avg_time:.2f} +/- {std_time:.2f} ms")

            # Giải phóng bộ nhớ
            del model
            tf.keras.backend.clear_session()

        except Exception as e:
            print(f"[{name}] Error: {e}")

    # Tổng kết
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<15} {'Avg (ms)':<12} {'Std (ms)':<12} {'FPS':<10}")
    print("-" * 60)

    for name, data in results.items():
        fps = 1000 / data['avg'] if data['avg'] > 0 else 0
        print(f"{name:<15} {data['avg']:<12.2f} {data['std']:<12.2f} {fps:<10.1f}")

    print("-" * 60)
    print("Note: Thời gian trên chưa bao gồm thời gian trích xuất đặc trưng YOLO")
    print("      (thường khoảng 8-10ms trên RTX 3060)")


if __name__ == "__main__":
    main()
