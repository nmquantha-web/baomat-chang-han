
import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from collections import deque
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# HẰNG SỐ
# ─────────────────────────────────────────────
MODEL_DIR   = "models"
FEATURE_META_PATH = os.path.join(MODEL_DIR, "feature_metadata.json")
WINDOW_SIZE = 10    # Phải nhất quán với preprocessing.py và lstm_model.py

# Nhãn kết quả hiển thị trên giao diện
DISPLAY_LABEL = {
    "BENIGN"                    : "Bình thường",
    "DoS Hulk"                  : "Tấn công DoS Hulk",
    "PortScan"                  : "Quét cổng",
    "DDoS"                      : "Tấn công DDoS",
    "DoS GoldenEye"             : "Tấn công DoS GoldenEye",
    "FTP-Patator"               : "Brute Force FTP",
    "SSH-Patator"               : "Brute Force SSH",
    "DoS slowloris"             : "Tấn công DoS Slowloris",
    "DoS Slowhttptest"          : "Tấn công DoS Slowhttptest",
    "Bot"                       : "Botnet",
    "Web Attack – Brute Force"  : "Tấn công Web Brute Force",
    "Web Attack – XSS"          : "Tấn công XSS",
    "Infiltration"              : "Xâm nhập nội bộ",
    "Heartbleed"                : "Lỗ hổng Heartbleed",
}

# Mức độ nguy hiểm
SEVERITY = {
    "BENIGN"                    : "normal",
    "DoS Hulk"                  : "high",
    "PortScan"                  : "medium",
    "DDoS"                      : "critical",
    "DoS GoldenEye"             : "high",
    "FTP-Patator"               : "medium",
    "SSH-Patator"               : "high",
    "DoS slowloris"             : "high",
    "DoS Slowhttptest"          : "high",
    "Bot"                       : "critical",
    "Web Attack – Brute Force"  : "medium",
    "Web Attack – XSS"          : "medium",
    "Infiltration"              : "critical",
    "Heartbleed"                : "critical",
}


# ─────────────────────────────────────────────
# 1. TẢI MODELS VÀ ENCODERS (singleton pattern)
# ─────────────────────────────────────────────
class ModelLoader:
    """
    Singleton: load model một lần duy nhất vào bộ nhớ,
    tránh reload mỗi lần có request mới từ Flask.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self):
        if self._loaded:
            return
        print("  [ModelLoader] Đang tải models...")
        self.scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        self.le     = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
        self.clf_if = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))
        self.lstm   = tf.keras.models.load_model(
            os.path.join(MODEL_DIR, "lstm_best.h5")
        )
        self._loaded = True
        print("  [ModelLoader] Tải hoàn tất.")

    @property
    def is_loaded(self):
        return self._loaded


model_loader = ModelLoader()


# ─────────────────────────────────────────────
# 2. TIỀN XỬ LÝ ĐẦU VÀO
# ─────────────────────────────────────────────
def preprocess_input(df: pd.DataFrame) -> np.ndarray:
    """
    Áp dụng cùng pipeline tiền xử lý đã dùng lúc training:
    1. Loại bỏ cột định danh nếu có.
    2. Xử lý Inf / NaN.
    3. Chuẩn hóa bằng scaler đã lưu.
    Trả về mảng 2D shape (n_samples, n_features).
    """
    model_loader.load()

    drop_cols = ["Flow ID", "Source IP", "Source Port",
                 "Destination IP", "Destination Port", "Timestamp", "Label"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Xử lý Inf / NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    X = df.values.astype(np.float32)

    # Cắt bớt / padding nếu số features không khớp
    expected_features = model_loader.scaler.n_features_in_
    if X.shape[1] > expected_features:
        X = X[:, :expected_features]
    elif X.shape[1] < expected_features:
        pad = np.zeros((X.shape[0], expected_features - X.shape[1]),
                       dtype=np.float32)
        X = np.hstack([X, pad])

    X_scaled = model_loader.scaler.transform(X)
    return X_scaled


# ─────────────────────────────────────────────
# 3. TẠO SEQUENCE CHO LSTM (STREAMING)
# ─────────────────────────────────────────────
class SequenceBuffer:
    """
    Buffer trượt dùng deque: duy trì WINDOW_SIZE mẫu gần nhất.
    Khi buffer đủ window_size phần tử → tạo sequence để đưa vào LSTM.
    Phù hợp với chế độ real-time (từng dòng một).
    """
    def __init__(self, window_size: int = WINDOW_SIZE):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def add(self, sample: np.ndarray) -> np.ndarray | None:
        """Thêm 1 mẫu; trả về sequence nếu buffer đủ, ngược lại trả về None."""
        self.buffer.append(sample)
        if len(self.buffer) == self.window_size:
            return np.array(self.buffer, dtype=np.float32)[np.newaxis, ...]  # (1, W, F)
        return None

    def reset(self):
        self.buffer.clear()


def batch_to_sequences(X: np.ndarray,
                        window_size: int = WINDOW_SIZE) -> np.ndarray:
    """Tạo sequence cho toàn bộ batch đồng bộ với index mẫu gốc."""
    if len(X) == 0:
        return np.empty((0, window_size, X.shape[1]), dtype=np.float32)

    if len(X) < window_size:
        pad = np.zeros((window_size - len(X), X.shape[1]), dtype=np.float32)
        X = np.vstack([pad, X])

    sequences = []
    for i in range(len(X)):
        start = max(0, i - window_size + 1)
        window = X[start:i + 1]
        if len(window) < window_size:
            pad = np.zeros((window_size - len(window), X.shape[1]), dtype=np.float32)
            window = np.vstack([pad, window])
        sequences.append(window)
    return np.array(sequences, dtype=np.float32)


def load_feature_metadata():
    """Đọc metadata feature đầu vào do preprocessing lưu lại."""
    if not os.path.exists(FEATURE_META_PATH):
        raise FileNotFoundError(
            f"Thiếu metadata feature: '{FEATURE_META_PATH}'. "
            "Hãy chạy preprocessing.py trước khi inference."
        )
    import json
    with open(FEATURE_META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_feature_frame(df: pd.DataFrame, metadata: dict | None = None) -> pd.DataFrame:
    """Đưa DataFrame về đúng feature order và fill NaN/Inf theo metadata."""
    if metadata is None:
        metadata = load_feature_metadata()

    feature_names = metadata.get("feature_names", [])
    medians = metadata.get("medians", {})

    if not feature_names:
        feature_names = [c for c in df.columns if c != "Label"]

    df = df.copy()
    df = df.reindex(columns=feature_names, fill_value=0.0)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        if col in medians:
            df[col] = df[col].fillna(float(medians[col]))
        else:
            df[col] = df[col].fillna(df[col].median())

    return df


def create_inference_sequences(X: np.ndarray, window_size: int = WINDOW_SIZE) -> np.ndarray:
    """Tạo sequence theo đúng index mẫu: mỗi hàng tạo khung cửa sổ kết thúc ở hàng tương ứng."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if len(X) == 0:
        return np.empty((0, window_size, X.shape[1]), dtype=np.float32)

    seqs = []
    for i in range(len(X)):
        start = max(0, i - window_size + 1)
        win = X[start:i + 1]
        if len(win) < window_size:
            pad = np.zeros((window_size - len(win), X.shape[1]), dtype=np.float32)
            win = np.vstack([pad, win])
        seqs.append(win)
    return np.array(seqs, dtype=np.float32)


# ─────────────────────────────────────────────
# 4. HÀM DỰ ĐOÁN CHÍNH (PIPELINE 2 TẦNG)
# ─────────────────────────────────────────────
def predict(df: pd.DataFrame) -> list[dict]:
    """
    Pipeline đầy đủ IF → LSTM cho một DataFrame đầu vào.
    Có bảo đảm mapping đúng mẫu gốc đến sequence tương ứng.
    """
    model_loader.load()
    metadata = load_feature_metadata()

    prepared = prepare_feature_frame(df, metadata)
    X_scaled = preprocess_input(prepared)
    n_samples = len(X_scaled)
    results = []

    if_preds = model_loader.clf_if.predict(X_scaled)
    if_scores = model_loader.clf_if.score_samples(X_scaled)

    anomaly_indices = np.where(if_preds == -1)[0]
    normal_indices = np.where(if_preds == 1)[0]

    for idx in normal_indices:
        results.append({
            "index": int(idx),
            "if_label": "normal",
            "if_score": float(if_scores[idx]),
            "label": "BENIGN",
            "label_vi": DISPLAY_LABEL.get("BENIGN", "Bình thường"),
            "severity": "normal",
            "confidence": 1.0,
            "source": "IF",
        })

    if len(anomaly_indices) > 0:
        X_anomaly = X_scaled[anomaly_indices]
        X_seq = create_inference_sequences(X_anomaly, WINDOW_SIZE)

        proba = model_loader.lstm.predict(X_seq, batch_size=256, verbose=0)
        pred_classes = np.argmax(proba, axis=1)
        pred_labels = model_loader.le.inverse_transform(pred_classes)
        confidences = np.max(proba, axis=1)

        for row_pos, orig_idx in enumerate(anomaly_indices):
            label = pred_labels[row_pos]
            results.append({
                "index": int(orig_idx),
                "if_label": "anomaly",
                "if_score": float(if_scores[orig_idx]),
                "label": label,
                "label_vi": DISPLAY_LABEL.get(label, label),
                "severity": SEVERITY.get(label, "medium"),
                "confidence": float(confidences[row_pos]),
                "source": "IF+LSTM",
            })

    results.sort(key=lambda x: x["index"])
    return results


# ─────────────────────────────────────────────
# 5. HÀM DỰ ĐOÁN TỪ FILE
# ─────────────────────────────────────────────
def predict_from_file(filepath: str) -> list[dict]:
    """
    Nhận đường dẫn file CSV, đọc và chạy pipeline dự đoán.
    Dùng khi người dùng upload file qua giao diện Flask.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")

    df = pd.read_csv(filepath, encoding="latin1", low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"  [predict] Đọc file: {filepath} — {len(df):,} dòng")

    return predict(df)


# ─────────────────────────────────────────────
# 6. HÀM THỐNG KÊ KẾT QUẢ
# ─────────────────────────────────────────────
def summarize_results(results: list[dict]) -> dict:
    """
    Tóm tắt kết quả dự đoán:
    - Tổng số mẫu, số mẫu bình thường, số mẫu bất thường
    - Phân phối nhãn tấn công
    - Phân phối mức độ nguy hiểm
    """
    total    = len(results)
    normal   = sum(1 for r in results if r["label"] == "BENIGN")
    abnormal = total - normal

    label_counts = {}
    severity_counts = {"normal": 0, "medium": 0, "high": 0, "critical": 0}

    for r in results:
        label = r["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
        sev = r.get("severity", "medium")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    return {
        "total"           : total,
        "normal"          : normal,
        "abnormal"        : abnormal,
        "attack_rate"     : round(abnormal / total * 100, 2) if total > 0 else 0,
        "label_counts"    : label_counts,
        "severity_counts" : severity_counts,
    }


# ─────────────────────────────────────────────
# 7. CLI TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Cách dùng: python predict.py <đường_dẫn_file.csv>")
        sys.exit(1)

    filepath = sys.argv[1]
    print(f"\n  Dự đoán file: {filepath}")
    results = predict_from_file(filepath)
    summary = summarize_results(results)

    print("\n===== TÓM TẮT KẾT QUẢ =====")
    print(f"  Tổng mẫu   : {summary['total']:,}")
    print(f"  Bình thường: {summary['normal']:,}")
    print(f"  Bất thường : {summary['abnormal']:,} ({summary['attack_rate']}%)")
    print(f"\n  Phân phối nhãn:")
    for label, count in sorted(summary["label_counts"].items(),
                                key=lambda x: -x[1]):
        print(f"    {label:<35} {count:>8,}")
