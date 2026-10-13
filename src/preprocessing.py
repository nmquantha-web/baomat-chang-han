
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. CÁC ĐƯỜNG DẪN VÀ HẰNG SỐ
# ─────────────────────────────────────────────
DATA_DIR    = "data/raw"          # Thư mục chứa file CSV gốc của CICIDS2017
OUTPUT_DIR  = "data/processed"   # Thư mục lưu dữ liệu đã xử lý
MODEL_DIR   = "models"           # Thư mục lưu scaler, encoder
WINDOW_SIZE = 10                  # Độ dài cửa sổ trượt cho LSTM
RANDOM_SEED = 42
FEATURE_META_PATH = os.path.join(MODEL_DIR, "feature_metadata.json")

# Nhãn lớp gốc trong CICIDS2017 (giữ nguyên tên gốc để LabelEncoder nhất quán)
LABEL_COLUMN = " Label"   # Cột nhãn trong CICIDS2017 có dấu cách đầu

# Các loại tấn công cần giữ lại (bỏ qua các nhãn ít mẫu < 500)
KEEP_LABELS = [
    "BENIGN",
    "DoS Hulk",
    "PortScan",
    "DDoS",
    "DoS GoldenEye",
    "FTP-Patator",
    "SSH-Patator",
    "DoS slowloris",
    "DoS Slowhttptest",
    "Bot",
    "Web Attack – Brute Force",
    "Web Attack – XSS",
    "Infiltration",
    "Heartbleed",
]

# ─────────────────────────────────────────────
# 2. TẢI VÀ GỘP CÁC FILE CSV
# ─────────────────────────────────────────────
def load_cicids2017(data_dir: str) -> pd.DataFrame:
    """
    Đọc tất cả file CSV trong thư mục data_dir và gộp thành một DataFrame.
    CICIDS2017 được chia theo ngày (Monday.csv, Tuesday.csv, ...).
    """
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"Không tìm thấy file CSV nào trong '{data_dir}'.")

    frames = []
    for fname in sorted(csv_files):
        fpath = os.path.join(data_dir, fname)
        print(f"  [+] Đọc file: {fname} ...", end=" ")
        df = pd.read_csv(fpath, encoding="utf-8", low_memory=False)
        print(f"{len(df):,} dòng")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  [√] Tổng số mẫu sau khi gộp: {len(combined):,}")
    print(f"  [√] Số features: {combined.shape[1]}")
    return combined


# ─────────────────────────────────────────────
# 3. KHẢO SÁT DỮ LIỆU THÔ
# ─────────────────────────────────────────────
def explore_data(df: pd.DataFrame) -> None:
    """In thống kê mô tả cơ bản để hiểu cấu trúc dữ liệu."""
    print("\n===== KHẢO SÁT DỮ LIỆU THÔ =====")
    print(f"Kích thước: {df.shape}")
    print(f"\nPhân phối nhãn:\n{df[LABEL_COLUMN].value_counts()}")
    missing = df.isnull().sum()
    print(f"\nSố features có giá trị null: {(missing > 0).sum()}")
    print(missing[missing > 0])
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    print(f"\nTổng số giá trị vô cực (Inf): {inf_count:,}")


# ─────────────────────────────────────────────
# 4. LÀM SẠCH DỮ LIỆU
# ─────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bước 1: Chuẩn hóa tên cột (strip khoảng trắng).
    Bước 2: Lọc nhãn không cần thiết.
    Bước 3: Xử lý Inf, NaN, giá trị âm không hợp lệ.
    Bước 4: Loại bỏ cột không sử dụng (IP, port nguồn/đích).
    """
    print("\n===== LÀM SẠCH DỮ LIỆU =====")

    # Chuẩn hóa tên cột
    df.columns = df.columns.str.strip()
    global LABEL_COLUMN
    LABEL_COLUMN = "Label"

    # Lọc chỉ giữ lại các nhãn quan tâm
    original_len = len(df)
    df = df[df[LABEL_COLUMN].isin(KEEP_LABELS)].copy()
    print(f"  [√] Sau khi lọc nhãn: {len(df):,} / {original_len:,} mẫu")

    # Thay thế Inf bằng NaN rồi điền bằng median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    medians = df[numeric_cols].median()
    df[numeric_cols] = df[numeric_cols].fillna(medians)
    print(f"  [√] Đã xử lý Inf và NaN ({len(numeric_cols)} cột số)")

    # Loại bỏ các cột định danh không có giá trị phân loại
    drop_cols = ["Flow ID", "Source IP", "Source Port",
                 "Destination IP", "Destination Port", "Timestamp"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    print(f"  [√] Đã loại bỏ {len(drop_cols)} cột định danh")

    # Xóa dòng trùng lặp hoàn toàn
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"  [√] Đã xóa {before - len(df):,} dòng trùng lặp")

    return df


# ─────────────────────────────────────────────
# 5. LOẠI BỎ FEATURES DƯ THỪA (TƯƠNG QUAN CAO)
# ─────────────────────────────────────────────
def remove_correlated_features(df: pd.DataFrame,
                                threshold: float = 0.98) -> pd.DataFrame:
    """
    Loại bỏ một trong hai cột nếu hệ số tương quan Pearson vượt ngưỡng threshold.
    Giữ lại cột có tên xuất hiện trước trong danh sách.
    """
    print(f"\n===== LOẠI BỎ FEATURES TƯƠNG QUAN CAO (ngưỡng={threshold}) =====")
    label_col = df[LABEL_COLUMN]
    feature_df = df.drop(columns=[LABEL_COLUMN])

    corr_matrix = feature_df.corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper_tri.columns
               if any(upper_tri[col] > threshold)]

    feature_df.drop(columns=to_drop, inplace=True)
    print(f"  [√] Đã loại bỏ {len(to_drop)} features tương quan cao: {to_drop}")

    df_cleaned = pd.concat([feature_df, label_col], axis=1)
    print(f"  [√] Số features còn lại: {feature_df.shape[1]}")
    return df_cleaned


# ─────────────────────────────────────────────
# 6. MÃ HÓA NHÃN VÀ CHUẨN HÓA ĐẶC TRƯNG
# ─────────────────────────────────────────────
def encode_and_scale(df: pd.DataFrame):
    """
    - Mã hóa nhãn chuỗi → số nguyên bằng LabelEncoder.
    - Chuẩn hóa features về [0, 1] bằng MinMaxScaler.
    Trả về: X (numpy array), y (numpy array), encoder, scaler.
    """
    print("\n===== MÃ HÓA NHÃN VÀ CHUẨN HÓA =====")

    # Mã hóa nhãn
    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COLUMN])
    print(f"  [√] Các nhãn đã mã hóa: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Chuẩn hóa features
    X_raw = df.drop(columns=[LABEL_COLUMN]).values.astype(np.float32)
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Lý do chọn MinMaxScaler thay vì StandardScaler:
    # CICIDS2017 có nhiều features phân phối lệch nặng (skewed), MinMaxScaler
    # không giả định phân phối chuẩn và giữ nguyên cấu trúc điểm ngoại lệ
    # — điều này quan trọng vì Isolation Forest cần phát hiện outlier.
    X = scaler.fit_transform(X_raw)

    print(f"  [√] Shape X: {X.shape}, y: {y.shape}")
    return X, y, le, scaler


# ─────────────────────────────────────────────
# 7. XỬ LÝ MẤT CÂN BẰNG LỚP (SMOTE)
# ─────────────────────────────────────────────
def apply_smote(X_train: np.ndarray, y_train: np.ndarray,
                random_state: int = RANDOM_SEED):
    """
    Áp dụng SMOTE (Synthetic Minority Over-sampling Technique) trên tập train.
    SMOTE tạo mẫu tổng hợp cho các lớp thiểu số bằng cách nội suy giữa
    các điểm lân cận k-nearest, thay vì sao chép lại mẫu thô.
    Chỉ áp dụng trên tập TRAIN để tránh data leakage sang tập test.
    """
    print("\n===== XỬ LÝ MẤT CÂN BẰNG LỚP (SMOTE) =====")
    print(f"  Trước SMOTE: {np.bincount(y_train)}")

    # k_neighbors=5 là giá trị mặc định; nếu lớp thiểu số < 6 mẫu thì giảm k
    min_class_count = np.bincount(y_train).min()
    k = min(5, min_class_count - 1)
    smote = SMOTE(random_state=random_state, k_neighbors=k)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(f"  Sau SMOTE:  {np.bincount(y_res)}")
    print(f"  [√] Tổng mẫu train sau SMOTE: {len(X_res):,}")
    return X_res, y_res


# ─────────────────────────────────────────────
# 8. TẠO SEQUENCE CHO LSTM (SLIDING WINDOW)
# ─────────────────────────────────────────────
def create_sequences(X: np.ndarray, y: np.ndarray,
                     window_size: int = WINDOW_SIZE):
    """
    Biến đổi dữ liệu dạng bảng (2D) thành chuỗi thời gian (3D) cho LSTM.
    Với mỗi mẫu tại vị trí i, cửa sổ trượt lấy [i-window_size+1 ... i]
    làm một sequence. Nhãn của sequence là nhãn của mẫu cuối trong cửa sổ.

    Đầu vào : X shape (N, F), y shape (N,)
    Đầu ra  : X_seq shape (N-window_size+1, window_size, F)
              y_seq shape (N-window_size+1,)
    """
    Xs, ys = [], []
    for i in range(window_size - 1, len(X)):
        Xs.append(X[i - window_size + 1: i + 1])  # shape: (window_size, F)
        ys.append(y[i])
    X_seq = np.array(Xs, dtype=np.float32)
    y_seq = np.array(ys, dtype=np.int32)
    print(f"  [√] Sau sliding window: X_seq {X_seq.shape}, y_seq {y_seq.shape}")
    return X_seq, y_seq


# ─────────────────────────────────────────────
# 9. HÀM CHÍNH
# ─────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("  TIỀN XỬ LÝ DỮ LIỆU CICIDS2017")
    print("=" * 60)

    # ── Bước 1: Tải dữ liệu ──────────────────────────────────
    df = load_cicids2017(DATA_DIR)

    # ── Bước 2: Khảo sát ─────────────────────────────────────
    explore_data(df)

    # ── Bước 3: Làm sạch ─────────────────────────────────────
    df = clean_data(df)

    # ── Bước 4: Loại features tương quan cao ─────────────────
    df = remove_correlated_features(df, threshold=0.98)

    # ── Bước 5: Mã hóa và chuẩn hóa ─────────────────────────
    X, y, le, scaler = encode_and_scale(df)

    # ── Bước 6: Chia tập dữ liệu 70/15/15 ───────────────────
    # Chia train+temp (70/30) rồi chia temp thành val/test (50/50 → 15/15)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
    )
    print(f"\n  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # ── Bước 7: SMOTE trên tập train ─────────────────────────
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # ── Bước 8: Tạo sequence cho LSTM ────────────────────────
    print("\n===== TẠO SEQUENCE CHO LSTM =====")
    X_train_seq, y_train_seq = create_sequences(X_train_res, y_train_res, WINDOW_SIZE)
    X_val_seq,   y_val_seq   = create_sequences(X_val,       y_val,       WINDOW_SIZE)
    X_test_seq,  y_test_seq  = create_sequences(X_test,      y_test,      WINDOW_SIZE)

    # ── Bước 9: Lưu kết quả ──────────────────────────────────
    print("\n===== LƯU KẾT QUẢ =====")
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train_res)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train_res)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"),   X_val)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"),  y_test)

    np.save(os.path.join(OUTPUT_DIR, "X_train_seq.npy"), X_train_seq)
    np.save(os.path.join(OUTPUT_DIR, "y_train_seq.npy"), y_train_seq)
    np.save(os.path.join(OUTPUT_DIR, "X_val_seq.npy"),   X_val_seq)
    np.save(os.path.join(OUTPUT_DIR, "y_val_seq.npy"),   y_val_seq)
    np.save(os.path.join(OUTPUT_DIR, "X_test_seq.npy"),  X_test_seq)
    np.save(os.path.join(OUTPUT_DIR, "y_test_seq.npy"),  y_test_seq)

    feature_names = [c for c in df.columns if c != "Label"]
    metadata = {
        "feature_names": feature_names,
        "medians": {str(k): float(v) for k, v in df[feature_names].median().items()},
        "window_size": WINDOW_SIZE,
        "label_column": "Label",
    }

    with open(FEATURE_META_PATH, "w", encoding="utf-8") as f:
        import json
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Lưu scaler và encoder để sử dụng lúc inference
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(le,     os.path.join(MODEL_DIR, "label_encoder.pkl"))

    print(f"  [√] Đã lưu dữ liệu vào '{OUTPUT_DIR}/'")
    print(f"  [√] Đã lưu scaler và encoder vào '{MODEL_DIR}/'")
    print(f"  [√] Đã lưu metadata feature vào '{FEATURE_META_PATH}'")
    print("\n  Tiền xử lý hoàn tất!")


if __name__ == "__main__":
    main()
