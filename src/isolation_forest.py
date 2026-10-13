
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")   # Không cần GUI khi chạy server
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_fscore_support)

# ─────────────────────────────────────────────
# HẰNG SỐ
# ─────────────────────────────────────────────
DATA_DIR   = "data/processed"
MODEL_DIR  = "models"
RESULT_DIR = "results"
RANDOM_SEED = 42

# Tham số Isolation Forest
IF_PARAMS = {
    "n_estimators"  : 100,    # Số cây Isolation Tree (càng nhiều → ổn định hơn)
    "contamination" : 0.1,    # Ước lượng tỷ lệ outlier trong dữ liệu (~10%)
    "max_samples"   : "auto", # Mặc định: min(256, n_samples)
    "max_features"  : 1.0,    # Sử dụng toàn bộ features
    "bootstrap"     : False,  # Không lấy mẫu có hoàn lại
    "random_state"  : RANDOM_SEED,
    "n_jobs"        : -1,     # Dùng toàn bộ CPU core
}


# ─────────────────────────────────────────────
# 1. TẢI DỮ LIỆU
# ─────────────────────────────────────────────
def load_data():
    """
    Tải X_train (flat, chưa tạo sequence) và X_test, y_test từ thư mục processed.
    IF hoạt động trên không gian đặc trưng 2D, không cần sequence.
    """
    print("  Đang tải dữ liệu...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    print(f"  [√] X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, y_train, X_test, y_test, le


# ─────────────────────────────────────────────
# 2. CHUẨN BỊ DỮ LIỆU HUẤN LUYỆN IF
# ─────────────────────────────────────────────
def prepare_if_training_data(X_train: np.ndarray,
                              y_train: np.ndarray,
                              le) -> np.ndarray:
    """
    IF được huấn luyện CHỈ trên traffic BENIGN (unsupervised).
    Ý tưởng: mô hình học phân phối "bình thường", từ đó phát hiện
    bất kỳ lưu lượng nào lệch khỏi phân phối này là bất thường.
    """
    benign_idx = np.where(y_train == le.transform(["BENIGN"])[0])[0]
    X_benign = X_train[benign_idx]
    print(f"  [√] Số mẫu BENIGN dùng để huấn luyện IF: {len(X_benign):,}")
    return X_benign


# ─────────────────────────────────────────────
# 3. HUẤN LUYỆN ISOLATION FOREST
# ─────────────────────────────────────────────
def train_isolation_forest(X_benign: np.ndarray) -> IsolationForest:
    """Khởi tạo và huấn luyện Isolation Forest."""
    print("\n  Đang huấn luyện Isolation Forest...")
    clf = IsolationForest(**IF_PARAMS)
    clf.fit(X_benign)
    print("  [√] Huấn luyện hoàn tất.")
    return clf


# ─────────────────────────────────────────────
# 4. DỰ ĐOÁN VÀ CHUYỂN ĐỔI NHÃN
# ─────────────────────────────────────────────
def predict_if(clf: IsolationForest, X: np.ndarray):
    """
    Trả về:
      - y_pred_binary : mảng 0 (bình thường) hoặc 1 (bất thường)
        (IF trả về  1 = inlier → chuyển thành 0,  -1 = outlier → chuyển thành 1)
      - scores        : anomaly score ∈ [-0.5, 0.5]; score càng âm → càng bất thường
    """
    raw_pred = clf.predict(X)          # 1 hoặc -1
    scores   = clf.score_samples(X)    # log-likelihood; âm hơn → bất thường hơn

    # Chuyển sang nhãn nhị phân rõ ràng: 0 = normal, 1 = anomaly
    y_pred_binary = np.where(raw_pred == -1, 1, 0)
    return y_pred_binary, scores


# ─────────────────────────────────────────────
# 5. ĐÁNH GIÁ KẾT QUẢ
# ─────────────────────────────────────────────
def evaluate_if(y_true_binary: np.ndarray,
                y_pred_binary: np.ndarray,
                scores: np.ndarray) -> dict:
    """
    y_true_binary: 0 = BENIGN, 1 = bất kỳ tấn công nào.
    """
    print("\n===== KẾT QUẢ ISOLATION FOREST =====")
    report = classification_report(
        y_true_binary, y_pred_binary,
        target_names=["BENIGN (normal)", "ATTACK (anomaly)"],
        output_dict=True
    )
    print(classification_report(
        y_true_binary, y_pred_binary,
        target_names=["BENIGN (normal)", "ATTACK (anomaly)"]
    ))

    # AUC: dùng score âm (anomaly score) làm xác suất dự đoán bất thường
    try:
        auc = roc_auc_score(y_true_binary, -scores)
        print(f"  ROC-AUC: {auc:.4f}")
    except Exception:
        auc = None

    cm = confusion_matrix(y_true_binary, y_pred_binary)
    print(f"\n  Confusion Matrix:\n{cm}")
    tn, fp, fn, tp = cm.ravel()
    print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"  False Positive Rate (FPR): {fp / (fp + tn):.4f}")
    print(f"  False Negative Rate (FNR): {fn / (fn + tp):.4f}")

    return {"report": report, "cm": cm, "auc": auc,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


# ─────────────────────────────────────────────
# 6. VẼ BIỂU ĐỒ PHÂN PHỐI ANOMALY SCORE
# ─────────────────────────────────────────────
def plot_anomaly_score_distribution(scores: np.ndarray,
                                    y_true_binary: np.ndarray,
                                    save_path: str) -> None:
    """
    Hình 4.1: Phân phối anomaly score của traffic bình thường (xanh)
    và traffic bất thường (đỏ). Đường nét đứt thể hiện ngưỡng phân cách.
    """
    normal_scores  = scores[y_true_binary == 0]
    anomaly_scores = scores[y_true_binary == 1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(normal_scores,  bins=80, alpha=0.6, color="#2196F3",
            label="BENIGN (bình thường)", density=True)
    ax.hist(anomaly_scores, bins=80, alpha=0.6, color="#F44336",
            label="ATTACK (bất thường)",  density=True)

    # Vẽ ngưỡng quyết định
    threshold = np.percentile(scores, 10)
    ax.axvline(x=threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"Ngưỡng quyết định ≈ {threshold:.3f}")

    ax.set_xlabel("Anomaly Score (score_samples)", fontsize=12)
    ax.set_ylabel("Mật độ xác suất", fontsize=12)
    ax.set_title("Hình 4.1 – Phân phối Anomaly Score: BENIGN vs ATTACK", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [√] Đã lưu biểu đồ: {save_path}")


# ─────────────────────────────────────────────
# 7. HÀM CHÍNH
# ─────────────────────────────────────────────
def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("=" * 60)
    print("  ISOLATION FOREST – PHASE 1")
    print("=" * 60)

    # Tải dữ liệu
    X_train, y_train, X_test, y_test, le = load_data()

    # Chuẩn bị dữ liệu BENIGN để train
    X_benign = prepare_if_training_data(X_train, y_train, le)

    # Huấn luyện
    clf = train_isolation_forest(X_benign)

    # Dự đoán trên tập test
    y_pred_binary, scores = predict_if(clf, X_test)

    # Chuyển y_test sang nhãn nhị phân (0=BENIGN, 1=attack)
    benign_label = le.transform(["BENIGN"])[0]
    y_test_binary = np.where(y_test == benign_label, 0, 1)

    # Đánh giá
    metrics = evaluate_if(y_test_binary, y_pred_binary, scores)

    # Vẽ biểu đồ
    plot_anomaly_score_distribution(
        scores, y_test_binary,
        save_path=os.path.join(RESULT_DIR, "if_score_distribution.png")
    )

    # Lưu mô hình
    model_path = os.path.join(MODEL_DIR, "isolation_forest.pkl")
    joblib.dump(clf, model_path)
    print(f"\n  [√] Đã lưu mô hình IF: {model_path}")

    # Lưu nhãn IF dự đoán để Phase 2 (LSTM) sử dụng
    np.save(os.path.join(DATA_DIR, "if_pred_test.npy"),   y_pred_binary)
    np.save(os.path.join(DATA_DIR, "if_scores_test.npy"), scores)
    print("  [√] Đã lưu kết quả dự đoán IF cho Phase 2.")
    print("\n  Isolation Forest hoàn tất!")


if __name__ == "__main__":
    main()
