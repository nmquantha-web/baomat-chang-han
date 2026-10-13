
import os
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# TensorFlow / Keras
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # Tắt log TF không cần thiết
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (LSTM, Dense, Dropout,
                                     BatchNormalization, Input)
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, TensorBoard)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ─────────────────────────────────────────────
# HẰNG SỐ
# ─────────────────────────────────────────────
DATA_DIR    = "data/processed"
MODEL_DIR   = "models"
RESULT_DIR  = "results"
LOG_DIR     = "logs/lstm"
RANDOM_SEED = 42

# Cài đặt seed toàn cục để kết quả tái lập được
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Siêu tham số mô hình
LSTM_CONFIG = {
    "window_size"   : 10,     # Phải khớp với preprocessing.py
    "lstm_units_1"  : 128,    # Số đơn vị LSTM tầng 1
    "lstm_units_2"  : 64,     # Số đơn vị LSTM tầng 2
    "dense_units"   : 64,     # Số nơ-ron lớp Dense ẩn
    "dropout_rate"  : 0.30,   # Tỷ lệ Dropout để tránh overfitting
    "batch_size"    : 512,
    "epochs"        : 50,
    "learning_rate" : 0.001,
}


# ─────────────────────────────────────────────
# 1. TẢI DỮ LIỆU SEQUENCE
# ─────────────────────────────────────────────
def load_sequence_data():
    """
    Tải dữ liệu đã được tạo sequence từ preprocessing.py.
    Trả về X_train_seq, y_train_seq, X_val_seq, y_val_seq,
           X_test_seq, y_test_seq, label_encoder.
    """
    print("  Đang tải dữ liệu sequence...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train_seq.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train_seq.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, "X_val_seq.npy"))
    y_val   = np.load(os.path.join(DATA_DIR, "y_val_seq.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test_seq.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test_seq.npy"))
    le      = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

    n_classes = len(le.classes_)
    n_features = X_train.shape[2]
    print(f"  [√] X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")
    print(f"  [√] Số lớp: {n_classes} | Số features: {n_features}")
    return X_train, y_train, X_val, y_val, X_test, y_test, le, n_classes, n_features


# ─────────────────────────────────────────────
# 2. XÂY DỰNG MÔ HÌNH LSTM
# ─────────────────────────────────────────────
def build_lstm_model(window_size: int,
                     n_features: int,
                     n_classes: int) -> tf.keras.Model:
    """
    Kiến trúc LSTM 2 tầng với BatchNormalization và Dropout.

    ┌─────────────────────────────────┐
    │ Input   (window_size, n_feat)   │
    ├─────────────────────────────────┤
    │ LSTM(128, return_sequences=True)│
    │ BatchNorm + Dropout(0.3)        │
    ├─────────────────────────────────┤
    │ LSTM(64, return_sequences=False)│
    │ BatchNorm + Dropout(0.3)        │
    ├─────────────────────────────────┤
    │ Dense(64, activation='relu')    │
    │ Dropout(0.3)                    │
    ├─────────────────────────────────┤
    │ Dense(n_classes, 'softmax')     │
    └─────────────────────────────────┘

    Lý do chọn categorical_crossentropy: bài toán phân loại đa lớp
    với nhãn one-hot encoded — đây là hàm mất mát chuẩn nhất.
    """
    model = Sequential([
        Input(shape=(window_size, n_features)),

        # ── Tầng LSTM 1: trích xuất đặc trưng chuỗi dài hạn ──────────────
        LSTM(LSTM_CONFIG["lstm_units_1"], return_sequences=True,
             recurrent_dropout=0.0,        # Tránh làm chậm GPU
             kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        BatchNormalization(),
        Dropout(LSTM_CONFIG["dropout_rate"]),

        # ── Tầng LSTM 2: tổng hợp biểu diễn ─────────────────────────────
        LSTM(LSTM_CONFIG["lstm_units_2"], return_sequences=False,
             kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        BatchNormalization(),
        Dropout(LSTM_CONFIG["dropout_rate"]),

        # ── Lớp fully connected ──────────────────────────────────────────
        Dense(LSTM_CONFIG["dense_units"], activation="relu",
              kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        Dropout(LSTM_CONFIG["dropout_rate"]),

        # ── Lớp đầu ra ───────────────────────────────────────────────────
        Dense(n_classes, activation="softmax"),
    ], name="LSTM_IDS")

    optimizer = Adam(learning_rate=LSTM_CONFIG["learning_rate"],
                     clipnorm=1.0)   # Gradient clipping tránh exploding gradient
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
    return model


# ─────────────────────────────────────────────
# 3. THIẾT LẬP CALLBACKS
# ─────────────────────────────────────────────
def get_callbacks(model_path: str) -> list:
    """
    EarlyStopping  : dừng sớm nếu val_loss không cải thiện sau 7 epoch.
    ModelCheckpoint: lưu trọng số tốt nhất theo val_accuracy.
    ReduceLROnPlateau: giảm learning rate khi val_loss bình ổn.
    TensorBoard    : ghi log để theo dõi training curve.
    """
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(log_dir=LOG_DIR, histogram_freq=0),
    ]
    return callbacks


# ─────────────────────────────────────────────
# 4. HUẤN LUYỆN
# ─────────────────────────────────────────────
def train_lstm(model: tf.keras.Model,
               X_train: np.ndarray, y_train_ohe: np.ndarray,
               X_val: np.ndarray,   y_val_ohe: np.ndarray,
               callbacks: list) -> tf.keras.callbacks.History:
    """Thực hiện vòng lặp huấn luyện với validation."""
    print("\n  Bắt đầu huấn luyện LSTM...")
    history = model.fit(
        X_train, y_train_ohe,
        validation_data=(X_val, y_val_ohe),
        epochs=LSTM_CONFIG["epochs"],
        batch_size=LSTM_CONFIG["batch_size"],
        callbacks=callbacks,
        verbose=1
    )
    return history


# ─────────────────────────────────────────────
# 5. VẼ LEARNING CURVE
# ─────────────────────────────────────────────
def plot_learning_curve(history: tf.keras.callbacks.History,
                        save_path: str) -> None:
    """
    Hình 3.3 / 4.3: Đường cong học tập của mô hình LSTM.
    Trục hoành: số epoch; trục tung: giá trị loss / accuracy.
    Đường xanh = training, đường cam = validation.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history.history["loss"]) + 1)

    # ── Loss ─────────────────────────────────────────────────────────────
    ax1.plot(epochs, history.history["loss"],     color="#1565C0",
             linewidth=2, label="Training loss")
    ax1.plot(epochs, history.history["val_loss"], color="#EF6C00",
             linewidth=2, label="Validation loss", linestyle="--")
    ax1.set_title("Loss theo epoch", fontsize=13)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Categorical Cross-Entropy Loss", fontsize=11)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # ── Accuracy ─────────────────────────────────────────────────────────
    ax2.plot(epochs, history.history["accuracy"],     color="#2E7D32",
             linewidth=2, label="Training accuracy")
    ax2.plot(epochs, history.history["val_accuracy"], color="#C62828",
             linewidth=2, label="Validation accuracy", linestyle="--")
    ax2.set_title("Accuracy theo epoch", fontsize=13)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Hình 4.3 – Learning Curve của mô hình LSTM", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [√] Đã lưu learning curve: {save_path}")


# ─────────────────────────────────────────────
# 6. VẼ CONFUSION MATRIX
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: list,
                          save_path: str) -> None:
    """
    Hình 4.2: Confusion matrix LSTM — màu sắc theo tỷ lệ (normalize='true').
    Ô (i, j): tỷ lệ mẫu thực tế thuộc lớp i được dự đoán thành lớp j.
    """
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5)
    ax.set_xlabel("Nhãn dự đoán", fontsize=12)
    ax.set_ylabel("Nhãn thực tế",  fontsize=12)
    ax.set_title("Hình 4.2 – Confusion Matrix (LSTM, normalized)", fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0,  fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [√] Đã lưu confusion matrix: {save_path}")


# ─────────────────────────────────────────────
# 7. ĐÁNH GIÁ TRÊN TẬP TEST
# ─────────────────────────────────────────────
def evaluate_lstm(model: tf.keras.Model,
                  X_test: np.ndarray,
                  y_test: np.ndarray,
                  le) -> None:
    """In classification report đầy đủ và vẽ confusion matrix."""
    print("\n===== KẾT QUẢ LSTM (Phase 2) =====")
    y_pred_proba = model.predict(X_test, batch_size=512, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    print(classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        digits=4
    ))

    plot_confusion_matrix(
        y_test, y_pred,
        class_names=list(le.classes_),
        save_path=os.path.join(RESULT_DIR, "lstm_confusion_matrix.png")
    )

    # Lưu dự đoán để dùng trong pipeline kết hợp
    np.save(os.path.join(DATA_DIR, "lstm_pred_test.npy"), y_pred)
    np.save(os.path.join(DATA_DIR, "lstm_proba_test.npy"), y_pred_proba)


# ─────────────────────────────────────────────
# 8. HÀM CHÍNH
# ─────────────────────────────────────────────
def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,    exist_ok=True)

    print("=" * 60)
    print("  LSTM – PHASE 2")
    print("=" * 60)

    # Tải dữ liệu
    (X_train, y_train, X_val, y_val,
     X_test, y_test, le, n_classes, n_features) = load_sequence_data()

    # One-hot encoding nhãn
    y_train_ohe = to_categorical(y_train, num_classes=n_classes)
    y_val_ohe   = to_categorical(y_val,   num_classes=n_classes)
    y_test_ohe  = to_categorical(y_test,  num_classes=n_classes)

    # Xây dựng mô hình
    model = build_lstm_model(
        window_size=LSTM_CONFIG["window_size"],
        n_features=n_features,
        n_classes=n_classes
    )

    # Thiết lập callbacks
    model_path = os.path.join(MODEL_DIR, "lstm_best.h5")
    callbacks  = get_callbacks(model_path)

    # Huấn luyện
    history = train_lstm(model, X_train, y_train_ohe,
                                X_val,   y_val_ohe, callbacks)

    # Vẽ learning curve
    plot_learning_curve(
        history,
        save_path=os.path.join(RESULT_DIR, "lstm_learning_curve.png")
    )

    # Tải lại trọng số tốt nhất
    print(f"\n  Tải lại mô hình tốt nhất từ '{model_path}'...")
    best_model = load_model(model_path)

    # Đánh giá trên tập test
    evaluate_lstm(best_model, X_test, y_test, le)

    print("\n  LSTM hoàn tất!")


if __name__ == "__main__":
    main()
