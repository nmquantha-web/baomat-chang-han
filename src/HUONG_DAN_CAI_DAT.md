# PHỤ LỤC B — HƯỚNG DẪN CÀI ĐẶT VÀ CHẠY HỆ THỐNG

## Yêu cầu tối thiểu

| Thành phần | Yêu cầu |
|---|---|
| Python | 3.10–3.12 (khuyến nghị 3.12) |
| RAM | ≥ 8 GB (16 GB để xử lý dataset đầy đủ) |
| Ổ đĩa | ≥ 10 GB trống (dataset ~7 GB + model) |
| GPU | Không bắt buộc (CPU đủ dùng) |
| OS | Windows 10/11, Ubuntu 20.04+, macOS 12+ |

> Lưu ý: do TensorFlow hiện có các build mới hơn, project nên chạy trên Python 3.12 và dùng version tương thích từ `requirements.txt`.

---

## BƯỚC 1 — Tải dataset CICIDS2017

Dataset CICIDS2017 được cung cấp miễn phí bởi Canadian Institute for Cybersecurity:

```
https://www.unb.ca/cic/datasets/ids-2017.html
```

Tải về 8 file CSV (Monday → Friday, ~7 GB tổng):
- `Monday-WorkingHours.pcap_ISCX.csv`
- `Tuesday-WorkingHours.pcap_ISCX.csv`
- `Wednesday-workingHours.pcap_ISCX.csv`
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
- `Friday-WorkingHours-Morning.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-PortScan.csv`

Đặt tất cả vào thư mục: `data/raw/` ở thư mục gốc của project.

Ví dụ:
```bash
mkdir -p data/raw
```

---

## BƯỚC 2 — Cài đặt môi trường Python

### 2a. Tạo môi trường ảo (khuyến nghị)

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2b. Cài đặt thư viện

Từ thư mục gốc của project:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r src/requirements.txt
```

> **Lưu ý GPU (tùy chọn):** Nếu có NVIDIA GPU và đã cài CUDA 11.8, bạn có thể dùng build TensorFlow phù hợp với máy. Với môi trường hiện tại, điều kiện an toàn là dùng constraint trong `src/requirements.txt`.

---

## BƯỚC 3 — Khởi tạo cấu trúc thư mục

```bash
mkdir -p data/raw data/processed models results logs/lstm uploads templates static
```

Cấu trúc thư mục sau khi tạo:

```
project/
├── data/
│   ├── raw/          ← Đặt file CSV CICIDS2017 vào đây
│   └── processed/    ← Dữ liệu sau tiền xử lý (tự động tạo)
├── models/           ← Lưu model đã huấn luyện (tự động tạo)
├── results/          ← Biểu đồ kết quả (tự động tạo)
├── logs/lstm/        ← TensorBoard logs (tự động tạo)
├── uploads/          ← File CSV upload tạm thời
├── templates/        ← File HTML (index, result, dashboard)
├── static/           ← CSS, JS tùy chỉnh (nếu có)
├── preprocessing.py
├── isolation_forest.py
├── lstm_model.py
├── predict.py
├── app.py
└── requirements.txt
```

---

## BƯỚC 4 — Tiền xử lý dữ liệu

```bash
cd src
python preprocessing.py
```

Thời gian ước tính: **10–20 phút** tùy cấu hình máy.

Đầu ra:
- `data/processed/X_train.npy`, `y_train.npy`, ... (6 file)
- `data/processed/X_train_seq.npy`, ... (6 file sequence)
- `models/scaler.pkl`, `models/label_encoder.pkl`

---

## BƯỚC 5 — Huấn luyện Isolation Forest

```bash
python isolation_forest.py
```

Thời gian ước tính: **2–5 phút**.

Đầu ra:
- `models/isolation_forest.pkl`
- `results/if_score_distribution.png`

---

## BƯỚC 6 — Huấn luyện LSTM

```bash
python lstm_model.py
```

Thời gian ước tính: **15–40 phút** (CPU) hoặc **3–8 phút** (GPU).

Đầu ra:
- `models/lstm_best.h5`
- `results/lstm_learning_curve.png`
- `results/lstm_confusion_matrix.png`

> **Theo dõi training với TensorBoard:**
> ```bash
> # Mở terminal mới, chạy song song:
> tensorboard --logdir logs/lstm
> # Mở trình duyệt tại: http://localhost:6006
> ```

---

## BƯỚC 7 — Khởi động ứng dụng Web

```bash
python app.py
```

Mở trình duyệt tại: **http://localhost:5000**

---

## BƯỚC 8 — Kiểm thử hệ thống

### Cách 1: Qua giao diện web
1. Truy cập `http://localhost:5000`
2. Kéo thả hoặc click chọn file CSV (có thể dùng một phần file test)
3. Nhấn **"Bắt đầu phân tích"**
4. Chờ kết quả hiển thị (polling tự động mỗi 2 giây)

### Cách 2: Qua command line
```bash
python predict.py path/to/your_traffic.csv
```

### Cách 3: Qua REST API
```bash
curl -X POST http://localhost:5000/api/predict \
  -F "file=@path/to/traffic.csv"
```

### Kiểm tra sức khỏe hệ thống
```bash
curl http://localhost:5000/api/health
```

---

## Xử lý lỗi phổ biến

| Lỗi | Nguyên nhân | Giải pháp |
|---|---|---|
| `ModuleNotFoundError: No module named 'tensorflow'` | Chưa cài TF | `pip install tensorflow>=2.13.0` |
| `FileNotFoundError: data/raw` | Chưa tải dataset | Tải CICIDS2017 và đặt vào `data/raw/` |
| `OOM (Out of Memory)` | RAM không đủ | Giảm `batch_size` trong `lstm_model.py` xuống 128 |
| `Cannot import imbalanced` | Thiếu thư viện | `pip install imbalanced-learn` |
| `Port 5000 in use` | Cổng đang bị chiếm | Đổi port trong `app.py`: `port=5001` |
| `h5py` error khi load model | Phiên bản không tương thích | `pip install h5py==3.9.0` |

---

## Tái lập kết quả thực nghiệm

Để đảm bảo kết quả tái lập được, file đã thiết lập:
- `RANDOM_SEED = 42` trong tất cả các file
- `tf.random.set_seed(42)` và `np.random.seed(42)` trong `lstm_model.py`

Chạy toàn bộ pipeline một lần:
```bash
python preprocessing.py && python isolation_forest.py && python lstm_model.py
```

---

## PHỤ LỤC C — Tài nguyên tham khảo thêm

### Dataset
- CICIDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
- CICIDS2018: https://www.unb.ca/cic/datasets/ids-2018.html
- CIC-DDoS2019: https://www.unb.ca/cic/datasets/ddos-2019.html
- NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html

### Tài liệu thư viện
- scikit-learn IsolationForest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
- TensorFlow LSTM: https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
- imbalanced-learn SMOTE: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html

### Khóa học online liên quan
- "Deep Learning Specialization" – Andrew Ng (Coursera)
- "Machine Learning for Cybersecurity" – Udemy
- "Network Security with Python" – Pluralsight

### Repository tham khảo
- https://github.com/topics/network-intrusion-detection
- https://github.com/topics/cicids2017
- https://github.com/topics/anomaly-detection-network
