
import os
import uuid
import json
import time
import threading
from datetime import datetime

from flask import (Flask, request, jsonify, render_template,
                   redirect, url_for, flash, send_from_directory)
from werkzeug.utils import secure_filename

# Import pipeline dự đoán
from predict import predict_from_file, summarize_results, model_loader

# ─────────────────────────────────────────────
# CẤU HÌNH ỨNG DỤNG
# ─────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", "change-me-in-production")

UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", os.path.join(os.getcwd(), "uploads"))
ALLOWED_EXTENSIONS = {"csv"}
MAX_CONTENT_LENGTH = 200 * 1024 * 1024   # Tối đa 200 MB

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─────────────────────────────────────────────
# BỘ NHỚ KẾT QUẢ (In-memory store)
# Trong production nên dùng Redis hoặc database.
# ─────────────────────────────────────────────
results_store: dict = {}   # {job_id: {"status": ..., "results": ..., "summary": ...}}
store_lock = threading.Lock()


# ─────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return ("." in filename and
            filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS)


def run_prediction_async(job_id: str, filepath: str):
    """
    Chạy pipeline dự đoán trong thread riêng biệt để không block server.
    Cập nhật results_store khi hoàn thành.
    """
    try:
        with store_lock:
            results_store[job_id]["status"] = "processing"

        start_time = time.time()
        raw_results = predict_from_file(filepath)
        summary     = summarize_results(raw_results)
        elapsed     = round(time.time() - start_time, 2)

        # Chuyển kết quả thành JSON-serializable
        serializable = []
        for r in raw_results[:2000]:   # Giới hạn 2000 dòng trả về frontend
            serializable.append({
                "index"     : r["index"],
                "label"     : r["label"],
                "label_vi"  : r["label_vi"],
                "severity"  : r["severity"],
                "confidence": round(r["confidence"] * 100, 1),
                "if_score"  : round(r["if_score"], 4),
                "source"    : r["source"],
            })

        with store_lock:
            results_store[job_id].update({
                "status"       : "done",
                "results"      : serializable,
                "summary"      : summary,
                "elapsed_sec"  : elapsed,
                "total_rows"   : len(raw_results),
                "completed_at" : datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
            })

        # Xóa file upload sau khi xử lý xong
        if os.path.exists(filepath):
            os.remove(filepath)

    except Exception as exc:
        with store_lock:
            results_store[job_id].update({
                "status" : "error",
                "message": str(exc),
            })


# ─────────────────────────────────────────────
# ROUTES – GIAO DIỆN WEB
# ─────────────────────────────────────────────

@app.route("/")
def index():
    """Trang chủ — màn hình upload file."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    """
    Nhận file CSV từ form HTML, kiểm tra hợp lệ,
    lưu vào thư mục uploads, kích hoạt job dự đoán async,
    rồi redirect đến trang kết quả.
    """
    if "file" not in request.files:
        flash("Chưa chọn file.", "danger")
        return redirect(url_for("index"))

    f = request.files["file"]
    if f.filename == "":
        flash("Tên file không hợp lệ.", "danger")
        return redirect(url_for("index"))

    if not allowed_file(f.filename):
        flash("Chỉ hỗ trợ file .CSV.", "danger")
        return redirect(url_for("index"))

    # Tạo tên file duy nhất để tránh xung đột
    safe_name = secure_filename(f.filename)
    job_id    = str(uuid.uuid4())[:8]
    filename  = f"{job_id}_{safe_name}"
    filepath  = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(filepath)

    file_size_mb = round(os.path.getsize(filepath) / (1024 * 1024), 2)

    # Khởi tạo job
    with store_lock:
        results_store[job_id] = {
            "status"    : "queued",
            "filename"  : safe_name,
            "file_size" : file_size_mb,
            "job_id"    : job_id,
            "created_at": datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
        }

    # Chạy dự đoán trong background thread
    t = threading.Thread(target=run_prediction_async,
                         args=(job_id, filepath), daemon=True)
    t.start()

    return redirect(url_for("result_page", job_id=job_id))


@app.route("/result/<job_id>")
def result_page(job_id: str):
    """Trang hiển thị kết quả — polling từ frontend đến API."""
    with store_lock:
        job = results_store.get(job_id)
    if not job:
        flash("Không tìm thấy job.", "warning")
        return redirect(url_for("index"))
    return render_template("result.html", job_id=job_id, job=job)


@app.route("/dashboard")
def dashboard():
    """Trang dashboard tổng hợp toàn bộ jobs."""
    with store_lock:
        jobs = list(results_store.values())
    return render_template("dashboard.html", jobs=jobs)


# ─────────────────────────────────────────────
# ROUTES – REST API
# ─────────────────────────────────────────────

@app.route("/api/status/<job_id>")
def api_status(job_id: str):
    """
    API polling — frontend gọi mỗi 2 giây để kiểm tra tiến trình.
    Trả về JSON với status: queued | processing | done | error.
    """
    with store_lock:
        job = results_store.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    API trực tiếp — nhận file CSV qua multipart/form-data,
    trả về kết quả dự đoán dạng JSON (đồng bộ, phù hợp file nhỏ < 5000 dòng).
    """
    if "file" not in request.files:
        return jsonify({"error": "Thiếu file"}), 400

    f = request.files["file"]
    if not allowed_file(f.filename):
        return jsonify({"error": "Chỉ hỗ trợ .CSV"}), 400

    job_id   = str(uuid.uuid4())[:8]
    filename = f"{job_id}_{secure_filename(f.filename)}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(filepath)

    try:
        start      = time.time()
        raw_results = predict_from_file(filepath)
        summary    = summarize_results(raw_results)
        elapsed    = round(time.time() - start, 2)

        return jsonify({
            "job_id"      : job_id,
            "elapsed_sec" : elapsed,
            "total_rows"  : len(raw_results),
            "summary"     : summary,
            "results"     : raw_results[:500],   # Giới hạn để tránh payload quá lớn
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/api/jobs")
def api_jobs():
    """Trả về danh sách tất cả jobs và trạng thái."""
    with store_lock:
        jobs = {jid: {k: v for k, v in job.items() if k != "results"}
                for jid, job in results_store.items()}
    return jsonify(jobs)


@app.route("/api/health")
def health_check():
    """Health check endpoint — dùng khi deploy với Docker/Kubernetes."""
    return jsonify({
        "status"      : "ok",
        "model_loaded": model_loader.is_loaded,
        "timestamp"   : datetime.now().isoformat(),
    })


# ─────────────────────────────────────────────
# KHỞI ĐỘNG SERVER
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Preload model khi khởi động để request đầu tiên không bị chậm
    print("  [App] Đang preload models...")
    model_loader.load()
    print("  [App] Khởi động server tại http://0.0.0.0:5000")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,     # Tắt debug mode khi deploy
        threaded=True,
    )
