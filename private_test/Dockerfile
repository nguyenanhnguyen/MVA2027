# Gói nộp private test (cả hai track dùng chung ảnh này). CHƯA ĐƯỢC KIỂM THỬ: phải build và chạy thử trên
# Linux/WSL2 và đối chiếu điểm với CodaBench trước hạn 17/2/2027 (xem docs/private_test.md của repo baseline).
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY configs ./configs
COPY private_test/run_track1.sh private_test/run_track2.sh ./
# Trọng số phải nằm trong gói (không có internet khi chạy): models/det_final.pt, models/pred.pt
COPY models ./models

ENV PYTHONPATH=/app/src SMOTP_MODELS=/app/models SMOTP_WORK=/tmp/smotp_work PYTHONIOENCODING=utf-8 YOLO_OFFLINE=1
