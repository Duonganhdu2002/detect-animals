# Sử dụng image Python chính thức
FROM python:3.9-slim

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Đặt thư mục làm việc trong container
WORKDIR /app

# Sao chép file requirements.txt vào container
COPY requirements.txt .

# Cài đặt các thư viện cần thiết
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn và mô hình vào container
COPY . .
COPY ~/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth /root/.cache/torch/hub/checkpoints/

# Đặt biến môi trường cho Flask (hoặc framework khác nếu bạn sử dụng)
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Expose cổng mà ứng dụng sẽ chạy (ví dụ: Flask mặc định là 5000)
EXPOSE 5000

# Chạy ứng dụng
CMD ["python", "app.py"]
