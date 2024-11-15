# Sử dụng image Python chính thức
FROM python:3.9-slim

# Cài đặt thư viện hệ thống cần thiết cho OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Đặt thư mục làm việc trong container
WORKDIR /app

# Sao chép file requirements.txt vào container
COPY requirements.txt .

# Cài đặt các thư viện cần thiết
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn vào container
COPY . .

# Đặt biến môi trường cho Flask
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Expose cổng mà ứng dụng sẽ chạy (ví dụ: Flask mặc định là 5000)
EXPOSE 5000

# Chạy ứng dụng với Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
