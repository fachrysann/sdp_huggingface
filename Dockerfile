# 1. Gunakan image Python 3.12 versi slim agar ukuran file lebih kecil
FROM python:3.12-slim

# 2. Set direktori kerja di dalam container Docker
WORKDIR /app

# 3. Install library sistem operasi (Linux) yang dibutuhkan
# libgl1 & libglib2.0-0 -> Dibutuhkan oleh OpenCV (Updated untuk Linux terbaru)
# libsndfile1 -> Dibutuhkan oleh library Soundfile (Audio)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy file requirements.txt ke dalam container
COPY requirements.txt .

# 5. Install semua library Python
# Menggunakan flag --extra-index-url agar memaksa download PyTorch versi CPU (Bukan GPU/CUDA yang sizenya 2GB+)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 6. Copy seluruh folder dan file project (kecuali yang ada di .dockerignore)
COPY . .

# 7. Buka port 8000 agar bisa diakses dari luar
EXPOSE 8000

# 8. Perintah untuk menjalankan Gunicorn Server
# Menggunakan 2 worker uvicorn. Timeout diset 120 detik untuk jaga-jaga proses ML agak lama.
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "--timeout", "120", "app.main:app", "-b", "0.0.0.0:8000"]