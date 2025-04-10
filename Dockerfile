FROM python:3.9-slim

WORKDIR /app

# 🧱 Install system dependencies for OpenCV and TensorFlow
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 📁 Copy your project files
COPY . .

# 📦 Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# 🌐 Expose the Flask port
EXPOSE 5000

# 🚀 Run the app
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]