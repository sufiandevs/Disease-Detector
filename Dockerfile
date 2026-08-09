FROM python:3.11-slim

WORKDIR /app

# Install build tools (some Python packages need gcc)
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire app
COPY . .

# HF Spaces expects port 7860
EXPOSE 7860

# Run with gunicorn (1 worker to save RAM, 120s timeout for Drive download)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1"]
