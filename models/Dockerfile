FROM python:3.12.3-slim

WORKDIR /app

# Install system dependencies (runtime + build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Runtime dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Build dependencies (needed for insightface)
    g++ \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip && \
    # Remove build tools to save space
    apt-get purge -y --auto-remove g++ gcc && \
    apt-get clean


COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
