# Dockerfile

# FROM python:3.12-slim
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    python \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# CUDA 호환성 패키지 (선택적)
# 실제 CUDA 지원이 필요하면 nvidia/cuda 베이스 이미지를 사용해야 합니다
# 이 예제에서는 CPU 모드로 실행을 가정합니다

# Python 패키지 설치를 위한 requirements.txt 복사
COPY requirements.txt .

# 파이썬 패키지 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 필요한 디렉토리 생성
RUN mkdir -p /app/models/downloads && \
    mkdir -p /app/data/training && \
    mkdir -p /app/temp

# 환경 변수 설정
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    DEVICE=cpu \
    MODEL_NAME=llava-hf/llava-1.5-7b-hf \
    PRELOAD_MODEL=false

# 포트 노출
EXPOSE 8000

# 시작 명령
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# 헬스체크
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health/ || exit 1