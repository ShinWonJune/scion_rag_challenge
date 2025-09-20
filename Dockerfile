# 1. 베이스 이미지: Ubuntu 22.04 + CUDA 12.1 개발 환경
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 2. 시스템 환경 설정 및 Python 3.11 설치 (3.12 대신 안정적인 3.11 사용)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    git \
    build-essential \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

# 3. python3.11을 기본 python으로 설정
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# 4. pip 설치 (get-pip.py 사용)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
RUN python3 -m pip install --upgrade pip setuptools wheel

# 5. PyTorch 설치 (CUDA 12.1 호환)
RUN pip install \
    torch==2.8.0 \
    transformers==4.56.1 \
    torchvision==0.23.0 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# 6. vLLM 및 핵심 라이브러리 설치
RUN pip install \
    vllm==0.10.2 \
    openai>=1.0.0 \
    httpx>=0.24.0

# 7. 작업 디렉토리 설정
WORKDIR /app

# 8. 프로젝트 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 9. 소스 코드 복사
COPY . .

# 10. 포트 및 기본 명령어
EXPOSE 8000
CMD ["tail", "-f", "/dev/null"]
