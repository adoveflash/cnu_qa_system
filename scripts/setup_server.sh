#!/bin/bash
# 연구실 서버 환경 구축 스크립트
# 사용법: bash scripts/setup_server.sh

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="cnu_qa"
PYTHON_VER="3.10"

echo "=== 충남대 QA 시스템 - 서버 환경 구축 ==="
echo "프로젝트 경로: $PROJECT_DIR"

# 1. conda 환경 생성
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "[SKIP] conda 환경 '$ENV_NAME' 이미 존재"
else
    echo "[1/4] conda 환경 생성 (Python $PYTHON_VER)..."
    conda create -n "$ENV_NAME" python="$PYTHON_VER" -y
fi

# 2. 환경 활성화
echo "[2/4] 환경 활성화..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# 3. PyTorch (CUDA 12.4 호환)
echo "[3/4] PyTorch 설치 (CUDA 12.4)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. 나머지 의존성
echo "[4/4] 프로젝트 의존성 설치..."
pip install -r "$PROJECT_DIR/requirements.txt"

# 5. 확인
echo ""
echo "=== 설치 확인 ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_mem / 1024**3
        print(f'  GPU {i}: {name} ({mem:.0f} GB)')
"

echo ""
echo "=== 완료! ==="
echo "학습 실행: conda activate $ENV_NAME && python scripts/run_train.py"
