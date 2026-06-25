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

# 3. PyTorch — Gemma 4는 torch 2.7+ 필요(float8_e8m0fnu).
# 드라이버 CUDA 12.4 박스는 cu126 빌드로 설치(cu130 기본 빌드는 드라이버가 거부,
# cu124는 torch 2.6이 상한이라 gemma-4 불가). minor-compat으로 12.4 드라이버에서 동작.
echo "[3/4] PyTorch 설치 (torch 2.12.1 / cu126)..."
pip install torch==2.12.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 4. 나머지 의존성 (torch는 위에서 설치했으므로 제외)
echo "[4/4] 프로젝트 의존성 설치..."
grep -viE '^(torch|torchvision|torchaudio)' "$PROJECT_DIR/requirements.txt" | pip install -r /dev/stdin

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
