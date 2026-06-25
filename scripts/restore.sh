#!/usr/bin/env bash
# 박스 벡터DB 복구(병합): 박스 raw(여러 학과: pharm/biz/chem/math/econ ...) +
# Mac 코어(식단·dorm·plus_kr·computer 깊은 커버리지) 를 합쳐 가장 풍부한 챗봇 코퍼스를 만든다.
# (5개 카테고리는 Task1 분류기 얘기. Task2 챗봇은 넓을수록 좋다.)
#
# 실행 전 반드시: git fetch origin && git reset --hard origin/main   (Mac 코어 cleaned.jsonl 받기)
# 그 다음(레포 루트에서): bash scripts/restore.sh
set -e

# CUDA 라이브러리 가드: 시스템 옛 libnvJitLink가 torch 휠보다 먼저 잡혀
# cusparse __nvJitLinkComplete_12_4 심볼 누락으로 torch import가 죽는 걸 우회.
_NVJIT_LIB="$(python -c 'import os,nvidia.nvjitlink as m; print(os.path.dirname(m.__file__)+"/lib")' 2>/dev/null || true)"
if [ -n "$_NVJIT_LIB" ]; then
    export LD_LIBRARY_PATH="${_NVJIT_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

CORE=/tmp/cleaned_core.jsonl
BOX=/tmp/cleaned_box.jsonl
MERGED=/tmp/cleaned_merged.jsonl

echo "[0/4] Mac 코어(마스킹됨) 백업..."
cp data/corpus/cleaned.jsonl "$CORE"

echo "[1/4] 박스 raw 재정제+마스킹 (여러 학과)..."
python -c "
from pathlib import Path
from src.data.cleaner import clean_all
from src.data.pii_masker import mask_file
clean_all(Path('data/corpus/raw'), Path('$BOX'))
mask_file(Path('$BOX'), Path('$BOX'))
"

echo "[2/4] 병합 (코어 + 학과) → 재청킹..."
cat "$CORE" "$BOX" > "$MERGED"
CUDA_VISIBLE_DEVICES=0 python -c "from pathlib import Path; from src.data.chunker import chunk_all; chunk_all(Path('$MERGED'), Path('data/corpus/chunks.jsonl'))"

echo "[3/4] 벡터DB 재구축 (bge-m3 재임베딩)..."
CUDA_VISIBLE_DEVICES=0 python scripts/build_db.py

echo "[4/4] manual 큐레이션 청크 투입..."
CUDA_VISIBLE_DEVICES=0 python scripts/add_manual.py

echo "✅ 병합 복구 완료 (박스 학과 + Mac 코어) — 'python scripts/rag_check.py'로 확인"
