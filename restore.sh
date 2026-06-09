#!/usr/bin/env bash
# 박스 벡터DB 복구: Mac에서 올린 '좋은' cleaned.jsonl(식단·학사일정·plus_kr 깊은 커버리지,
# 정제+마스킹 완료)로 재청킹 → 재임베딩 → manual 청크 투입.
#
# 실행 전 반드시 좋은 cleaned.jsonl을 받아야 함:
#   git fetch origin && git reset --hard origin/main
# 그 다음:
#   bash restore.sh
set -e

echo "[1/3] 재청킹 (cleaned.jsonl → chunks.jsonl)..."
CUDA_VISIBLE_DEVICES=0 python -c "from pathlib import Path; from src.data.chunker import chunk_all; chunk_all(Path('data/corpus/cleaned.jsonl'), Path('data/corpus/chunks.jsonl'))"

echo "[2/3] 벡터DB 재구축 (bge-m3 재임베딩)..."
CUDA_VISIBLE_DEVICES=0 python build_db.py

echo "[3/3] manual 큐레이션 청크 투입..."
CUDA_VISIBLE_DEVICES=0 python add_manual.py

echo "✅ 복구 완료 — 'bash validate.sh'로 품질 확인하세요"
