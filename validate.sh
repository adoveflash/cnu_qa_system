#!/usr/bin/env bash
# 재빌드 검증: 청크 통계 + 5카테고리 답변 + 졸업 답변을 rebuild_check.txt로 모아 push.
# 사용법: bash validate.sh
set -e

echo "[1/3] 청크 통계..."
python rag_check.py > rebuild_check.txt 2>&1

echo "[2/3] 5개 카테고리 답변 생성... (수 분)"
CUDA_VISIBLE_DEVICES=0 python answer_test.py >> rebuild_check.txt 2>&1

echo "[3/3] 졸업 답변..."
CUDA_VISIBLE_DEVICES=0 python debug_grad.py >> rebuild_check.txt 2>&1

git add rebuild_check.txt
git commit -q -m "wip: 재빌드 검증" || true
git push origin main
echo "✅ done — rebuild_check.txt push 완료"
