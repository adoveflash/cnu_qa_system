#!/bin/bash
# 제출용 zip 생성 스크립트
# 사용법: bash scripts/make_submission.sh

set -e

NAME="장윤상"
ZIP_NAME="Termproject_${NAME}.zip"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "제출 파일 생성: $ZIP_NAME"
echo ""

# 필수 파일 존재 확인
echo "=== 필수 파일 확인 ==="
for f in \
    src/classifier.ipynb \
    chatbot.sh \
    requirements.txt \
    README.md \
    data/train.json \
    data/valid.json \
    data/test_cls.json \
    data/test_chat.json \
    model/README.md \
    model/download_model.py \
    ; do
    if [ -f "$f" ]; then
        echo "  ✅ $f"
    else
        echo "  ❌ $f (없음!)"
    fi
done

echo ""
echo "=== 출력 파일 확인 ==="
for f in outputs/cls_output.json outputs/chat_output.json outputs/realtime_output.json; do
    if [ -f "$f" ]; then
        echo "  ✅ $f"
    else
        echo "  ⚠️  $f (아직 생성 안됨)"
    fi
done

echo ""
echo "=== zip 생성 중 ==="

# 불필요 파일 제외하고 압축
zip -r "$ZIP_NAME" \
    data/train.json \
    data/valid.json \
    data/test_cls.json \
    data/test_chat.json \
    data/test_realtime.json \
    data/corpus/chunks.jsonl \
    data/corpus/cleaned.jsonl \
    data/qa/ \
    src/ \
    model/ \
    outputs/ \
    chatbot.sh \
    requirements.txt \
    README.md \
    submission.ipynb \
    -x "*.pyc" \
    -x "*/__pycache__/*" \
    -x "*/.ipynb_checkpoints/*" \
    -x "*.DS_Store" \
    2>/dev/null

SIZE=$(du -h "$ZIP_NAME" | cut -f1)
echo ""
echo "=== 완료 ==="
echo "  파일: $ZIP_NAME ($SIZE)"
echo ""
echo "⚠️  vector_db는 용량이 커서 제외됨 → HF Hub에서 다운로드"
echo "⚠️  발표자료, UI 영상은 별도 추가 필요"
