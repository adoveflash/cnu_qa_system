#!/bin/bash
# chatbot.sh — Task 2 & Task 3 평가 진입점
#
# 실행 순서:
#   1. pip 의존성 설치
#   2. (Task 3) live 컬렉션 갱신 + realtime_output.json 생성
#   3. (Task 2) chat_output.json 생성 + 챗봇 UI 실행
#
# 사용법:
#   bash chatbot.sh              # 전체 (갱신 + 배치 + UI)
#   bash chatbot.sh --batch-only # JSON만 생성 (UI 미실행)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  CNU Q&A 챗봇 — Task 2 & 3"
echo "============================================"

# ── 의존성 확인 ──
echo ""
echo "[1/4] 의존성 확인..."
pip install -q -r requirements.txt 2>/dev/null || true

# ── outputs 디렉터리 생성 ──
mkdir -p outputs

# ── Task 3: 실시간 정보 (Optional) ──
echo ""
echo "[2/4] Task 3 — 실시간 정보 반영..."
if [ -f "data/test_realtime.json" ]; then
    python -m src.realtime_model "$@" || echo "  [경고] Task 3 실행 실패 — 건너뜀"
else
    echo "  data/test_realtime.json 없음 — 건너뜀"
fi

# ── Task 2: 배치 추론 ──
echo ""
echo "[3/4] Task 2 — 챗봇 배치 추론 (chat_output.json)..."
python -m src.chatbot_ui --batch-only "$@" || echo "  [경고] 배치 추론 실패"

# ── Task 2: UI 실행 ──
echo ""
echo "[4/4] Task 2 — 챗봇 UI 실행..."

# --batch-only 플래그가 있으면 UI 건너뜀
if echo "$@" | grep -q "batch-only"; then
    echo "  --batch-only 모드 — UI 건너뜀"
    echo ""
    echo "============================================"
    echo "  완료! 결과 파일:"
    echo "    - outputs/chat_output.json"
    [ -f "outputs/realtime_output.json" ] && echo "    - outputs/realtime_output.json"
    echo "============================================"
else
    python -m src.chatbot_ui --ui-only "$@"
fi
