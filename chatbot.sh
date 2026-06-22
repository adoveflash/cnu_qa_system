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

# ── CUDA 라이브러리 가드 ──
# 시스템의 옛 libnvJitLink.so.12가 torch 휠보다 먼저 잡히면 cusparse가
# __nvJitLinkComplete_12_4 심볼을 못 찾아 torch import가 죽는다.
# pip 휠의 nvjitlink lib을 LD_LIBRARY_PATH 맨 앞에 둬서 우회한다.
_NVJIT_LIB="$(python -c 'import os,nvidia.nvjitlink as m; print(os.path.dirname(m.__file__)+"/lib")' 2>/dev/null || true)"
if [ -n "$_NVJIT_LIB" ]; then
    export LD_LIBRARY_PATH="${_NVJIT_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

echo "============================================"
echo "  CNU Q&A 챗봇 — Task 2 & 3"
echo "============================================"

# ── 의존성 확인 ──
echo ""
echo "[1/4] 의존성 확인..."
pip install -q -r requirements.txt 2>/dev/null || true

# ── outputs 디렉터리 생성 ──
mkdir -p outputs

# ── 벡터 DB 준비 (없으면 HuggingFace 공개 저장소에서 다운로드) ──
if [ ! -d "data/vector_db" ]; then
    echo ""
    echo "[*] 벡터 DB가 없어 HuggingFace에서 다운로드합니다..."
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='adoveflash/cnu-qa-system', repo_type='model', local_dir='.', allow_patterns=['data/vector_db/**'])"
fi

# ── Task 3: 실시간 정보 (Optional) ──
echo ""
echo "[2/4] Task 3 — 실시간 정보 반영..."
if [ -f "data/test_realtime.json" ]; then
    # realtime_model은 --batch-only를 모르므로 제거하고 나머지 인자만 전달
    RT_ARGS=()
    for _a in "$@"; do [ "$_a" != "--batch-only" ] && RT_ARGS+=("$_a"); done
    python -m src.realtime_model "${RT_ARGS[@]}" || echo "  [경고] Task 3 실행 실패 — 건너뜀"
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
    # Streamlit UI 실행 (프로젝트 루트를 PYTHONPATH에 둬서 `import src...` 동작)
    PYTHONPATH="$SCRIPT_DIR" streamlit run src/app/streamlit_app.py \
        --server.port=8501 --server.address=0.0.0.0
fi
