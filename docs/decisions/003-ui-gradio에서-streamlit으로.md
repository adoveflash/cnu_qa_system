# 003 — UI: Gradio → Streamlit 전환

## 결정 사항

Task 2 챗봇 웹 UI를 Gradio에서 **Streamlit**으로 전환한다.

- 신규 앱: `src/app/streamlit_app.py`
- 평가 진입점 `chatbot.sh`의 UI 실행 단계를 `streamlit run src/app/streamlit_app.py`로 변경
- `src/chatbot_ui.py:launch_ui()`도 streamlit 서브프로세스를 띄우도록 변경
- 기존 Gradio(`src/app/ui.py`)는 삭제하지 않고 레거시 폴백으로 유지

## 검토한 대안

- **Gradio 유지**: CLAUDE.md의 고정 기술 결정이었으나, 외부(과제 요구사항)에서 Streamlit 사용을 요구.
- **Gradio 완전 제거**: 한글 말풍선 세로깨짐 등 그동안 쌓인 Gradio 노하우([[project_ui_korean_bug]])가 사라지므로, 파일은 남겨 폴백으로 둔다.

## 선택 이유

- 과제/외부 요구사항이 Streamlit 명시 → 단순 취향 변경이 아닌 외부 제약.
- Streamlit은 입력마다 스크립트를 재실행하므로 무거운 Retriever/모델은
  `@st.cache_resource`로 1회만 로드 (핵심 차이점).
- `generate_answer_stream`은 누적 텍스트를 yield하므로 `st.empty()` placeholder를
  매 partial마다 갱신하는 방식으로 스트리밍 구현.
- 출처는 HTML(`<div class="source-row">`)이라 `unsafe_allow_html=True`로 렌더.
- `streamlit run`은 스크립트 폴더만 sys.path에 넣으므로, 앱 상단에서 프로젝트
  루트를 sys.path에 추가하고 `chatbot.sh`는 `PYTHONPATH`를 함께 전달.

## 주의

- 한글 렌더링: Streamlit은 기본 마크다운 렌더라 Gradio에서 겪던 말풍선
  세로깨짐([[project_ui_korean_bug]]) 문제는 발생하지 않음. 메시지 레이아웃
  CSS는 최소한만 사용.
