"""Task 2 — Streamlit 챗봇 웹 UI.

충남대학교 학내 정보 Q&A 챗봇 (블루 테마).

실행:
    streamlit run src/app/streamlit_app.py

설계 메모:
- Streamlit은 사용자 입력마다 스크립트 전체를 재실행하므로 무거운
  Retriever/모델은 @st.cache_resource 로 1회만 로드한다.
- generate_answer_stream 은 "누적 텍스트"를 yield 하므로, 델타가 아니라
  placeholder.markdown(partial) 로 갱신하며 스트리밍한다.
- 출처(_format_sources)는 HTML <div> 라서 unsafe_allow_html=True 로 렌더한다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# `streamlit run` 은 스크립트 폴더(src/app)만 sys.path에 넣으므로
# 프로젝트 루트를 직접 추가해 `import src...` 가 동작하게 한다.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from src.model.inference import fallback_answer, generate_answer_stream  # noqa: E402

_WELCOME = (
    "안녕하세요! 충남대학교 학내 정보를 안내하는 AI 챗봇이에요. 😊\n\n"
    "졸업요건 · 학사일정 · 공지사항 · 식단 · 셔틀버스 등 "
    "궁금한 걸 편하게 물어봐 주세요."
)

_EXAMPLES = [
    "컴퓨터융합학부 졸업 요건이 어떻게 되나요?",
    "이번 학기 수강신청은 언제야?",
    "오늘 기숙사 점심 메뉴 뭐야?",
    "셔틀버스 시간표 알려줘",
    "최근 공지사항 보여줘",
]

_CSS = """
<style>
.block-container { max-width: 820px; padding-top: 1.5rem; }
.title-area { text-align: center; padding: 8px 0 4px 0; }
.title-area .logo-icon {
    display: inline-flex; align-items: center; justify-content: center;
    width: 58px; height: 58px; border-radius: 16px;
    background: linear-gradient(135deg, #3B82F6, #1D4ED8);
    font-size: 1.7em; margin-bottom: 10px;
    box-shadow: 0 8px 20px rgba(37, 99, 235, 0.32);
}
.title-area h1 { font-size: 1.9em; font-weight: 800; color: #0B2F5E; margin: 0 0 4px 0; }
.title-area p { color: #5C6B84; font-size: 0.95em; margin: 0; }
.category-badges {
    display: flex; justify-content: center; gap: 8px; flex-wrap: wrap; margin: 14px 0 18px 0;
}
.category-badges span {
    background: #fff; color: #1D4ED8; padding: 6px 14px; border-radius: 999px;
    font-size: 0.82em; font-weight: 600; border: 1px solid #D4E1F4;
}
.source-row { margin-top: 10px; font-size: 0.85em; }
.source-label {
    color: #5C6B84; font-weight: 600; margin-right: 6px;
}
.source-row a {
    display: inline-block; background: #EFF4FE; color: #1D4ED8;
    padding: 3px 10px; border-radius: 999px; margin: 2px 4px 2px 0;
    text-decoration: none; border: 1px solid #D4E1F4;
}
</style>
"""

_HEADER = """
<div class="title-area">
    <div class="logo-icon">🎓</div>
    <h1>CNU Campus AI</h1>
    <p>충남대학교 학내 정보를 안내해 드릴게요</p>
</div>
<div class="category-badges">
    <span>🎯 졸업요건</span>
    <span>📢 공지사항</span>
    <span>📅 학사일정</span>
    <span>🍽️ 식단</span>
    <span>🚌 셔틀버스</span>
</div>
"""


@st.cache_resource(show_spinner="모델·검색기를 불러오는 중입니다...")
def _load_pipeline() -> tuple[Any, Any | None, Any | None]:
    """Retriever + (model, tokenizer)를 1회만 로드한다.

    Returns:
        (retriever, model_or_None, tokenizer_or_None) 튜플.
        모델 로드에 실패하면 RAG 폴백만 사용한다.
    """
    from src.rag.retriever import Retriever

    retriever = Retriever()

    model = None
    tokenizer = None
    try:
        from src.model.inference import load_inference_model

        model, tokenizer = load_inference_model()
    except Exception as e:  # noqa: BLE001 - UI에서 폴백으로 계속 동작
        print(f"[streamlit] 모델 로드 실패 (RAG만 사용): {e}")

    return retriever, model, tokenizer


# RAG 모드: 표시 라벨 → retriever 모드 값
_RAG_MODES = {
    "기본 (dense)": "naive",
    "Hybrid (dense+BM25)": "hybrid",
    "Rerank (cross-encoder)": "rerank",
    "Hybrid + Rerank": "hybrid_rerank",
}


def _answer(
    question: str,
    history: list[dict],
    mode: str = "naive",
    top_k: int = 5,
    use_tools: bool = True,
) -> Any:
    """질문에 대한 답변을 (가능하면 스트리밍 제너레이터로) 만든다.

    Args:
        question: 사용자 질문
        history: 직전 대화 이력 [{"role", "content"}, ...] (현재 질문 제외)
        mode: RAG 검색 모드 (naive/hybrid/rerank/hybrid_rerank)
        top_k: 검색 문서 수
        use_tools: LLM tool-calling 라우팅 사용 여부 (A/B 비교에서 B는 끈다)

    Returns:
        모델이 있으면 누적 텍스트를 yield 하는 제너레이터,
        없으면 완성된 폴백 답변 문자열.
    """
    retriever, model, tokenizer = _load_pipeline()
    context, urls = retriever.build_context(question, top_k=top_k, mode=mode)

    if model is not None and tokenizer is not None:
        # Gemma 챗 템플릿은 user/assistant 엄격 교대 + user 시작을 요구한다.
        # 맨 앞 환영 메시지(assistant)로 시작하면 깨지므로, user로 시작하도록
        # 앞쪽 비-user 메시지를 떼어낸다.
        prior = [m for m in history if m.get("content")]
        while prior and prior[0].get("role") != "user":
            prior.pop(0)
        return generate_answer_stream(
            question,
            context,
            urls,
            model,
            tokenizer,
            use_tools=use_tools,
            history=prior or None,
        )
    return fallback_answer(question, context, urls)


def _answer_final(
    question: str, history: list[dict], mode: str, top_k: int, use_tools: bool
) -> str:
    """A/B 후보용 — 답변을 끝까지 받아 완성 문자열 하나로 반환한다.

    `generate_answer_stream`은 누적 텍스트를 yield 하므로 마지막 값이 최종 답변이다.
    두 후보를 나란히 비교해야 하므로 스트리밍 대신 완성본을 만든다.
    """
    result = _answer(question, history, mode=mode, top_k=top_k, use_tools=use_tools)
    if isinstance(result, str):
        return result
    final = ""
    for partial in result:
        final = partial
    return final


def _render_ab_choice() -> None:
    """생성해 둔 두 후보(A 기본 / B 대안)를 나란히 보여주고, 고른 답변만 대화에 남긴다.

    Streamlit 버튼 클릭은 rerun을 유발하므로 후보는 st.session_state.ab_pending 에
    보존해 둔다. 사용자가 선택하면 해당 답변을 messages 에 append 하고 pending 을 비운다.
    """
    pend = st.session_state.ab_pending
    with st.chat_message("assistant"):
        st.markdown("**🤔 어떤 답변이 더 나은가요? 하나를 골라주세요.**")
        options = [
            ("A", "🅰️ 기본 (실시간·정밀 검색)", pend["A"]),
            ("B", "🅱️ 대안 (넓은 검색)", pend["B"]),
        ]
        for col, (key, label, text) in zip(st.columns(2), options):
            with col:
                st.markdown(f"**{label}**")
                st.markdown(text or "_(빈 답변)_", unsafe_allow_html=True)
                if st.button("이 답변 선택", key=f"ab_pick_{key}", use_container_width=True):
                    st.session_state.messages.append({"role": "assistant", "content": text})
                    st.session_state.pop("ab_pending", None)
                    st.rerun()


def main() -> None:
    """Streamlit 앱 본문."""
    st.set_page_config(page_title="CNU Q&A 챗봇", page_icon="🎓", layout="centered")
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(_HEADER, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": _WELCOME}]

    # 사이드바: 설정 톱니바퀴 + 예시 질문 + 초기화
    with st.sidebar:
        with st.expander("⚙️ 설정 (RAG)", expanded=False):
            mode_label = st.radio(
                "검색 방식",
                list(_RAG_MODES.keys()),
                index=0,
                help=(
                    "기본=의미검색만 · Hybrid=키워드(BM25) 결합 · "
                    "Rerank=정밀 재정렬 · Hybrid+Rerank=둘 다 (가장 정확, 느림)"
                ),
            )
            st.session_state.rag_mode = _RAG_MODES[mode_label]
            st.session_state.rag_top_k = st.slider("검색 문서 수 (top-k)", 3, 10, 5)
            st.session_state.ab_compare = st.checkbox(
                "응답 2개 비교 후 선택 (GPT식)",
                value=st.session_state.get("ab_compare", True),
                help=(
                    "답변을 기본(실시간·정밀)·대안(넓은 검색) 두 가지로 만들어 더 나은 쪽을 "
                    "고르게 합니다. 생성을 2회 하므로 응답이 느려져요."
                ),
            )
            st.caption("Rerank/Hybrid는 첫 사용 시 모델·인덱스 로딩으로 잠시 느릴 수 있어요.")

        st.subheader("💡 이런 질문을 해보세요")
        for ex in _EXAMPLES:
            if st.button(ex, use_container_width=True):
                st.session_state.pending = ex
        st.divider()
        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state.messages = [{"role": "assistant", "content": _WELCOME}]
            st.session_state.pop("pending", None)
            st.session_state.pop("ab_pending", None)
            st.rerun()

    # 기존 대화 렌더
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"], unsafe_allow_html=True)

    # A/B 후보가 대기 중이면 선택 UI만 렌더하고 새 입력은 막는다 (선택해야 다음 질문).
    if st.session_state.get("ab_pending"):
        _render_ab_choice()
        return

    # 입력: 채팅창 또는 사이드바 예시 버튼
    prompt = st.chat_input("궁금한 점을 물어보세요! (예: 졸업요건, 수강신청, 식단 등)")
    if not prompt and "pending" in st.session_state:
        prompt = st.session_state.pop("pending")

    if not prompt:
        return

    # 사용자 메시지
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 어시스턴트 응답
    history = st.session_state.messages[:-1]  # 현재 질문 제외
    mode = st.session_state.get("rag_mode", "naive")
    top_k = st.session_state.get("rag_top_k", 5)

    if st.session_state.get("ab_compare", True):
        # A/B: 두 후보를 완성본으로 생성 → 다음 rerun에서 _render_ab_choice 로 선택.
        #  A(기본) = 사이드바 모드 + 툴 ON → 툴 응답 또는 정밀 RAG
        #  B(대안) = naive(무컷오프 recall) + 툴 OFF → 코퍼스 RAG
        # 둘 다 RAG일 때는 정밀(A) vs recall(B) 대비, A가 툴일 때는 실시간 vs 코퍼스 대비.
        with st.chat_message("assistant"):
            with st.spinner("두 가지 답변을 준비하고 있어요... (조금 걸려요)"):
                cand_a = _answer_final(prompt, history, mode=mode, top_k=top_k, use_tools=True)
                cand_b = _answer_final(prompt, history, mode="naive", top_k=top_k, use_tools=False)
        st.session_state.ab_pending = {"question": prompt, "A": cand_a, "B": cand_b}
        st.rerun()
    else:
        # 단일 응답 (스트리밍)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            result = _answer(prompt, history, mode=mode, top_k=top_k)
            if isinstance(result, str):
                final = result
                placeholder.markdown(final, unsafe_allow_html=True)
            else:
                final = ""
                for partial in result:  # 누적 텍스트
                    final = partial
                    placeholder.markdown(final, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": final})


# `streamlit run` 은 스크립트를 __main__ 으로 실행한다. 한 번만 호출한다.
main()
