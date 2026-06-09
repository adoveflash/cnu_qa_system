"""Gradio 챗봇 웹 UI 모듈.

충남대학교 학내 정보 Q&A 챗봇 인터페이스.
Gradio 4.x/5.x (tuples) 및 6.x (messages) 자동 호환.
"""

from __future__ import annotations

from typing import Any

import gradio as gr

from src.model.inference import fallback_answer, generate_answer_stream

# Gradio 5.x는 type="messages" 필요, 6.x는 기본이 messages (파라미터 제거됨), 4.x는 tuples
_GRADIO_MAJOR = int(gr.__version__.split(".")[0])
_USE_MESSAGES = _GRADIO_MAJOR >= 5
_NEED_TYPE_PARAM = _GRADIO_MAJOR == 5  # 6.x는 type 파라미터 없음

_CSS = """
/* ===== Claude 스타일: 따뜻한 아이보리 + 점토색(clay) 포인트 + 세리프 ===== */
:root {
    --clay: #D97757;          /* Claude 시그니처 점토색 */
    --clay-deep: #C15F3C;
    --ivory: #FAF9F5;         /* 따뜻한 배경 */
    --cream: #F0EEE6;         /* 카드/칩 배경 */
    --ink: #2A2A28;           /* 본문 텍스트 (웜 블랙) */
    --muted: #6B6B63;         /* 보조 텍스트 */
    --line: #E5E3DA;          /* 경계선 */
}
.gradio-container {
    max-width: 740px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Apple SD Gothic Neo', sans-serif !important;
    background: var(--ivory) !important;
    color: var(--ink) !important;
}
body, .gradio-container { background: var(--ivory) !important; }
.title-area {
    text-align: center;
    padding: 44px 0 10px 0;
    background: transparent;
    border: none;
    box-shadow: none;
}
.title-area .logo-icon {
    font-size: 2.4em;
    color: var(--clay);
    line-height: 1;
    margin-bottom: 10px;
}
.title-area h1 {
    font-family: 'Tiempos Text', Georgia, 'Times New Roman', serif;
    font-size: 2.0em;
    font-weight: 500;
    margin-bottom: 6px;
    color: var(--ink);
    letter-spacing: -0.4px;
}
.title-area p {
    color: var(--muted);
    font-size: 0.96em;
    margin-top: 0;
}
.category-badges {
    display: flex;
    justify-content: center;
    gap: 7px;
    flex-wrap: wrap;
    margin: 18px 0 22px 0;
}
.category-badges span {
    background: var(--cream);
    color: var(--muted);
    padding: 5px 13px;
    border-radius: 8px;
    font-size: 0.82em;
    font-weight: 500;
    border: 1px solid var(--line);
    transition: all 0.18s ease;
}
.category-badges span:hover {
    background: #fff;
    color: var(--clay-deep);
    border-color: var(--clay);
}
footer {visibility: hidden}
/* 채팅 영역 */
.chatbot {
    border-radius: 18px !important;
    border: 1px solid var(--line) !important;
    background: #FFFDF9 !important;
    box-shadow: none !important;
}
/* 메시지 말풍선 (Gradio 버전별 방어적 타겟팅) */
.chatbot .message.user, .chatbot [data-testid="user"], .message-row.user-row .message {
    background: var(--cream) !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    color: var(--ink) !important;
}
.chatbot .message.bot, .chatbot [data-testid="bot"], .message-row.bot-row .message {
    background: transparent !important;
    border: none !important;
    color: var(--ink) !important;
}
/* 전송 버튼 */
button.primary {
    background: var(--clay) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    color: #fff !important;
    transition: background 0.18s ease !important;
}
button.primary:hover {
    background: var(--clay-deep) !important;
}
/* 보조 버튼 (초기화) */
button.secondary {
    background: transparent !important;
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
    color: var(--muted) !important;
}
button.secondary:hover {
    border-color: var(--clay) !important;
    color: var(--clay-deep) !important;
}
/* 입력창 */
textarea {
    border-radius: 12px !important;
    border: 1px solid var(--line) !important;
    background: #FFFDF9 !important;
    color: var(--ink) !important;
    font-size: 1.0em !important;
}
textarea:focus {
    border-color: var(--clay) !important;
    box-shadow: 0 0 0 3px rgba(217, 119, 87, 0.13) !important;
}
/* 출처 배지 */
.source-row {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
    margin-top: 12px;
    padding-top: 11px;
    border-top: 1px solid var(--line);
}
.source-label {
    font-size: 0.72em;
    color: var(--muted);
    font-weight: 600;
    margin-right: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.source-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: var(--cream);
    border: 1px solid var(--line);
    color: var(--clay-deep);
    padding: 4px 11px;
    border-radius: 8px;
    font-size: 0.76em;
    font-weight: 500;
    box-shadow: none;
}
/* 예시 질문 */
.examples-row button, .examples button {
    border-radius: 10px !important;
    font-size: 0.86em !important;
    background: #FFFDF9 !important;
    border: 1px solid var(--line) !important;
    color: var(--ink) !important;
}
.examples-row button:hover, .examples button:hover {
    border-color: var(--clay) !important;
    background: #fff !important;
}
"""

_EXAMPLES = [
    "컴퓨터융합학부 졸업 요건이 어떻게 되나요?",
    "이번 학기 수강신청은 언제야?",
    "오늘 기숙사 점심 메뉴 뭐야?",
    "셔틀버스 시간표 알려줘",
    "최근 공지사항 보여줘",
]

_WELCOME = (
    "안녕하세요! 충남대학교 학내 정보를 안내하는 AI 챗봇이에요. 😊\n\n"
    "졸업요건 · 학사일정 · 공지사항 · 식단 · 셔틀버스 등 "
    "궁금한 걸 편하게 물어봐 주세요."
)


def _initial_history() -> list:
    """Gradio 버전에 맞는 초기 환영 메시지 이력을 만든다."""
    if _USE_MESSAGES:
        return [{"role": "assistant", "content": _WELCOME}]
    return [[None, _WELCOME]]


def _extract_text(content: Any) -> str:
    """Gradio content에서 순수 텍스트를 추출한다."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("value", content.get("text", str(content)))
    return str(content)


def create_app(
    retriever: Any,
    model: Any | None = None,
    tokenizer: Any | None = None,
) -> gr.Blocks:
    """Gradio 챗봇 UI를 생성한다."""

    with gr.Blocks(css=_CSS, title="CNU Q&A 챗봇") as app:
        gr.HTML(
            """
            <div class="title-area">
                <div class="logo-icon">&#10037;</div>
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
        )

        chatbot_kwargs = {"height": 520, "show_label": False, "container": False}
        if _NEED_TYPE_PARAM:
            chatbot_kwargs["type"] = "messages"
        chatbot = gr.Chatbot(
            **chatbot_kwargs, value=_initial_history(), elem_classes=["chatbot"]
        )

        with gr.Row():
            msg = gr.Textbox(
                placeholder="궁금한 점을 물어보세요! (예: 졸업요건, 수강신청, 식단 등)",
                show_label=False,
                scale=9,
                container=False,
            )
            submit_btn = gr.Button("전송", variant="primary", scale=1, min_width=80)

        with gr.Row():
            gr.Examples(
                examples=_EXAMPLES,
                inputs=msg,
                label="💡 이런 질문을 해보세요",
            )

        with gr.Row():
            clear_btn = gr.Button("🗑️ 대화 초기화", variant="secondary", size="sm")

        def respond(message: str, history: list):
            if not message.strip():
                yield "", history
                return

            question = _extract_text(message)
            context, urls = retriever.build_context(question)

            if _USE_MESSAGES:
                history = history + [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": ""},
                ]
            else:
                history = history + [[question, ""]]

            if model is not None and tokenizer is not None:
                # 멀티턴: 이전 대화 이력을 모델에 전달
                chat_history = None
                if _USE_MESSAGES and len(history) > 2:
                    prior = [
                        {"role": h["role"], "content": h["content"]}
                        for h in history[:-2]
                        if isinstance(h, dict) and h.get("content")
                    ]
                    # 환영 메시지 등 user 이전 메시지 제거 → 반드시 user로 시작
                    while prior and prior[0]["role"] != "user":
                        prior.pop(0)
                    chat_history = prior or None
                for partial in generate_answer_stream(
                    question, context, urls, model, tokenizer, history=chat_history
                ):
                    if _USE_MESSAGES:
                        history[-1]["content"] = partial
                    else:
                        history[-1][1] = partial
                    yield "", history
            else:
                answer = fallback_answer(question, context, urls)
                if _USE_MESSAGES:
                    history[-1]["content"] = answer
                else:
                    history[-1][1] = answer
                yield "", history

        submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear_btn.click(lambda: (_initial_history(), ""), outputs=[chatbot, msg])

    return app


def launch(
    retriever: Any,
    model: Any | None = None,
    tokenizer: Any | None = None,
    share: bool = True,
) -> None:
    """Gradio 챗봇 UI를 실행한다."""
    app = create_app(retriever, model, tokenizer)
    print("[ui] Gradio UI 시작...")
    app.launch(share=share)
