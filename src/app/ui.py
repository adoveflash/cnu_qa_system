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
.gradio-container {
    max-width: 850px !important;
    margin: 0 auto !important;
    font-family: 'Pretendard', 'Apple SD Gothic Neo', -apple-system, sans-serif !important;
    background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%) !important;
}
.title-area {
    text-align: center;
    padding: 28px 0 14px 0;
    background: linear-gradient(135deg, #1e3a5f 0%, #2d6da3 100%);
    border-radius: 16px;
    margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(30, 58, 95, 0.15);
}
.title-area h1 {
    font-size: 1.9em;
    margin-bottom: 6px;
    color: #ffffff;
    letter-spacing: -0.5px;
}
.title-area .logo-icon {
    font-size: 2.2em;
    margin-bottom: 4px;
}
.title-area p {
    color: #c8ddf0;
    font-size: 0.95em;
    margin-top: 0;
}
.category-badges {
    display: flex;
    justify-content: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 16px;
}
.category-badges span {
    background: #eef6ff;
    color: #1a5276;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.83em;
    font-weight: 500;
    border: 1px solid #cce0f0;
    transition: all 0.2s;
}
.category-badges span:hover {
    background: #d4ebff;
    transform: translateY(-1px);
}
footer {visibility: hidden}
/* 채팅 영역 스타일 */
.chatbot {
    border-radius: 12px !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
}
/* 전송 버튼 */
button.primary {
    background: linear-gradient(135deg, #1e3a5f, #2d6da3) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}
button.primary:hover {
    opacity: 0.9 !important;
}
/* 입력창 */
textarea {
    border-radius: 10px !important;
    border: 1.5px solid #d0dbe6 !important;
}
textarea:focus {
    border-color: #2d6da3 !important;
    box-shadow: 0 0 0 3px rgba(45, 109, 163, 0.1) !important;
}
/* 출처 배지 */
.source-row {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid #e8edf2;
}
.source-label {
    font-size: 0.73em;
    color: #8899aa;
    font-weight: 600;
    margin-right: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.source-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: linear-gradient(135deg, #f0f7ff, #e8f2fc);
    border: 1px solid #d0e3f4;
    color: #1e4a6e;
    padding: 4px 11px;
    border-radius: 20px;
    font-size: 0.76em;
    font-weight: 500;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
/* 예시 질문 */
.examples-row button {
    border-radius: 8px !important;
    font-size: 0.88em !important;
}
"""

_EXAMPLES = [
    "컴퓨터융합학부 졸업 요건이 어떻게 되나요?",
    "이번 학기 수강신청은 언제야?",
    "오늘 기숙사 점심 메뉴 뭐야?",
    "셔틀버스 시간표 알려줘",
    "최근 공지사항 보여줘",
]


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
                <div class="logo-icon">🎓</div>
                <h1>CNU Campus AI</h1>
                <p>충남대학교 학내 정보 Q&A 챗봇</p>
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
        chatbot = gr.Chatbot(**chatbot_kwargs, elem_classes=["chatbot"])

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
                    chat_history = [
                        {"role": h["role"], "content": h["content"]}
                        for h in history[:-2]
                        if isinstance(h, dict) and h.get("content")
                    ]
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
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])

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
