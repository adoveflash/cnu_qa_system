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
    max-width: 800px !important;
    margin: 0 auto !important;
    font-family: 'Pretendard', 'Apple SD Gothic Neo', sans-serif !important;
}
.title-area {
    text-align: center;
    padding: 20px 0 10px 0;
}
.title-area h1 {
    font-size: 1.8em;
    margin-bottom: 4px;
}
.title-area p {
    color: #666;
    font-size: 0.95em;
    margin-top: 0;
}
.category-badges {
    display: flex;
    justify-content: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 12px;
}
.category-badges span {
    background: #e8f4f8;
    color: #1a6b8a;
    padding: 4px 12px;
    border-radius: 16px;
    font-size: 0.85em;
    font-weight: 500;
}
footer {visibility: hidden}
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
                <h1>충남대학교 학내 정보 Q&A</h1>
                <p>궁금한 점을 자유롭게 질문하세요</p>
            </div>
            <div class="category-badges">
                <span>졸업요건</span>
                <span>공지사항</span>
                <span>학사일정</span>
                <span>식단 안내</span>
                <span>셔틀버스</span>
            </div>
            """
        )

        chatbot_kwargs = {"height": 480, "show_label": False, "container": False}
        if _NEED_TYPE_PARAM:
            chatbot_kwargs["type"] = "messages"
        chatbot = gr.Chatbot(**chatbot_kwargs)

        with gr.Row():
            msg = gr.Textbox(
                placeholder="질문을 입력하세요...",
                show_label=False,
                scale=9,
                container=False,
            )
            submit_btn = gr.Button("전송", variant="primary", scale=1, min_width=80)

        with gr.Row():
            gr.Examples(
                examples=_EXAMPLES,
                inputs=msg,
                label="이런 질문을 해보세요",
            )

        clear_btn = gr.Button("대화 초기화", variant="secondary", size="sm")

        def respond(message: str, history: list):
            if not message.strip():
                yield "", history
                return

            question = _extract_text(message)
            context, urls = retriever.build_context(question, top_k=5)

            if _USE_MESSAGES:
                history = history + [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": ""},
                ]
            else:
                history = history + [[question, ""]]

            if model is not None and tokenizer is not None:
                for partial in generate_answer_stream(question, context, urls, model, tokenizer):
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
