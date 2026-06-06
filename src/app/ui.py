"""Gradio 챗봇 웹 UI 모듈.

충남대학교 학내 정보 Q&A 챗봇 인터페이스.
Gradio 4.x/5.x (tuples) 및 6.x (messages) 자동 호환.
"""

from __future__ import annotations

from typing import Any

import gradio as gr

from src.model.inference import fallback_answer, generate_answer_stream

# Gradio 5+ 는 messages 형식, 4.x는 tuples 형식
_USE_MESSAGES = int(gr.__version__.split(".")[0]) >= 5


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

    with gr.Blocks() as app:
        gr.Markdown("# 충남대학교 학내 정보 Q&A\n졸업요건 | 공지사항 | 학사일정 | 식단 | 셔틀버스")

        chatbot_kwargs = {"height": 500}
        if _USE_MESSAGES:
            chatbot_kwargs["type"] = "messages"
        chatbot = gr.Chatbot(**chatbot_kwargs)
        msg = gr.Textbox(placeholder="질문을 입력하세요...", show_label=False)

        with gr.Row():
            submit_btn = gr.Button("전송", variant="primary")
            clear_btn = gr.Button("대화 초기화")

        def respond(message: str, history: list) -> tuple[str, list]:
            if not message.strip():
                return "", history

            question = _extract_text(message)
            context, urls = retriever.build_context(question, top_k=5)

            if model is not None and tokenizer is not None:
                answer = ""
                for partial in generate_answer_stream(question, context, urls, model, tokenizer):
                    answer = partial
            else:
                answer = fallback_answer(question, context, urls)

            if _USE_MESSAGES:
                history = history + [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            else:
                history = history + [[question, answer]]
            return "", history

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
