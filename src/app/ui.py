"""Gradio 챗봇 웹 UI 모듈.

충남대학교 학내 정보 Q&A 챗봇 인터페이스.
모델 유무에 관계없이 동작한다 (모델 없으면 RAG fallback).
"""

from __future__ import annotations

from typing import Any

import gradio as gr

from src.model.inference import fallback_answer, generate_answer_stream


def create_app(
    retriever: Any,
    model: Any | None = None,
    tokenizer: Any | None = None,
) -> gr.Blocks:
    """Gradio 챗봇 UI를 생성한다."""

    with gr.Blocks() as app:
        gr.Markdown("# 충남대학교 학내 정보 Q&A\n졸업요건 | 공지사항 | 학사일정 | 식단 | 셔틀버스")

        chatbot = gr.Chatbot(height=500)
        msg = gr.Textbox(placeholder="질문을 입력하세요...", show_label=False)

        with gr.Row():
            submit_btn = gr.Button("전송", variant="primary")
            clear_btn = gr.Button("대화 초기화")

        def respond(message: str, history: list) -> tuple[str, list]:
            """동기 응답 — 질문에 대해 답변을 생성한다."""
            if not message.strip():
                return "", history

            history = history + [[message, None]]
            question = message

            # dict로 올 수 있는 경우 방어
            if isinstance(question, dict):
                question = question.get("value", question.get("text", str(question)))

            context, urls = retriever.build_context(question, top_k=5)

            if model is not None and tokenizer is not None:
                answer = ""
                for partial in generate_answer_stream(question, context, urls, model, tokenizer):
                    answer = partial
                history[-1][1] = answer
            else:
                history[-1][1] = fallback_answer(question, context, urls)

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
