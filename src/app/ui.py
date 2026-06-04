"""Gradio 챗봇 웹 UI 모듈.

충남대학교 학내 정보 Q&A 챗봇 인터페이스.
모델 유무에 관계없이 동작한다 (모델 없으면 RAG fallback).
"""

from __future__ import annotations

from typing import Any

import gradio as gr

from src.model.inference import generate_answer, fallback_answer

EXAMPLES = [
    "컴퓨터융합학부 졸업 요건이 어떻게 되나요?",
    "이번 학기 수강신청은 언제 시작하나요?",
    "오늘 학식 뭐 나와요?",
    "셔틀버스 시간표 알려주세요",
    "최근 공지사항 알려줘",
]


def create_app(
    retriever: Any,
    model: Any | None = None,
    tokenizer: Any | None = None,
) -> gr.Blocks:
    """Gradio 챗봇 UI를 생성한다.

    Args:
        retriever: Retriever 인스턴스
        model: LLM 모델 (None이면 RAG fallback)
        tokenizer: 토크나이저

    Returns:
        Gradio Blocks 앱
    """

    def respond(message: str, history: list[dict[str, str]]) -> str:
        if not message.strip():
            return "질문을 입력해주세요."

        context, urls = retriever.build_context(message, top_k=5)

        if model is not None and tokenizer is not None:
            return generate_answer(message, context, urls, model, tokenizer)
        return fallback_answer(message, context, urls)

    with gr.Blocks(title="CNU Q&A 챗봇") as app:
        gr.Markdown(
            "# 충남대학교 학내 정보 Q&A\n"
            "졸업요건 | 공지사항 | 학사일정 | 식단 | 셔틀버스"
        )

        chatbot = gr.Chatbot(height=520)
        msg = gr.Textbox(placeholder="질문을 입력하세요...", show_label=False)

        with gr.Row():
            submit_btn = gr.Button("전송", variant="primary")
            clear_btn = gr.Button("대화 초기화")

        gr.Examples(examples=EXAMPLES, inputs=msg)

        def user_submit(message: str, history: list) -> tuple:
            """사용자 메시지 제출 처리."""
            if not message.strip():
                return "", history
            history = history + [{"role": "user", "content": message}]
            answer = respond(message, history)
            history = history + [{"role": "assistant", "content": answer}]
            return "", history

        submit_btn.click(user_submit, [msg, chatbot], [msg, chatbot])
        msg.submit(user_submit, [msg, chatbot], [msg, chatbot])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])

    return app


def launch(
    retriever: Any,
    model: Any | None = None,
    tokenizer: Any | None = None,
    share: bool = True,
) -> None:
    """Gradio 챗봇 UI를 실행한다.

    Args:
        retriever: Retriever 인스턴스
        model: LLM 모델 (None이면 RAG fallback)
        tokenizer: 토크나이저
        share: 공유 링크 생성 여부
    """
    app = create_app(retriever, model, tokenizer)
    print("[ui] Gradio UI 시작...")
    app.launch(share=share)
