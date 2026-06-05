"""Gradio 챗봇 웹 UI 모듈.

충남대학교 학내 정보 Q&A 챗봇 인터페이스.
모델 유무에 관계없이 동작한다 (모델 없으면 RAG fallback).
스트리밍 지원: 모델이 있으면 토큰 단위로 실시간 출력.
Gradio 5.x / 6.x 호환.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import gradio as gr

from src.model.inference import fallback_answer, generate_answer_stream

_GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6

_CSS = """
.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
}
.header-container {
    text-align: center;
    padding: 24px 0 12px;
}
.header-title {
    font-size: 28px;
    font-weight: 700;
    color: #1a365d;
    margin-bottom: 4px;
}
.header-sub {
    font-size: 14px;
    color: #64748b;
}
.category-row {
    display: flex;
    justify-content: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 8px;
}
.category-btn {
    font-size: 13px !important;
    padding: 6px 14px !important;
    border-radius: 20px !important;
    border: 1px solid #cbd5e1 !important;
    background: #f8fafc !important;
    color: #334155 !important;
    cursor: pointer;
}
.category-btn:hover {
    background: #e2e8f0 !important;
    border-color: #94a3b8 !important;
}
.input-row {
    gap: 8px;
}
.send-btn {
    min-width: 80px !important;
    border-radius: 10px !important;
}
footer {
    display: none !important;
}
"""

_CATEGORIES = {
    "졸업요건": "컴퓨터융합학부 졸업 요건이 어떻게 되나요?",
    "공지사항": "최근 공지사항 알려줘",
    "학사일정": "이번 학기 수강신청은 언제 시작하나요?",
    "식단": "오늘 학식 뭐 나와요?",
    "셔틀버스": "셔틀버스 시간표 알려주세요",
}


def _make_chatbot_kwargs() -> dict:
    """Gradio 버전에 맞는 Chatbot kwargs를 반환한다."""
    kwargs: dict[str, Any] = {"height": 480, "show_label": False}
    if not _GRADIO_6:
        kwargs["type"] = "tuples"
        kwargs["placeholder"] = "질문을 입력하거나 위 카테고리를 눌러보세요."
    return kwargs


def _make_blocks_kwargs() -> dict:
    """Gradio 버전에 맞는 Blocks kwargs를 반환한다."""
    kwargs: dict[str, Any] = {"title": "CNU Q&A 챗봇"}
    if not _GRADIO_6:
        kwargs["theme"] = gr.themes.Soft(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.slate,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Noto Sans KR"),
        )
        kwargs["css"] = _CSS
    return kwargs


def _make_launch_kwargs(share: bool) -> dict:
    """Gradio 버전에 맞는 launch kwargs를 반환한다."""
    kwargs: dict[str, Any] = {"share": share}
    if _GRADIO_6:
        kwargs["css"] = _CSS
    return kwargs


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
    with gr.Blocks(**_make_blocks_kwargs()) as app:
        # 헤더
        gr.HTML(
            '<div class="header-container">'
            '<div class="header-title">충남대학교 학내 정보 Q&A</div>'
            '<div class="header-sub">'
            "Qwen3-8B + RAG 기반 캠퍼스 챗봇"
            "</div>"
            "</div>"
        )

        # 카테고리 바로가기 버튼
        with gr.Row(elem_classes="category-row"):
            cat_buttons = {}
            for label in _CATEGORIES:
                cat_buttons[label] = gr.Button(label, size="sm", elem_classes="category-btn")

        # 채팅 영역
        chatbot = gr.Chatbot(**_make_chatbot_kwargs())

        # 입력 영역
        with gr.Row(elem_classes="input-row"):
            msg = gr.Textbox(
                placeholder="충남대에 대해 무엇이든 물어보세요...",
                show_label=False,
                scale=6,
                container=False,
            )
            submit_btn = gr.Button("전송", variant="primary", scale=1, elem_classes="send-btn")

        with gr.Row():
            clear_btn = gr.Button("대화 초기화", size="sm", variant="secondary")

        def add_user_message(message: str, history: list) -> tuple[str, list]:
            """사용자 메시지를 히스토리에 추가하고 입력창을 비운다."""
            if not message.strip():
                return "", history
            history = history + [[message, None]]
            return "", history

        def bot_respond(history: list) -> Generator[list, None, None]:
            """마지막 사용자 메시지에 대해 스트리밍 응답을 생성한다."""
            if not history or history[-1][0] is None:
                yield history
                return

            question = history[-1][0]
            # Gradio 6에서 content가 dict로 올 수 있음
            if isinstance(question, dict):
                question = question.get("value", question.get("text", str(question)))

            context, urls = retriever.build_context(question, top_k=5)

            if model is not None and tokenizer is not None:
                for partial in generate_answer_stream(question, context, urls, model, tokenizer):
                    history[-1][1] = partial
                    yield history
            else:
                answer = fallback_answer(question, context, urls)
                history[-1][1] = answer
                yield history

        # 전송 / Enter 이벤트
        submit_btn.click(add_user_message, [msg, chatbot], [msg, chatbot]).then(
            bot_respond, chatbot, chatbot
        )
        msg.submit(add_user_message, [msg, chatbot], [msg, chatbot]).then(
            bot_respond, chatbot, chatbot
        )

        # 카테고리 버튼 이벤트
        for label, question in _CATEGORIES.items():
            cat_buttons[label].click(lambda q=question: q, outputs=[msg]).then(
                add_user_message, [msg, chatbot], [msg, chatbot]
            ).then(bot_respond, chatbot, chatbot)

        # 초기화
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
    app.launch(**_make_launch_kwargs(share))
