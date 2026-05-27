"""평가 모듈.

eval.jsonl 100개에 대해 일괄 추론 후 LLM judge 점수와 latency를 측정한다.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests


def run_batch_inference(
    eval_path: Path,
    retriever: Any,
    model: Any,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """eval 데이터에 대해 일괄 추론을 수행한다.

    Args:
        eval_path: data/qa/eval.jsonl 경로
        retriever: Retriever 인스턴스
        model: LLM 모델
        tokenizer: 토크나이저

    Returns:
        [{question, reference, prediction, urls, latency_ms}, ...]
    """
    from src.model.inference import generate_answer

    with open(eval_path, encoding="utf-8") as f:
        eval_data = [json.loads(line) for line in f if line.strip()]

    results = []
    for i, qa in enumerate(eval_data, 1):
        print(f"  [{i}/{len(eval_data)}] {qa['question'][:40]}...", end=" ", flush=True)

        start = time.time()
        context, urls = retriever.build_context(qa["question"])
        answer = generate_answer(qa["question"], context, urls, model, tokenizer)
        elapsed_ms = (time.time() - start) * 1000

        results.append(
            {
                "question": qa["question"],
                "reference": qa["answer"],
                "prediction": answer,
                "urls": urls,
                "latency_ms": round(elapsed_ms, 1),
            }
        )
        print(f"{elapsed_ms:.0f}ms")

    return results


def llm_judge_score(
    question: str,
    reference: str,
    prediction: str,
    ollama_model: str = "qwen2.5:7b",
    base_url: str = "http://localhost:11434",
) -> dict[str, Any]:
    """LLM judge로 답변 품질을 1~5점으로 평가한다.

    Args:
        question: 질문
        reference: 정답
        prediction: 모델 예측
        ollama_model: 평가용 Ollama 모델명
        base_url: Ollama 서버 URL

    Returns:
        {"score": int, "reason": str}
    """
    prompt = f"""다음 질문에 대한 모델 답변을 평가하세요.

질문: {question}
정답: {reference}
모델 답변: {prediction}

평가 기준:
- 5점: 정답과 동일하거나 더 상세한 정확한 답변
- 4점: 핵심 내용은 맞지만 일부 세부사항 누락
- 3점: 부분적으로 맞지만 중요한 내용 누락 또는 부정확
- 2점: 대부분 부정확하거나 관련 없는 답변
- 1점: 완전히 틀리거나 답변 거부

반드시 아래 JSON 형식으로만 응답하세요:
{{"score": 점수, "reason": "이유"}}"""

    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "seed": 42},
            },
            timeout=60,
        )
        resp.raise_for_status()
        text = resp.json()["response"]

        import re

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            result = json.loads(match.group())
            return {"score": int(result.get("score", 0)), "reason": str(result.get("reason", ""))}
    except Exception:
        pass

    return {"score": 0, "reason": "평가 실패"}


def evaluate_all(
    results: list[dict[str, Any]],
    output_path: Path | None = None,
    ollama_model: str = "qwen2.5:7b",
) -> dict[str, Any]:
    """전체 결과에 대해 LLM judge 평가 및 통계를 산출한다.

    Args:
        results: run_batch_inference 결과
        output_path: 결과 저장 경로 (None이면 저장 안 함)
        ollama_model: 평가용 모델명

    Returns:
        {"avg_score": float, "avg_latency_ms": float, "results": list}
    """
    print(f"\nLLM Judge 평가 시작 (모델: {ollama_model})")
    for i, r in enumerate(results, 1):
        print(f"  [{i}/{len(results)}] 평가 중...", end=" ", flush=True)
        judge = llm_judge_score(
            r["question"], r["reference"], r["prediction"], ollama_model=ollama_model
        )
        r["judge_score"] = judge["score"]
        r["judge_reason"] = judge["reason"]
        print(f"점수: {judge['score']}")

    scores = [r["judge_score"] for r in results if r["judge_score"] > 0]
    latencies = [r["latency_ms"] for r in results]

    summary = {
        "avg_score": round(sum(scores) / len(scores), 2) if scores else 0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
        "total": len(results),
        "scored": len(scores),
        "results": results,
    }

    print(f"\n평균 점수: {summary['avg_score']}/5.0 ({summary['scored']}/{summary['total']}건)")
    print(f"평균 지연: {summary['avg_latency_ms']}ms")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"결과 저장: {output_path}")

    return summary
