"""Q&A 쌍 자동 생성 모듈.

Ollama 로컬 LLM을 사용하여 청크에서 Q&A 쌍을 생성한다.
이미 처리한 chunk_id는 건너뛰어 증분 생성을 지원한다.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import requests

_DEFAULT_OLLAMA_URL = "http://localhost:11434"
_DEFAULT_MODEL = "qwen2.5:7b"
_PROMPT_PATH = Path("src/data/prompts/qa_gen.txt")


def load_prompt_template(prompt_path: Path = _PROMPT_PATH) -> str:
    """프롬프트 템플릿 파일을 읽는다.

    Args:
        prompt_path: 프롬프트 파일 경로

    Returns:
        프롬프트 템플릿 문자열 ({context}, {title}, {url} 플레이스홀더 포함)
    """
    return prompt_path.read_text(encoding="utf-8")


def _load_done_chunk_ids(output_path: Path) -> set[str]:
    """이미 생성된 Q&A의 chunk_id 목록을 로드한다.

    Args:
        output_path: 기존 Q&A JSONL 파일 경로

    Returns:
        처리 완료된 chunk_id 집합
    """
    done: set[str] = set()
    if not output_path.exists():
        return done
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                done.add(record["chunk_id"])
    return done


def _call_ollama(
    prompt: str,
    model: str = _DEFAULT_MODEL,
    base_url: str = _DEFAULT_OLLAMA_URL,
    temperature: float = 0.7,
    seed: int = 42,
) -> str:
    """Ollama API를 호출하여 텍스트 응답을 반환한다.

    Args:
        prompt: 프롬프트 전문
        model: Ollama 모델명
        base_url: Ollama 서버 URL
        temperature: 생성 온도
        seed: 재현성 시드

    Returns:
        모델 응답 텍스트

    Raises:
        requests.RequestException: API 호출 실패 시
    """
    resp = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "seed": seed,
            },
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def _parse_qa_response(response_text: str) -> list[dict[str, str]]:
    """LLM 응답에서 JSON Q&A 배열을 파싱한다.

    Args:
        response_text: LLM 응답 전문

    Returns:
        [{"question": str, "answer": str}, ...] 리스트. 파싱 실패 시 빈 리스트.
    """
    # 첫 번째 '[' ~ 매칭되는 ']' 까지 추출 (중국어 전환 등으로 뒤가 깨져도 대응)
    start = response_text.find("[")
    if start == -1:
        return []

    # 브래킷 매칭으로 올바른 JSON 배열 끝 찾기
    depth = 0
    end = -1
    for i in range(start, len(response_text)):
        if response_text[i] == "[":
            depth += 1
        elif response_text[i] == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end == -1:
        # 닫는 괄호 없으면 마지막 '}' 뒤에 ']' 추가 시도
        last_brace = response_text.rfind("}")
        if last_brace > start:
            candidate = response_text[start : last_brace + 1] + "]"
        else:
            return []
    else:
        candidate = response_text[start:end]

    try:
        pairs = json.loads(candidate)
    except json.JSONDecodeError:
        # 개별 객체라도 추출 시도
        valid = []
        for m in re.finditer(
            r'\{"question"\s*:\s*"[^"]+"\s*,\s*"answer"\s*:\s*"[^"]+"\}', candidate
        ):
            try:
                obj = json.loads(m.group())
                valid.append(obj)
            except json.JSONDecodeError:
                continue
        pairs = valid if valid else []

    if not isinstance(pairs, list):
        return []

    # 유효성 검증: 한국어 응답만 허용
    valid = []
    for pair in pairs:
        if isinstance(pair, dict) and "question" in pair and "answer" in pair:
            q = str(pair["question"]).strip()
            a = str(pair["answer"]).strip()
            # 중국어 문자 비율이 높으면 제외
            cjk_ratio = sum(1 for c in q + a if "\u4e00" <= c <= "\u9fff") / max(len(q + a), 1)
            if len(q) > 5 and len(a) > 10 and cjk_ratio < 0.1:
                valid.append({"question": q, "answer": a})
    return valid


def generate_qa_for_chunk(
    chunk: dict[str, Any],
    prompt_template: str,
    model: str = _DEFAULT_MODEL,
    base_url: str = _DEFAULT_OLLAMA_URL,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """단일 청크에 대해 Q&A 쌍을 생성한다.

    Args:
        chunk: 청크 레코드 (text, chunk_id, url, title, source 키)
        prompt_template: 프롬프트 템플릿
        model: Ollama 모델명
        base_url: Ollama 서버 URL
        seed: 재현성 시드

    Returns:
        Q&A 레코드 리스트 (question, answer, chunk_id, url, source 키 포함)
    """
    prompt = prompt_template.format(
        context=chunk["text"],
        title=chunk["title"],
        url=chunk["url"],
    )

    try:
        response = _call_ollama(prompt, model=model, base_url=base_url, seed=seed)
    except requests.RequestException as e:
        print(f"  [오류] {chunk['chunk_id']}: API 호출 실패 — {e}")
        return []

    pairs = _parse_qa_response(response)
    if not pairs:
        print(f"  [경고] {chunk['chunk_id']}: 파싱 실패, 건너뜀")
        return []

    return [
        {
            "question": p["question"],
            "answer": p["answer"],
            "chunk_id": chunk["chunk_id"],
            "url": chunk["url"],
            "source": chunk["source"],
        }
        for p in pairs
    ]


def generate_all(
    chunks_path: Path,
    output_path: Path,
    prompt_path: Path = _PROMPT_PATH,
    model: str = _DEFAULT_MODEL,
    base_url: str = _DEFAULT_OLLAMA_URL,
    seed: int = 42,
    max_chunks: int | None = None,
) -> int:
    """전체 청크에 대해 Q&A를 증분 생성하여 JSONL로 저장한다.

    이미 output_path에 존재하는 chunk_id는 건너뛴다.

    Args:
        chunks_path: data/corpus/chunks.jsonl
        output_path: data/qa/train.jsonl
        prompt_path: 프롬프트 템플릿 경로
        model: Ollama 모델명
        base_url: Ollama 서버 URL
        seed: 재현성 시드
        max_chunks: 처리할 최대 청크 수 (None이면 전체)

    Returns:
        새로 생성된 Q&A 쌍 수
    """
    prompt_template = load_prompt_template(prompt_path)

    chunks: list[dict[str, Any]] = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    # 이미 처리한 chunk_id 로드 (증분 처리)
    done_ids = _load_done_chunk_ids(output_path)
    remaining = [c for c in chunks if c["chunk_id"] not in done_ids]

    if max_chunks is not None:
        remaining = remaining[:max_chunks]

    total = len(remaining)
    print(f"  전체 청크: {len(chunks)}개, 처리 완료: {len(done_ids)}개, 남은 처리 대상: {total}개")

    if total == 0:
        print("  처리할 청크가 없습니다.")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_qa_count = 0

    with open(output_path, "a", encoding="utf-8") as out:
        for i, chunk in enumerate(remaining, 1):
            print(f"  [{i}/{total}] {chunk['chunk_id']} ...", end=" ", flush=True)
            qa_pairs = generate_qa_for_chunk(
                chunk, prompt_template, model=model, base_url=base_url, seed=seed
            )
            for qa in qa_pairs:
                out.write(json.dumps(qa, ensure_ascii=False) + "\n")
            out.flush()
            new_qa_count += len(qa_pairs)
            print(f"{len(qa_pairs)}쌍 생성")

    return new_qa_count


def split_eval_set(
    train_path: Path,
    eval_path: Path,
    eval_count: int = 100,
    seed: int = 42,
) -> None:
    """train.jsonl에서 eval_count개를 랜덤 샘플링하여 eval.jsonl로 분리한다.

    분리된 항목은 train.jsonl에서 제거한다.

    Args:
        train_path: data/qa/train.jsonl
        eval_path: data/qa/eval.jsonl
        eval_count: 평가용 샘플 수
        seed: 랜덤 시드
    """
    with open(train_path, encoding="utf-8") as f:
        all_qa = [json.loads(line) for line in f if line.strip()]

    if len(all_qa) < eval_count:
        print(f"  [경고] Q&A가 {len(all_qa)}개뿐이라 eval {eval_count}개 분리 불가. 건너뜀.")
        return

    random.seed(seed)
    eval_indices = set(random.sample(range(len(all_qa)), eval_count))

    eval_set = [all_qa[i] for i in sorted(eval_indices)]
    train_set = [all_qa[i] for i in range(len(all_qa)) if i not in eval_indices]

    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, "w", encoding="utf-8") as f:
        for qa in eval_set:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    with open(train_path, "w", encoding="utf-8") as f:
        for qa in train_set:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    print(f"  eval: {len(eval_set)}개 → {eval_path}")
    print(f"  train: {len(train_set)}개 → {train_path}")
    print("  ⚠ eval.jsonl의 100개 항목을 반드시 수동 검수하세요!")
