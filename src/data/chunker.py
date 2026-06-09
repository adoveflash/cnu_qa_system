"""텍스트 청킹 모듈.

클리닝+마스킹된 문서를 300-500 토큰 청크로 분할한다.
문단·제목 경계를 우선 존중하고, 50 토큰 오버랩을 적용한다.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerFast

_DEFAULT_MODEL = "Qwen/Qwen3-8B"


def load_tokenizer(model_name: str = _DEFAULT_MODEL) -> PreTrainedTokenizerFast:
    """Qwen2.5 토크나이저를 로드한다. 토큰 수 계산에 사용.

    Args:
        model_name: HuggingFace 모델 이름

    Returns:
        토크나이저 객체
    """
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def _count_tokens(text: str, tokenizer: PreTrainedTokenizerFast) -> int:
    """텍스트의 토큰 수를 반환한다."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def _split_into_paragraphs(text: str) -> list[str]:
    """텍스트를 문단 단위로 분리한다. 빈 줄(2개 이상 연속 개행) 기준.

    Args:
        text: 입력 텍스트

    Returns:
        비어있지 않은 문단 리스트
    """
    paragraphs = re.split(r"\n{2,}", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_into_sentences(text: str) -> list[str]:
    """문단을 문장 단위로 분리한다.

    Args:
        text: 문단 텍스트

    Returns:
        문장 리스트
    """
    sentences = re.split(r"(?<=[.!?。])\s+|\n", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_document(
    text: str,
    tokenizer: PreTrainedTokenizerFast,
    max_tokens: int = 500,
    min_tokens: int = 300,
    overlap_tokens: int = 50,
) -> list[str]:
    """문서를 토큰 수 기반으로 청킹한다.

    알고리즘:
    1. 텍스트를 문단으로 분리
    2. 문단을 순차 누적하되, 누적 토큰이 max_tokens를 초과하면 청크 확정
    3. 단일 문단이 max_tokens를 초과하면 문장 단위로 재분할
    4. overlap은 이전 청크의 마지막 ~50 토큰을 다음 청크 앞에 붙임
    5. 마지막 청크가 min_tokens 미만이면 이전 청크에 병합

    Args:
        text: 문서 전문
        tokenizer: 토큰 수 계산용 토크나이저
        max_tokens: 청크 최대 토큰 수
        min_tokens: 청크 최소 토큰 수
        overlap_tokens: 청크 간 겹침 토큰 수

    Returns:
        청크 텍스트 리스트
    """
    paragraphs = _split_into_paragraphs(text)
    if not paragraphs:
        return []

    # 문단을 units로 펼치되, 긴 문단은 문장으로 재분할
    units: list[str] = []
    for para in paragraphs:
        if _count_tokens(para, tokenizer) > max_tokens:
            sentences = _split_into_sentences(para)
            units.extend(sentences)
        else:
            units.append(para)

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0

    for unit in units:
        unit_tokens = _count_tokens(unit, tokenizer)

        if buffer and buffer_tokens + unit_tokens > max_tokens:
            # 현재 버퍼를 청크로 확정
            chunks.append("\n\n".join(buffer))

            # 오버랩: 이전 청크에서 마지막 ~overlap_tokens만큼 가져옴
            overlap_text = _get_overlap_text(chunks[-1], tokenizer, overlap_tokens)
            if overlap_text:
                buffer = [overlap_text]
                buffer_tokens = _count_tokens(overlap_text, tokenizer)
            else:
                buffer = []
                buffer_tokens = 0

        buffer.append(unit)
        buffer_tokens = _count_tokens("\n\n".join(buffer), tokenizer)

    # 남은 버퍼 처리
    if buffer:
        last_chunk = "\n\n".join(buffer)
        if chunks and _count_tokens(last_chunk, tokenizer) < min_tokens:
            # 마지막 청크가 너무 짧으면 이전 청크에 병합
            chunks[-1] = chunks[-1] + "\n\n" + last_chunk
        else:
            chunks.append(last_chunk)

    return chunks


def _get_overlap_text(
    chunk_text: str,
    tokenizer: PreTrainedTokenizerFast,
    overlap_tokens: int,
) -> str:
    """청크의 마지막 ~overlap_tokens에 해당하는 텍스트를 반환한다.

    Args:
        chunk_text: 이전 청크 텍스트
        tokenizer: 토크나이저
        overlap_tokens: 오버랩 토큰 수

    Returns:
        오버랩 텍스트 (비어있을 수 있음)
    """
    token_ids = tokenizer.encode(chunk_text, add_special_tokens=False)
    if len(token_ids) <= overlap_tokens:
        return ""
    overlap_ids = token_ids[-overlap_tokens:]
    return tokenizer.decode(overlap_ids, skip_special_tokens=True)


def chunk_all(
    input_path: Path,
    output_path: Path,
    model_name: str = _DEFAULT_MODEL,
) -> int:
    """cleaned+masked JSONL을 읽어 청킹 후 chunks.jsonl로 저장한다.

    Args:
        input_path: data/corpus/cleaned.jsonl (PII 마스킹 완료)
        output_path: data/corpus/chunks.jsonl
        model_name: 토크나이저 모델명

    Returns:
        총 청크 수
    """
    print(f"  토크나이저 로드 중: {model_name}")
    tokenizer = load_tokenizer(model_name)

    docs: list[dict[str, Any]] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_chunks = 0
    dup_skipped = 0
    seen_hashes: set[str] = set()

    with open(output_path, "w", encoding="utf-8") as out:
        for doc_idx, doc in enumerate(docs):
            text_chunks = chunk_document(doc["content"], tokenizer)
            for chunk_idx, chunk_text in enumerate(text_chunks):
                # 청크 단위 중복 제거: 공백 정규화 후 해시 (여러 페이지에 반복되는
                # 동일 보일러플레이트 청크 제거). 오버랩 청크는 완전동일이 아니라 보존됨.
                norm = re.sub(r"\s+", " ", chunk_text).strip()
                h = hashlib.md5(norm.encode()).hexdigest()
                if h in seen_hashes:
                    dup_skipped += 1
                    continue
                seen_hashes.add(h)

                chunk_record = {
                    "chunk_id": f"{doc['source']}_{doc_idx}_{chunk_idx}",
                    "text": chunk_text,
                    "url": doc["url"],
                    "title": doc["title"],
                    "source": doc["source"],
                    "token_count": _count_tokens(chunk_text, tokenizer),
                }
                out.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")
                total_chunks += 1

    if dup_skipped:
        print(f"  중복 청크 제거: {dup_skipped}개")
    return total_chunks
