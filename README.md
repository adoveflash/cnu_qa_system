# 충남대학교 학내 정보 Q/A 시스템

자연어처리 텀프로젝트 — RAG + QLoRA 파인튜닝 기반 학사·장학금 안내 챗봇

## 개요

충남대학교 공개 웹페이지를 크롤링해 구축한 코퍼스 위에서, **Retrieval-Augmented Generation(RAG)** 과 **QLoRA 파인튜닝**을 결합하여 학사·장학금 관련 질문에 답변하는 시스템입니다.

## 기술 스택

| 구성 요소 | 선택 |
|-----------|------|
| Base 모델 | `Qwen/Qwen2.5-7B-Instruct` (4bit NF4) |
| 파인튜닝 | QLoRA (r=16, alpha=32, target: q/k/v/o_proj) |
| 임베딩 | `BAAI/bge-m3` |
| 벡터 DB | ChromaDB (로컬) |
| 웹 UI | Gradio (`share=True`) |

## 추론 파이프라인

```
질문
 └─► bge-m3 임베딩
      └─► ChromaDB top-5 검색
           └─► [시스템 프롬프트 + 컨텍스트 + 질문]
                └─► Qwen2.5-7B + LoRA
                     └─► 답변 + 출처 URL
```

컨텍스트에 없는 내용은 추측하지 않고 "확인되지 않은 정보입니다"로 응답합니다.

## 도메인 범위

- 학사 정보: 수강신청, 졸업 요건, 휴·복학, 전과·복수전공
- 장학금: 교내·외 종류, 신청 절차, 자격 요건

로그인이 필요한 페이지는 범위에서 제외합니다.

## 디렉터리 구조

```
.
├── submission.ipynb        # 제출용 노트북 (평가자가 "모두 실행"으로 동작)
├── notebooks/              # 단계별 개발·실험 노트북
├── src/
│   ├── crawl/              # 크롤러, HTML 정제
│   ├── data/               # 청킹, Q&A 자동 생성
│   ├── rag/                # 임베딩, 벡터 DB, 검색
│   ├── model/              # base 로드, LoRA 학습, 추론
│   ├── eval/               # LLM judge, latency 측정
│   └── app/                # Gradio UI
├── data/
│   ├── corpus/             # 크롤링 원본 (.jsonl)
│   ├── qa/train.jsonl
│   └── qa/eval.jsonl       # 100개 평가셋 (학습 데이터에서 제외)
├── models/lora_adapter/    # LoRA 어댑터만 저장 (< 200 MB)
└── docs/decisions/         # 기술 의사결정 기록
```

## 실행 방법

### 환경 설치

```bash
pip install -r requirements.txt
```

### 코드 품질 검사

```bash
ruff check src/
ruff format --line-length 100 src/
```

### 스모크 테스트 (Colab T4)

`notebooks/00_model_smoke_test.ipynb` — 모델 로드 + VRAM 측정

### 제출 노트북 실행

`submission.ipynb`를 Colab T4 런타임에서 처음부터 끝까지 실행합니다.
사용자 입력 없이 자동으로 완료되어야 합니다.

## 주의 사항

- `data/qa/eval.jsonl` 100개는 학습 데이터에 절대 포함하지 않습니다.
- KorQuAD 등 기존 공개 Q&A 데이터셋은 사용하지 않습니다.
- LoRA 어댑터만 커밋합니다. base 모델 가중치는 커밋하지 않습니다.
- API 키 등 시크릿은 `.env` 또는 Colab `userdata`에서만 읽습니다.
