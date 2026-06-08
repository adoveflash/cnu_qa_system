# 모델 안내

## Base 모델
- **모델명**: `google/gemma-4-12b-it`
- **양자화**: 4bit NF4 (BitsAndBytes)
- **파인튜닝**: 없음 (base 모델 직접 사용)
- **VRAM**: ~8-9GB (4bit 양자화 시)

## 다운로드
모델은 용량(~24GB)이 커서 저장소에 포함하지 않습니다.
`chatbot.sh` 실행 시 HuggingFace Hub에서 자동 다운로드됩니다.

수동 다운로드:
```bash
python model/download_model.py
```

## 벡터 DB
RAG 검색용 벡터 인덱스는 `data/vector_db/`에 저장되어 있습니다.
HuggingFace Hub에서 다운로드: `adoveflash/cnu-qa-system`

## 임베딩 모델
- **모델명**: `BAAI/bge-m3`
- 실행 시 자동 다운로드
