"""모델 다운로드 스크립트.

Gemma 4 12B (4bit 양자화)는 용량이 커서 저장소에 포함하지 않습니다.
이 스크립트를 실행하면 HuggingFace Hub에서 자동 다운로드됩니다.

사용법:
    python model/download_model.py

또는 chatbot.sh 실행 시 자동으로 다운로드됩니다.
"""

from huggingface_hub import snapshot_download

MODEL_NAME = "google/gemma-4-12b-it"

print(f"모델 다운로드 중: {MODEL_NAME}")
print("4bit 양자화 적용으로 실행 시 ~8GB VRAM 사용")
print()
print("이 모델은 transformers 라이브러리가 자동으로 캐시합니다.")
print("별도 다운로드 없이 chatbot.sh 실행 시 자동 다운로드됩니다.")
print()

snapshot_download(repo_id=MODEL_NAME)
print("다운로드 완료!")
