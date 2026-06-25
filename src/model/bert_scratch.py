"""밑바닥부터 직접 구현한 BERT (Transformer Encoder).

구현 내용:
  1. BERT 아키텍처(임베딩 + 멀티헤드 셀프어텐션 인코더 + 풀러)를 직접 구현
  2. HuggingFace 사전학습 가중치를 직접 구현한 모델로 copy
  3. HF 모델과 임베딩(last_hidden_state) 값이 일치하는지 검증

설계 메모:
- 서브모듈 이름을 HF `transformers.BertModel`과 동일하게 지어,
  `model.load_state_dict(hf_model.state_dict())` 한 줄로 param copy가 되게 한다.
  → 가중치는 그대로 옮겨지므로, 검증은 "직접 짠 forward 로직이 맞는가"를 테스트한다.
- 활성화는 exact GELU(erf 기반) — HF BERT 기본값과 동일.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BertConfig:
    """BERT 하이퍼파라미터. HF BertConfig의 필요한 필드만 추린 것."""

    vocab_size: int = 32000
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    pad_token_id: int = 0


class BertEmbeddings(nn.Module):
    """토큰 + 위치 + 세그먼트 임베딩을 합치고 LayerNorm한다."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, input_ids: torch.Tensor, token_type_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        embeddings = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        embeddings = self.LayerNorm(embeddings)
        return self.dropout(embeddings)


class BertSelfAttention(nn.Module):
    """멀티헤드 스케일드 닷-프로덕트 셀프어텐션."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        """(B, S, H) → (B, num_heads, S, head_size)."""
        b, s, _ = x.size()
        x = x.view(b, s, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        q = self._shape(self.query(hidden_states))
        k = self._shape(self.key(hidden_states))
        v = self._shape(self.value(hidden_states))

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        scores = scores + attention_mask  # 확장 마스크: pad 위치에 -10000
        probs = self.dropout(F.softmax(scores, dim=-1))

        context = torch.matmul(probs, v)  # (B, heads, S, head_size)
        context = context.permute(0, 2, 1, 3).contiguous()
        b, s, _, _ = context.size()
        return context.view(b, s, self.all_head_size)


class BertSelfOutput(nn.Module):
    """어텐션 출력 dense + residual + LayerNorm."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(self.dense(hidden_states))
        return self.LayerNorm(hidden_states + input_tensor)


class BertAttention(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        self_out = self.self(hidden_states, attention_mask)
        return self.output(self_out, hidden_states)


class BertIntermediate(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.dense(hidden_states))  # exact gelu (HF 기본)


class BertOutput(nn.Module):
    """FFN 출력 dense + residual + LayerNorm."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(self.dense(hidden_states))
        return self.LayerNorm(hidden_states + input_tensor)


class BertLayer(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attn = self.attention(hidden_states, attention_mask)
        inter = self.intermediate(attn)
        return self.output(inter, attn)


class BertEncoder(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layer:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class BertPooler(nn.Module):
    """[CLS] 토큰 표현을 dense + tanh로 풀링."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        cls_token = hidden_states[:, 0]
        return torch.tanh(self.dense(cls_token))


class BertModel(nn.Module):
    """직접 구현한 BERT 인코더. HF BertModel과 동일한 서브모듈 이름 구조."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns: (last_hidden_state, pooled_output)."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        # (B, S) → (B, 1, 1, S) 확장 마스크. pad(0) 위치를 -10000으로.
        ext_mask = attention_mask[:, None, None, :].to(dtype=torch.float32)
        ext_mask = (1.0 - ext_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        sequence_output = self.encoder(embedding_output, ext_mask)
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output


class BertForQuestionClassification(nn.Module):
    """직접 구현한 BERT + 분류 헤드 (Task 1: 질문 5분류).

    HF BertForSequenceClassification과 같은 이름(bert/classifier)을 써,
    백본 가중치는 copy하고 분류 헤드만 새로 학습한다.
    """

    def __init__(self, config: BertConfig, num_labels: int = 5) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict:
        _, pooled = self.bert(input_ids, attention_mask, token_type_ids)
        logits = self.classifier(self.dropout(pooled))
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(logits, labels)
        return out


# ── HF 가중치 copy & 검증 ────────────────────────────────────────────────────


def config_from_hf(hf_name: str) -> BertConfig:
    """HF 체크포인트의 config를 읽어 직접 구현한 BertConfig로 옮긴다."""
    from transformers import AutoConfig

    c = AutoConfig.from_pretrained(hf_name)
    return BertConfig(
        vocab_size=c.vocab_size,
        hidden_size=c.hidden_size,
        num_hidden_layers=c.num_hidden_layers,
        num_attention_heads=c.num_attention_heads,
        intermediate_size=c.intermediate_size,
        max_position_embeddings=c.max_position_embeddings,
        type_vocab_size=c.type_vocab_size,
        layer_norm_eps=getattr(c, "layer_norm_eps", 1e-12),
        hidden_dropout_prob=c.hidden_dropout_prob,
        attention_probs_dropout_prob=c.attention_probs_dropout_prob,
        pad_token_id=getattr(c, "pad_token_id", 0),
    )


def load_hf_backbone(model: BertModel, hf_name: str) -> None:
    """HF BertModel 가중치를 직접 구현한 BertModel로 copy한다.

    서브모듈 이름이 동일하므로 state_dict를 그대로 적재한다.
    (position_ids 등 HF의 비학습 버퍼는 제외하고 strict=False로 흡수)
    """
    from transformers import AutoModel

    hf_model = AutoModel.from_pretrained(hf_name)
    hf_state = hf_model.state_dict()
    # HF가 등록하는 비파라미터 버퍼 제거
    hf_state = {
        k: v
        for k, v in hf_state.items()
        if not k.endswith("position_ids") and "embeddings.token_type_ids" not in k
    }
    missing, unexpected = model.load_state_dict(hf_state, strict=False)
    # pooler를 안 쓰는 일부 체크포인트 대비 — pooler 외 누락은 에러로 본다
    real_missing = [k for k in missing if not k.startswith("pooler.")]
    if real_missing:
        raise RuntimeError(f"가중치 copy 누락: {real_missing[:5]} ...")
    if unexpected:
        print(f"[bert_scratch] 무시한 HF 키: {unexpected[:3]} ...")


@torch.no_grad()
def verify_against_hf(hf_name: str, text: str = "충남대학교 졸업요건 알려줘") -> float:
    """직접 구현한 BERT와 HF BERT의 last_hidden_state 최대 절대오차를 반환한다.

    1e-4 미만이면 forward 구현이 정확하다고 본다.

    Args:
        hf_name: HF 체크포인트 이름
        text: 검증용 입력 문장

    Returns:
        두 모델 출력의 최대 절대오차(float)
    """
    from transformers import AutoModel, AutoTokenizer

    config = config_from_hf(hf_name)
    mine = BertModel(config).eval()
    load_hf_backbone(mine, hf_name)

    hf_model = AutoModel.from_pretrained(hf_name).eval()
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    enc = tokenizer(text, return_tensors="pt")

    my_seq, _ = mine(enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids"))
    hf_out = hf_model(**enc).last_hidden_state

    max_diff = (my_seq - hf_out).abs().max().item()
    print(f"[verify] 최대 절대오차 = {max_diff:.2e}  ({'OK ✅' if max_diff < 1e-4 else 'FAIL ❌'})")
    return max_diff


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "klue/bert-base"
    verify_against_hf(name)
