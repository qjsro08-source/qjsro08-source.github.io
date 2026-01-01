---
title:  "[Review] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
excerpt: "BERT - Review."
categories:
  - Multi-Modal Learning
tags:
  - Deep Learning
  - LLM
  - Self-supervised learning
toc: true
toc_sticky: true
author_profile: true
mathjax: true
---

## 1. Introduction
**BERT(Bidirectional Encoder Representations from Transformers)**는 2018년 구글에서 발표한 모델로, Transformer의 **Encoder(인코더)** 구조만을 사용하여 양방향 문맥을 학습하는 언어 모델입니다.

이 모델은 라벨링 되지 않은 대용량 텍스트 데이터로 사전 학습(Pre-training)을 수행한 뒤, 특정 Task에 맞춰 미세 조정(Fine-tuning)을 하는 방식으로 NLP 분야의 성능을 비약적으로 향상시켰습니다.

<div style="text-align: center;">
  <img src="/assets/images/BERT/BERT.png" alt="BERT Overall Architecture" width="80%">
  <br>
  <em>Figure 1. Overall architecture of the proposed BERT.</em>
</div>
<br>

이 글에서는 BERT의 입력 구조(Input Representation)와 핵심적인 두 가지 사전 학습 Task인 **MLM**과 **NSP**에 대해 정리합니다.

---

## 2. BERT Input Representation
BERT는 모델에 데이터를 넣을 때 3가지 임베딩을 합(Sum)하여 입력으로 사용합니다.

$$ Input = Token\ Emb + Segment\ Emb + Position\ Emb $$

아래 그림은 BERT의 입력 표현 방식을 보여줍니다.

<div style="text-align: center;">
  <img src="/assets/images/BERT/Embeding.png" alt="BERT Embeddings" width="100%">
  <br>
  <em>Figure 2. Embedding of BERT.</em>
</div>
<br>

### 2.1 Special Tokens
입력을 구성할 때 두 가지 중요한 특수 토큰이 사용됩니다.
* **`[CLS]` (Classification Token):** 모든 입력 문장의 **맨 앞**에 위치합니다. 나중에 Classification Task(예: 감성 분석, NSP 등)를 수행할 때, 이 토큰의 벡터가 문장 전체의 정보를 함축하게 됩니다.
* **`[SEP]` (Separator Token):** 첫 번째 문장과 두 번째 문장을 **구분**하기 위해 문장의 끝에 삽입됩니다.

### 2.2 3가지 Embedding
1.  **Token Embedding:** WordPiece 기반으로 단어(Subword) 자체를 벡터화한 임베딩입니다.
2.  **Segment Embedding:** 입력된 토큰이 첫 번째 문장(Sentence A)인지, 두 번째 문장(Sentence B)인지를 구분해 주는 임베딩입니다.
    * BERT는 두 개의 문장을 입력받을 수 있으므로, 이를 구별할 식별자가 필요합니다.
3.  **Position Embedding:** Transformer와 마찬가지로 토큰의 위치 정보를 더해줍니다. (BERT는 학습 가능한 Position Embedding을 사용합니다.)

---

## 3. Pre-training Tasks
BERT는 언어의 문맥을 이해하기 위해 다음 두 가지 Task를 동시에 학습합니다.

### 3.1 Task 1: Masked Language Modeling (MLM)
기존의 단방향(Left-to-Right) 언어 모델과 달리, 양방향 문맥을 파악하기 위한 빈칸 채우기 학습 방식입니다.

* **개념:** 입력 문장 전체 토큰의 **15%**를 무작위로 마스킹(Masking)하고, 모델이 주변 문맥을 보고 가려진 단어를 예측하도록 학습합니다.
* **학습 방법:**
    * 입력: `나의 [MASK]는 개발자입니다.`
    * 정답: `직업`
    * 모델은 `[MASK]` 자리에 들어갈 단어의 확률 분포를 출력하고, 실제 정답과의 Loss(Cross Entropy)를 통해 학습합니다.
* **80-10-10 규칙 (Mismatch Mitigation):**
    Fine-tuning 단계에서는 `[MASK]` 토큰이 등장하지 않으므로, 학습과 실전의 괴리를 줄이기 위해 15%로 선정된 토큰을 다음과 같이 처리합니다.
    * **80%:** `[MASK]` 토큰으로 바꿈 (예: `my dog` -> `my [MASK]`)
    * **10%:** 랜덤한 다른 단어로 바꿈 (예: `my dog` -> `my apple`)
    * **10%:** 원래 단어 그대로 둠 (예: `my dog` -> `my dog`)

### 3.2 Task 2: Next Sentence Prediction (NSP)
문장과 문장 사이의 논리적 연결 관계를 학습하기 위한 Task입니다. QA(질의응답)나 NLI(자연어 추론) 같은 Task에서 중요하게 사용됩니다.

* **개념:** 두 문장(Sentence A, Sentence B)을 주고, B가 A 바로 다음에 오는 문장이 맞는지 아닌지를 **이진 분류(Binary Classification)** 합니다.
* **학습 방법:**
    * 입력 문장의 맨 앞에 있는 **`[CLS]` 토큰**의 출력 벡터를 사용하여 `IsNext` (연결됨) 또는 `NotNext` (연결 안 됨)를 예측합니다.
    * **데이터 구성:** 학습 데이터의 **50%**는 실제 이어지는 문장(IsNext), 나머지 **50%**는 랜덤하게 선택된 관계없는 문장(NotNext)으로 구성하여 균형을 맞춥니다.

---

## 4. Summary
BERT는 위와 같은 구조와 학습 방법을 통해 단어의 양방향 문맥을 깊이 있게 이해할 수 있게 되었으며, 다양한 NLP Task에서 SOTA(State-of-the-Art)를 기록했습니다.

* **Input:** Token + Segment + Position Embedding의 합
* **MLM:** 빈칸 채우기 (양방향 문맥 학습, 80-10-10 규칙 적용)
* **NSP:** 다음 문장 예측 (문장 간 관계 학습)



<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>