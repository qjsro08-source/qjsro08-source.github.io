---
title:  "[Review] Learning Transferable Visual Models From Natural Language Supervision"
excerpt: "CLIP"
categories:
  - Multi-Modal Learning
tags:
  - Deep Learning
  - Transformer
  - Multi-modal Learning
toc: true
toc_sticky: true
author_profile: true
mathjax: true
---

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## 1. Introduction
**CLIP(Contrastive Language–Image Pre-training)**은 OpenAI가 2021년에 발표한 모델로, 이미지와 텍스트를 **같은 의미 공간(embedding space)** 으로 정렬(alignment)시키도록 학습하는 방식이 핵심임.

기존의 “특정 태스크(분류/VQA)를 위한 라벨 학습”과 달리, CLIP은 대규모 **이미지-텍스트 쌍**을 이용해 **대조학습(contrastive learning)** 을 수행하여

- 이미지 ↔ 텍스트를 서로 매칭할 수 있고
- 별도 파인튜닝 없이도 **Zero-shot 분류 / 검색(Retrieval)** 이 가능한

범용 멀티모달 표현을 학습함.

---

## 2. Model Architecture
CLIP은 “한 Transformer에 이미지+텍스트를 같이 넣는(fusion)” 구조가 아니라, **이미지 인코더와 텍스트 인코더를 분리한 Dual-Encoder** 구조를 사용함.

### 2.1 Two Encoders (Dual-Encoder)
- **Image Encoder**: ResNet 또는 ViT 같은 비전 모델
- **Text Encoder**: Transformer 기반 텍스트 인코더

각각 입력을 받아 최종적으로 **고정 길이 벡터(embedding)** 를 출력함.

- 이미지 → $$\mathbf{v} \in \mathbb{R}^d$$  
- 텍스트 → $$\mathbf{t} \in \mathbb{R}^d$$  

중요한 점은 두 벡터가 **같은 차원 $$d$$** 로 나오고, 학습을 통해 **“정답 이미지-텍스트는 가깝게 / 오답은 멀게”** 배치되도록 만든다는 것임.

### 2.2 Similarity (유사도 계산)
CLIP은 보통 **cosine similarity**(혹은 정규화 후 내적)로 유사도를 측정함.

- 정규화:
  - $$\hat{\mathbf{v}}=\frac{\mathbf{v}}{\|\mathbf{v}\|}, \quad \hat{\mathbf{t}}=\frac{\mathbf{t}}{\|\mathbf{t}\|}$$
- 유사도:
  - $$s(\mathbf{v},\mathbf{t})=\hat{\mathbf{v}}^\top \hat{\mathbf{t}}$$

또한 softmax의 “뾰족함”을 조절하기 위해 **temperature(또는 logit scale)** 파라미터를 학습함.

- $$\text{logit}_{ij} = \alpha \cdot (\hat{\mathbf{v}}_i^\top \hat{\mathbf{t}}_j)$$  
  여기서 $$\alpha$$ 는 learnable scale (temperature의 역수 역할)

---

## 3. Pre-training Task (Contrastive Learning)
CLIP의 프리트레이닝 목표는 한 줄로 요약 가능함.

> 같은 쌍(image-caption)은 가깝게, 다른 쌍은 멀게.

<div style="text-align: center;">
  <img src="/assets/images/CLIP/main1.png" alt="Contrastive Learning in CLIP" width="100%">
  <br>
  <em>Figure 1. Contrastive learning objective (positive pairs close, negatives far).</em>
</div>
<br>

배치에 $$N$$개의 정답쌍 $$\{(I_i, T_i)\}_{i=1}^N$$ 이 있다고 하자.

- 이미지 임베딩: $$\{\hat{\mathbf{v}}_1,\dots,\hat{\mathbf{v}}_N\}$$
- 텍스트 임베딩: $$\{\hat{\mathbf{t}}_1,\dots,\hat{\mathbf{t}}_N\}$$

모든 조합의 유사도를 계산해 $$N \times N$$ 유사도(=logit) 행렬을 만들고,  
각 이미지가 **정답 텍스트를 고르도록**, 각 텍스트가 **정답 이미지를 고르도록** 학습함.

### 3.1 Image → Text 방향 loss (I2T)
각 이미지 $$i$$에 대해 정답 텍스트는 $$i$$번째 하나뿐이므로, 행(row) 기준 softmax CE를 적용함.

- $$p(j|i)=\frac{\exp(\text{logit}_{ij})}{\sum_{k=1}^{N}\exp(\text{logit}_{ik})}$$
- $$\mathcal{L}_{\text{I2T}} = -\frac{1}{N}\sum_{i=1}^N \log p(i|i)$$

### 3.2 Text → Image 방향 loss (T2I)
반대로 텍스트 기준으로도 동일하게 열(column) 기준 softmax CE를 적용함.

- $$q(i|j)=\frac{\exp(\text{logit}_{ij})}{\sum_{k=1}^{N}\exp(\text{logit}_{kj})}$$
- $$\mathcal{L}_{\text{T2I}} = -\frac{1}{N}\sum_{j=1}^N \log q(j|j)$$

### 3.3 Final CLIP Loss
최종 loss는 두 방향을 평균내는 형태로 자주 사용함.

- $$\mathcal{L}_{\text{CLIP}} = \frac{1}{2}\left(\mathcal{L}_{\text{I2T}} + \mathcal{L}_{\text{T2I}}\right)$$

**포인트**
- MLM처럼 “토큰 복원”을 하지 않음
- ROI classification처럼 “detector 라벨”에 의존하지 않음
- 오직 **(이미지, 캡션) 짝맞추기**만으로 큰 표현 학습을 달성함

---

## 4. What can CLIP do? (Zero-shot / Retrieval)
CLIP의 가장 유명한 장점은 **파인튜닝 없이도(Zero-shot)** 다양한 태스크에 바로 쓸 수 있다는 점임.

### 4.1 Zero-shot Image Classification (Label Text로 분류기 만들기)
CLIP은 “분류”를 **이미지-텍스트 매칭 문제**로 바꿔서 해결함.

- 클래스 라벨(예: `dog`, `cat`)을 텍스트 프롬프트로 바꿈  
  예: `"a photo of a {label}"`
- 각 라벨 프롬프트의 텍스트 임베딩을 미리 계산해 “분류기 후보”로 사용
- 입력 이미지 임베딩과 가장 유사한 라벨을 예측

<div style="text-align: center;">
  <img src="/assets/images/CLIP/main2.png" alt="Create Dataset Classifier from Label Text" width="100%">
  <br>
  <em>Figure 2. Build a classifier from label text (prompts) for zero-shot classification.</em>
</div>
<br>

수식으로 쓰면 다음과 같음.

- 각 클래스 $$c$$에 대해 프롬프트 문장 만들기:
  - `"a photo of a {c}"`
- 텍스트 임베딩:
  - $$\hat{\mathbf{t}}_c$$
- 입력 이미지 임베딩:
  - $$\hat{\mathbf{v}}$$
- 예측:
  - $$\arg\max_c \ \hat{\mathbf{v}}^\top \hat{\mathbf{t}}_c$$

### 4.2 Image–Text Retrieval (검색)
검색도 동일한 원리로 동작함.

- 텍스트로 이미지 검색: 텍스트 임베딩 $$\hat{\mathbf{t}}$$ 와 후보 이미지 임베딩 $$\hat{\mathbf{v}}_1,\dots,\hat{\mathbf{v}}_M$$ 의 유사도 top-k 선택
- 이미지로 텍스트 검색: 반대로 이미지 임베딩을 쿼리로 사용

즉, CLIP은 학습된 “공유 임베딩 공간”에서 **유사도 계산만으로** 검색을 수행 가능함.

---

## 5. Summary
CLIP은 “이미지+텍스트를 한 모델에서 융합(fusion)하는 구조”가 아니라,  
**두 모달을 같은 공간에 정렬(alignment)하는 구조**임.

1. **Architecture:** Image encoder + Text encoder의 **Dual-Encoder**
2. **Training Objective:** 정답쌍은 가깝게 / 오답쌍은 멀게 하는 **대조학습(contrastive loss)**
3. **Capability:** 파인튜닝 없이도  
   - **Zero-shot 분류** (라벨을 텍스트 프롬프트로 변환)  
   - **Image–Text Retrieval**  
   등이 가능해짐