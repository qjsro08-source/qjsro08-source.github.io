---
title:  "[Review] An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
excerpt: "ViT - Review."
categories:
  - Vision Transformer
tags:
  - Deep Learning
  - Transformer
  - Computer Vision
toc: true
toc_sticky: true
author_profile: true
mathjax: true
---




<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## 1. Introduction
**ViT(Vision Transformer)**는 2020년 구글 연구팀이 발표한 모델로, "An Image is Worth 16x16 Words"라는 논문 제목처럼 이미지를 마치 자연어 문장(Sequence)처럼 처리하여 Transformer에 입력하는 방식을 제안함.

기존의 CNN(Convolutional Neural Network)이 가지고 있던 Inductive Bias(지역적 특징, 평행 이동 불변성 등)를 제거하고, 대용량 데이터로 학습했을 때 CNN보다 더 뛰어난 성능을 보임을 입증했음.

<div style="text-align: center;">
  <img src="/assets/images/ViT/main.png" alt="ViT Architecture" width="90%">
  <br>
  <em>Figure 1. Overview of the Vision Transformer (ViT).</em>
</div>
<br>

---

## 2. ViT Architecture Pipeline
ViT의 전체적인 처리 과정은 다음과 같음.

### 2.1 Patch Partitioning (이미지 자르기)
Transformer는 1D Sequence를 입력으로 받기 때문에, 2D 이미지를 바로 넣을 수 없음. 따라서 이미지를 고정된 크기의 **패치(Patch)** 로 분할함.

* 이미지 크기: $$H \times W \times C$$(예:$$224 \times 224 \times 3$$)
* 패치 크기: $$P \times P$$(예:$$16 \times 16$$)
* 패치의 개수($$N$$): $$N = HW / P^2$$(예:$$196$$개)

이렇게 잘린 패치들은 마치 문장의 단어(Token)처럼 취급됨.

### 2.2 Linear Projection of Flattened Patches (Patch Embedding)
잘린 각 패치(3차원)를 1차원 벡터로 **Flatten** 한 뒤, Linear Layer(FC Layer)를 통과시켜 고정된 차원($$D$$)의 벡터로 만듦.

* 이 과정을 통해 이미지는 Transformer가 이해할 수 있는 **Embedding Vector**가 됨.
* CNN을 쓰지 않고 단순한 Linear Transformation만 사용했다는 것이 특징임.

### 2.3 [CLS] Token & Position Embedding
BERT와 유사하게 두 가지 추가적인 작업이 수행됨.

1.  **[CLS] Token 추가:**
    * 입력 시퀀스의 **맨 앞**에 학습 가능한 클래스 토큰(`[CLS]`)을 추가함.
    * Transformer를 통과한 후, 이 토큰의 출력값만이 이미지 전체의 분류(Classification) 결과를 예측하는 데 사용됨.
2.  **Position Embedding 더하기:**
    * Transformer는 패치의 순서(위치) 정보를 알지 못하므로, 위치 정보를 더해줘야 함.
    * ViT는 2D 위치 정보가 아닌, 학습 가능한 **1D Learnable Position Embedding**을 사용함.

$$\mathbf{z}_0 = [ \mathbf{x}_{class}; \mathbf{x}^1_p \mathbf{E}; \dots; \mathbf{x}^N_p \mathbf{E} ] + \mathbf{E}_{pos}$$

### 2.4 Transformer Encoder
임베딩된 벡터들은 표준 **Transformer Encoder**에 입력됨.
* **MSA (Multi-Head Self Attention):** 패치들 간의 관계(Global Context)를 학습함.
* **MLP (Multi-Layer Perceptron):** 비선형성을 추가하며 특징을 추출함.
* Layer Norm과 Residual Connection(Skip Connection)이 각 블록마다 적용됨.

### 2.5 MLP Head (Classification)
마지막으로, Encoder를 통과하여 나온 출력 중 **맨 앞의 [CLS] 토큰 벡터**만을 가져옴. 이 벡터를 MLP Head에 통과시켜 최종 클래스를 분류함.

---

## 3. Summary
ViT는 이미지를 패치 단위의 시퀀스로 변환하여 Transformer 구조를 Vision 분야에 성공적으로 이식했음.

1.  **Patch Partition:** 이미지를 16x16 패치로 분할
2.  **Linear Embedding:** 패치를 벡터로 변환 + [CLS] 토큰 추가
3.  **Position Embedding:** 위치 정보 학습 (Learnable)
4.  **Transformer Encoder:** 전역적인 특징 학습
5.  **Classification:** [CLS] 토큰의 출력 사용

CNN과 달리 이미지 전체를 한 번에 볼 수 있는 **Global Receptive Field**를 가지며, 대규모 데이터셋(JFT-300M 등)으로 사전 학습할 경우 SOTA 성능을 달성함.

