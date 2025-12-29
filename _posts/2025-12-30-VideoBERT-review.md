---
title:  "[Paper Review] VideoBERT: A Joint Model for Video and Language Representation Learning"
excerpt: "VL-BERT."
categories:
  - Paper Review
tags:
  - Deep Learning
  - Transformer
  - Multi-modal Learning
toc: true
toc_sticky: true
author_profile: true
mathjax: true
---



## 1. Introduction
기존의 비디오 학습 모델들은 대부분 **Labeled Data(지도 학습)**에 의존했다. 하지만 레이블이 달린 비디오 데이터는 구축 비용이 매우 비싸다. 반면, BERT는 대규모 텍스트 데이터에서 **Self-Supervised Learning(Masked LM)**을 통해 혁신적인 성능을 보였다.

이 논문은 **"비디오도 BERT처럼 학습할 수 없을까?"**라는 질문에서 시작된다. 비디오의 시간적(Temporal) 정보를 자연어 문장처럼 취급하여 레이블 없이 비디오의 표현(Representation)을 학습하는 것이 목표다. 이를 위해 가장 큰 걸림돌이었던 **"연속적인(Continuous) 비디오 프레임을 어떻게 BERT가 이해할 수 있는 이산적인(Discrete) 토큰으로 만들 것인가?"**를 해결하는 것이 핵심 아이디어다.

<br>

## 2. Methodology: Video to "Visual Words"
BERT는 이산적인 단어 토큰을 입력으로 받는다. 따라서 비디오 프레임을 단어처럼 만드는 **Video Discretization (Vector Quantization)** 과정이 필수적이다.

<div style="text-align: center;">
  <img src="/assets/images/VideoBERT/main.png" alt="VedioBERT Architecture" width="100%">
  <br>
  <em>Figure 1. Architecture of VideoBERT.</em>
</div>
<br>

---


### 2.1. Video Feature Extraction (S3D)
단순한 2D 이미지가 아닌 비디오의 **동작(Motion)과 시간적 흐름**을 담기 위해 **S3D (3D CNN)**를 사용한다.
* 비디오를 1.5fps(초당 1.5프레임)로 샘플링하여 클립 단위로 나눈다.
* Kinetics-400 데이터셋으로 사전 학습된 S3D 모델을 통해 각 클립에서 **1,024차원의 Feature Vector**를 추출한다.

### 2.2. Visual Word Generation (Hierarchical K-means)
추출된 Feature Vector들은 연속적인 값(Continuous)이다. 이를 토큰화하기 위해 **Hierarchical K-means Clustering**을 수행한다.
* 비디오 Feature들을 군집화(Clustering)하여 **약 20,480개의 클러스터(Centroid)**를 만든다.
* 각 클립이 속한 클러스터의 인덱스(Index)가 곧 **Visual Word**가 된다.
* 이로써 비디오 클립들이 텍스트의 '단어'와 같은 ID를 갖게 되어 BERT의 입력으로 들어갈 수 있다.

<div style="text-align: center;">
  <img src="/assets/images/VideoBERT/VWG.png" alt="Visual Word" width="100%">
  <br>
  <em>Figure 1.Visual Word Generation.</em>
</div>
<br>

<br>

## 3. Training Tasks (Pre-training Objectives)
VideoBERT는 텍스트(예: 요리 레시피 자막)와 비디오(Visual Words)를 결합하여 하나의 긴 시퀀스로 만들고, Transformer(BERT)에 넣어 학습한다.

### 3.1. Masked Language Modeling (MLM)
* 일반적인 BERT와 동일하게 텍스트 토큰의 일부를 `[MASK]` 처리하고 원본 단어를 예측한다.
* 이때, 비디오 정보(Visual Context)를 참고하여 텍스트를 채우게 된다.

### 3.2. Masked Visual-token Modeling (MVM)
* **개념:** 비디오 버전의 MLM이다. 비디오 시퀀스 중 일부 Visual Word를 `[MASK]` 처리한다.
* **목표:** 주변의 비디오 프레임과 텍스트 정보를 통해 마스킹된 부분의 **Visual Word ID(클러스터 번호)**를 맞춘다.
* **Classification:** 픽셀 값을 직접 예측(Regression)하는 것은 매우 어렵기 때문에, 미리 정의된 클러스터 ID를 맞추는 Classification 문제로 변환하여 고차원적인 문맥(High-level Semantics)을 학습하도록 설계했다.

### 3.3. Video-Text Alignment
* `[CLS]` 토큰을 활용하여 입력된 텍스트와 비디오가 실제로 짝(Matched)인지, 아니면 임의로 섞인 것(Not Matched)인지 이진 분류한다.

<br>

## 4. Conclusion & Insight
이 논문은 **Continuous한 비디오 데이터를 Discrete한 토큰(Visual Word)으로 변환**하여 NLP의 BERT 구조를 비디오 도메인에 성공적으로 이식했다는 데에 큰 의의가 있다.