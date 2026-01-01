---
title:  "[Review] VL-BERT: PRE-TRAINING OF GENERIC VISUALLINGUISTIC REPRESENTATIONS"
excerpt: "VL-BERT."
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
**VL-BERT (Vision-Language BERT)**는 2020년 ICLR에 채택된 모델로, 텍스트만 처리하던 BERT의 강력한 성능을 **이미지+텍스트(Visual-Linguistic)** 멀티모달 영역으로 확장한 모델임.

기존 BERT 아키텍처를 거의 그대로 사용하되, **Visual Feature Embedding**을 추가하여 이미지 정보도 마치 단어 토큰처럼 처리할 수 있게 만든 것이 핵심임. 이를 통해 VQA(Visual Question Answering) 같은 Task에서 뛰어난 성능을 보임.

<div style="text-align: center;">
  <img src="/assets/images/VL_BERT/main.png" alt="VL-BERT Architecture" width="100%">
  <br>
  <em>Figure 1. Architecture of VL-BERT.</em>
</div>
<br>

---

## 2. Model Architecture
VL-BERT는 Transformer Encoder 구조(BERT)를 기반으로 하며, 입력으로 **문장(Sentence)** 과 **이미지(Image)** 를 동시에 받음.

### 2.1 Input Representation
입력 시퀀스는 다음과 같이 구성됨.

`[CLS], Word_1, Word_2, ..., [SEP], ROI_1, ROI_2, ..., [SEP]`

1.  **Linguistic Elements (텍스트):** 기존 BERT와 동일하게 WordPiece 토큰화 진행.
2.  **Visual Elements (이미지):**
    * 이미지 전체를 넣는 것이 아니라, **Faster R-CNN** 같은 Object Detector를 통해 추출한 **RoI(Region of Interest, 관심 영역)** 들을 각각 하나의 토큰으로 취급함.
    * 즉, "강아지", "고양이", "배경" 같은 이미지 조각들이 토큰으로 들어감.

### 2.2 Visual Feature Embedding (BERT와의 차이점)
텍스트와 이미지를 동일한 차원으로 맞추기 위해, VL-BERT는 기존 BERT 임베딩에 **Visual Feature Embedding**을 추가함.

* **Visual Appearance Embedding:** Faster R-CNN에서 추출한 해당 영역(RoI)의 특징 벡터.
* **Visual Geometry Embedding:** 해당 영역이 이미지의 어디에 위치하는지(좌표 $$x_{min}, y_{min}, x_{max}, y_{max}$$)를 나타내는 위치 정보.

이 두 가지를 결합하여 최종적으로 **이미지 토큰의 벡터**를 생성함.

---

## 3. Pre-training Tasks
VL-BERT는 대규모 이미지-텍스트 데이터셋(Conceptual Captions 등)을 사용하여 두 가지 Task를 학습함.

### 3.1 Task 1: Masked Language Modeling with Visual Clues (MLM)
기존 BERT의 MLM과 동일하지만, **이미지 정보**를 힌트로 사용할 수 있다는 점이 다름.

* **입력:** `[CLS] 여자가 [MASK]를 쓰고 있다 [SEP] (우산 쓴 여자 이미지 RoI들) [SEP]`
* **과정:** 텍스트의 15%를 마스킹하고 맞추게 함.
* **특징:** 모델은 단순히 문맥만 보는 게 아니라, 뒤에 붙은 **이미지 토큰(Visual Clues)** 을 참고하여 `[MASK]`가 '우산'임을 유추함.

### 3.2 Task 2: Masked RoI Classification with Linguistic Clues
텍스트뿐만 아니라 **이미지 영역(RoI)** 도 마스킹하고 이를 맞추는 Task임.

* **과정:**
    1.  입력 이미지 토큰(RoI) 중 15%를 무작위로 선택하여 픽셀 값을 0으로 만듦(Masking).
    2.  모델이 마스킹된 영역이 무엇인지 분류(Classification)하게 함.
* **정답(Label):** 사람이 라벨링한 것이 아니라, **Object Detector(Faster R-CNN)가 예측한 클래스**를 정답으로 사용함.
* **특징:** 모델은 주변의 다른 이미지 조각들과 **텍스트(Linguistic Clues)** 를 힌트로 사용하여 가려진 부분이 무엇인지 맞춤.

---

## 4. What is VQA? (Visual Question Answering)
VL-BERT가 풀고자 하는 대표적인 Task인 **VQA**에 대해 간단히 정리함.

**VQA(Visual Question Answering)** 란, 컴퓨터에게 **이미지**와 그 이미지에 대한 **자연어 질문**을 주었을 때, 올바른 **답변**을 예측하는 인공지능 분야임.

* **Example:**
    * **Input Image:** 어떤 사람이 벤치에 앉아 있는 사진
    * **Question:** "벤치에 앉아 있는 사람은 무엇을 입고 있나요?"
    * **Model Output:** "파란색 셔츠"

단순히 이미지만 보는 것(Vision)도 아니고, 텍스트만 보는 것(NLP)도 아닌, **두 정보를 결합하여 추론(Reasoning)** 해야만 풀 수 있는 고난이도 Task임. VL-BERT는 Pre-training을 통해 이 능력을 극대화함.

---

## 5. Summary
VL-BERT는 "이미지 영역(RoI)"을 "단어 토큰"과 대등한 위치로 격상시켜, 하나의 Transformer 안에서 통합 학습(Pre-training)을 수행한 모델임.

1.  **Architecture:** BERT에 Visual Feature Embedding을 추가하여 멀티모달 입력 처리.
2.  **Input:** `[CLS] 텍스트 [SEP] 이미지_RoI [SEP]` 구조.
3.  **Pre-training:**
    * **MLM:** 이미지 힌트를 보고 가려진 단어 맞추기.
    * **Masked RoI Classification:** 텍스트 힌트를 보고 가려진 이미지 영역 맞추기.

이러한 구조 덕분에 VQA, Visual Commonsense Reasoning(VCR) 등 다양한 Vision-Language Task에서 SOTA 성능을 기록함.