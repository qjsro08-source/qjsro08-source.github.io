---
title:  "Training data-efficient image transformers & distillation through attention"
excerpt: "DeiT - Review."
categories:
  - Paper Review
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
**DeiT(Data-efficient Image Transformers)**는 Facebook AI Research(FAIR)에서 발표한 논문으로, ViT(Vision Transformer)가 학습 데이터가 적을 때(예: ImageNet-1k) 성능이 잘 나오지 않는 문제를 해결하고자 함.

* **문제:** ViT는 Inductive Bias가 부족하여 JFT-300M 같은 초대형 데이터셋이 없으면 학습이 어려움.
* **해결책:** **Knowledge Distillation(지식 증류)** 기법을 도입하여, 적은 데이터로도 CNN만큼의 성능을 내도록 함.

<div style="text-align: center;">
  <img src="/assets/images/DeiT/main.png" alt="DeiT Architecture" width="90%">
  <br>
  <em>Figure 1. The distillation procedure of DeiT.</em>
</div>
<br>

---

## 2. Distillation Strategy
DeiT는 "CNN은 적은 데이터로도 학습이 잘 된다(Inductive Bias가 높다)"는 점을 이용함. 즉, **잘 학습된 CNN을 선생님(Teacher)으로 모시고, Transformer(Student)가 이를 배우게 하는 방식**임.

### 2.1 Teacher & Student
* **Teacher Model:** RegNetY (CNN 계열). 이미지의 지역적 특징(Locality)을 잘 파악함.
* **Student Model:** ViT (Transformer). Teacher가 알려주는 정보를 통해 Inductive Bias 부족 문제를 완화함.

### 2.2 Distillation Token
DeiT의 가장 큰 특징은 **Distillation Token**을 도입했다는 점임.
기존 BERT/ViT의 `[CLS]` 토큰 외에, `[Distill]` 토큰을 하나 더 추가하여 입력으로 넣음.

* **[CLS] Token:** 실제 정답(Ground Truth, Label)과 일치하도록 학습.
* **[Distill] Token:** Teacher 모델의 출력(Prediction)과 일치하도록 학습.

이 두 토큰은 Self-Attention을 통해 서로 정보를 주고받으며 학습됨.

---

## 3. Loss Functions (Soft vs Hard)
Student 모델을 학습시키는 방법(Loss)에는 두 가지가 있음.

### 3.1 Soft Distillation
일반적인 Knowledge Distillation 방법임. Teacher의 Softmax 출력 분포(Soft probabilities)를 Student가 따라가도록 함.
* Teacher의 불확실성이나 뉘앙스까지 배울 수 있다는 장점이 있음.
* **Loss 수식:**
  $$L_{soft} = (1-\lambda) L_{CE}(\psi(Z_s), y) + \lambda \tau^2 KL(\psi(Z_s/\tau), \psi(Z_t/\tau))$$
  *(CE: Cross Entropy, KL: KL Divergence, $$\tau$$: Temperature)*

### 3.2 Hard Distillation (DeiT's Choice)
DeiT 저자들은 Soft Label 대신, Teacher가 예측한 **가장 확률이 높은 클래스(Hard Label)** 를 정답지로 보고 학습하는 방식을 택함.
* **이유:** Teacher 모델도 완벽하지 않으므로, 애매한 확률 분포(Soft Label)를 따라가기보다 Teacher가 "이거다!"라고 찍은 정답(Hard Label)을 배우는 것이 더 효과적임.
* **Loss 수식:**
  $$L_{hard} = \frac{1}{2} L_{CE}(\psi(Z_s), y) + \frac{1}{2} L_{CE}(\psi(Z_s), y_t)$$
  *($$y$$: 실제 정답, $$y_t$$: Teacher가 예측한 Hard Label)*

### 3.3 Result
실험 결과, **Hard Distillation** 방식이 Soft Distillation보다 성능이 더 우수했음. CNN Teacher의 강력한 Inductive Bias를 "정답" 형태로 확실하게 주입받는 것이 ViT 학습에 유리하게 작용함.

---

## 4. Summary
DeiT는 Transformer 구조를 거의 바꾸지 않으면서, **Distillation Token**과 **Hard Distillation** 전략만으로 ImageNet 데이터셋에서 SOTA급 성능을 달성함.

1.  **CNN Teacher:** CNN의 장점(Inductive Bias)을 Transformer에 전수.
2.  **Distillation Token:** Teacher의 지식을 전담해서 배우는 특수 토큰 추가.
3.  **Hard Distillation:** Teacher의 확실한 예측값을 따라가는 것이 성능 향상에 도움됨.