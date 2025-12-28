---
title:  "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
excerpt: "SwinTransformer-Review."
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
**Swin Transformer**는 ICCV 2021에서 발표되어 Best Paper 상(Marr Prize)을 받은 논문임. 기존 ViT는 이미지 분류에는 강했지만, Object Detection이나 Segmentation 같은 Dense Prediction 태스크에는 적용하기 어려웠음.

* **ViT의 문제점:**
    1.  이미지 전체를 한 번에 보므로 해상도가 높을수록 연산량이 제곱($$N^2$$)으로 폭증함.
    2.  단일 해상도(Single Scale)만 사용하여, 다양한 크기의 물체를 인식해야 하는 태스크에 불리함.
* **Swin의 해결책:** CNN의 장점인 **계층적 구조(Hierarchical Feature map)** 와 **지역성(Locality)** 을 Transformer에 도입함.

<div style="text-align: center;">
  <img src="/assets/images/Swin/main.png" alt="Swin Transformer Architecture" width="100%">
  <br>
  <em>Figure 1. Architecture of Swin Transformer.</em>
</div>
<br>

---

## 2. Key Components
Swin Transformer는 크게 두 가지 핵심 아이디어로 구성됨.

### 2.1 Patch Merging (Hierarchical Downsampling)
Swin Transformer의 Patch Merging은 "패치를 다시 자르는 것"이라기보다, 이미 만들어진 토큰들을 $$2 \times 2$$로 묶어 하나의 토큰으로 **병합(Token Merging)** 하는 다운샘플링 레이어임. CNN의 Pooling처럼 해상도(토큰 격자)는 줄이고 채널은 늘려 계층적(Hierarchical) feature pyramid를 만듦.

#### **수학적/절차적 과정**
입력 토큰 feature가 $$\mathbf{X} \in \mathbb{R}^{H \times W \times C}$$일 때:

1.  **$$2 \times 2$$Grouping:** 인접한$$2 \times 2$$ 토큰을 한 그룹으로 묶음.
2.  **Concat:** 4개의 토큰을 채널 방향으로 합침(Concatenation).
    * 텐서 변화: $$\mathbb{R}^{H \times W \times C} \rightarrow \mathbb{R}^{\frac{H}{2} \times \frac{W}{2} \times 4C}$$
3.  **Linear Projection:** 채널이 너무 커지는 것을 막기 위해 $$4C \rightarrow 2C$$로 투영함.
    * 최종 변화: $$\mathbb{R}^{\frac{H}{2} \times \frac{W}{2} \times 2C}$$

#### **Layer 깊이에 따른 변화**
한 번의 Patch Merging마다 다음과 같이 변하며, 토큰이 대표하는 **Effective Patch(수용 영역)** 는 점점 커짐.
* **토큰 격자:** $$(H, W) \rightarrow (H/2, W/2)$$
* **토큰 수:** $$N \rightarrow N/4$$
* **채널:** $$C \rightarrow 2C$$

이러한 구조 덕분에 FPN(Feature Pyramid Network)이나 U-Net 같은 구조에 바로 적용이 가능함.

### 2.2 Window-based Self-Attention
Global Attention 대신, 이미지를 작은 윈도우($$M \times M$$, 보통 $$M=7$$)로 나누고 **그 윈도우 안에서만** Self-Attention을 수행하여 연산량을 줄임.
하지만 이 방식만 쓰면 윈도우 경계를 넘어선 정보 교류가 끊기는 문제가 발생함.

---

## 3. Shifted Window & Efficient Batch Computation
윈도우 간 정보 교류를 위해 Swin은 블록을 연속으로 쌓을 때, 다음 블록에서 윈도우 분할 그리드 자체를 일정량 이동(Shift)해서 Attention을 수행함. 이것이 **SW-MSA(Shifted Window MSA)** 의 핵심임.

### 3.1 Shifted Window Partitioning
* **블록 $$l$$(W-MSA):** 일반적인 격자 기준으로$$M \times M$$ 윈도우 분할.
* **블록 $$l+1$$(SW-MSA):** 윈도우 격자를$$(s, s)$$만큼 이동한 기준으로 분할 (보통 $$s = \lfloor M/2 \rfloor$$).

이렇게 하면 이전 블록에서 서로 다른 윈도우에 있었던 토큰들이, 다음 블록에서는 같은 윈도우에 함께 포함될 수 있어 **Cross-window connection(윈도우 간 정보 흐름)** 이 자연스럽게 생김.

### 3.2 Cyclic Shift & Attention Mask (Efficiency Trick)
실제 구현에서는 계산 효율을 위해 **Cyclic Shift**와 **Attention Mask**를 사용함. 이 두 가지가 SW-MSA를 "빠르게" 만들면서도 "원래 의도한 이웃끼리만" 섞이게 하는 핵심 장치임.

<div style="text-align: center;">
  <img src="/assets/images/Swin/cyclic-shift.png" alt="Cyclic Shift Mechanism" width="80%">
  <br>
  <em>Figure 2. Illustration of cyclic shift and masked self-attention.</em>
</div>
<br>

#### **1) Cyclic Shift (순환 이동)**
* **정의:** Feature map을 위/왼쪽으로 $$s$$칸 이동시키되, 밀려나간 부분을 반대편으로 감아서(wrap-around) 붙이는 연산 (PyTorch의 `torch.roll`).
* **연산:** $$X_{shift} = \text{Roll}(X, (-s, -s))$$
* **사용 이유:**
    * 윈도우 그리드를 옮겨서 자르면 가장자리에 $$4 \times 4$$가 아닌 $$2 \times 4$$, $$4 \times 2$$ 같은 자투리 윈도우가 생겨 배치 연산이 매우 비효율적임.
    * 대신 **이미지 자체를 밀어버리고(Roll)** 평범한 $$M \times M$$ 윈도우로 자르면, 모든 윈도우 크기가 일정해서 GPU 연산에 유리함.

#### **2) Attention Mask (가짜 이웃 차단)**
Cyclic Shift로 인해 이미지의 끝단이 반대편 끝단과 붙게 되어, 실제로는 이웃이 아닌데 한 윈도우에 들어오는 **"가짜 이웃(Fake Neighbor)"** 이 발생함. 이를 해결하기 위해 마스크를 사용함.

* **역할:** 윈도우 내부 Attention 계산 시, $$Mask$$를 더해줌.
    $$Attn = \text{Softmax}(\frac{QK^T}{\sqrt{d}} + B + Mask)V$$
    * **허용되는 토큰 쌍 (진짜 이웃):** $$0$$ 추가
    * **금지할 토큰 쌍 (가짜 이웃):** $$-\infty$$ 추가 (Softmax 후 확률 0)

#### **3) 예시: $$8 \times 8$$토큰, 윈도우$$M=4$$, 쉬프트 $$s=2$$**
* **Cyclic Shift:** $$(-2, -2)$$로 Roll을 하면, 원래 오른쪽/아래에 있던 영역(B, C, D)이 왼쪽/위로 말려 들어옴.
* **Window Partition:** 이 상태에서 $$4 \times 4$$로 깔끔하게 자르면, 하나의 윈도우 안에 서로 다른 영역(A, B, C, D)이 섞여 들어감.
    ```text
    [ A A | B B ]  <-- 한 윈도우 내부
    [ A A | B B ]
    [ ----+---- ]
    [ C C | D D ]
    [ C C | D D ]
    ```
* **Masking:** 마스크를 통해 A끼리, B끼리, C끼리, D끼리만 Attention 하도록 제한하고, A $$\leftrightarrow$$ B 처럼 서로 다른 영역 간의 연산은 차단함.

#### **4) 전체 처리 흐름**
1.  **Shift:** $$X \rightarrow X_{shift}$$ (Cyclic Shift)
2.  **Partition:** $$M \times M$$ 윈도우로 분할 (모두 같은 크기)
3.  **Attn:** Window Attention 수행 + **Mask 적용**
4.  **Merge:** 다시 전체 이미지로 합침
5.  **Reverse Shift:** 원래 좌표로 복구 (Reverse Roll)

### 3.3 Relative Position Bias
기존 ViT의 절대적 위치 임베딩(Absolute Positional Embedding) 대신, 토큰 간의 **상대적 거리(Relative Position)** 정보를 Attention Logit에 더해줌.

$$Attention(Q, K, V) = Softmax(\frac{QK^T}{\sqrt{d}} + B)V$$

여기서 $$B$$는 단순한 고정값이 아니라 **학습 가능한 파라미터 테이블(Look-up Table)** 에서 상대 좌표 인덱스에 맞춰 가져오는 값임. 이 방식은 평행 이동 불변성(Translation Invariance)을 반영할 수 있어 성능 향상에 기여함.

---

## 4. Summary
Swin Transformer는 Transformer를 범용적인 Vision Backbone으로 만들기 위해 CNN의 구조적 장점을 영리하게 가져왔음.

1.  **Window Attention:** 연산량을 획기적으로 줄임 (Linear Complexity).
2.  **Shifted Window (Cyclic Shift + Masking):** 효율적인 연산(고정된 윈도우 크기)과 윈도우 간 정보 흐름 연결을 동시에 달성.
3.  **Hierarchical Structure (Patch Merging):** 레이어가 깊어질수록 해상도를 줄이고 채널을 늘려 다양한 스케일의 특징을 학습함.
4.  **Relative Position Bias:** Look-up Table 기반의 상대 좌표 학습으로 위치 정보를 효과적으로 주입함.

결과적으로 ImageNet 분류뿐만 아니라 COCO Object Detection, ADE20K Segmentation 등 다양한 태스크에서 SOTA를 달성함.