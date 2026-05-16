---
title:  "[Review] BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models"
excerpt: "BLIP-2"
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

기존 Vision-Language(VL) 사전학습 모델은 크게 두 계열로 나뉜다.

- **Encoder-based(dual-encoder / fusion-encoder)**: 이미지-텍스트 정렬 및 retrieval/분류 등 이해(understanding) 태스크에 강하지만, 자연스러운 텍스트 생성(generation)에는 구조적으로 불리하다.
- **Encoder-Decoder 기반**: 캡션 생성, VQA 설명 생성 등 생성(generation)에 적합하지만, retrieval(특히 dual-encoder 기반 대규모 검색)에는 비효율적이거나 성능상 한계가 있다.

또한 데이터 관점에서, 웹에서 수집한 대규모 이미지-텍스트 페어는 규모는 크지만 **노이즈(잘못된 캡션, 약한 관련성, 스팸/중복 등)** 가 섞여 성능을 떨어뜨린다.

**BLIP**은 이 두 문제를 동시에 해결하고자 한다.

1. **모델 구조 문제 해결**: 하나의 프레임워크 안에서 understanding과 generation을 모두 커버하는 **MED (Multimodal Mixture of Encoder-Decoder)** 를 제안
2. **데이터 노이즈 문제 해결**: 웹 데이터의 노이즈를 줄이고 학습 데이터를 부트스트래핑하는 **CapFilt (Captioning and Filtering)** 파이프라인 제안

---

## 2. Model Architecture (MED)

<div style="text-align: center;">
  <img src="/assets/images/BLIP/model.png" alt="BLIP MED architecture" width="100%">
  <br>
  <em>Figure 1. BLIP의 MED(Multimodal Mixture of Encoder-Decoder) 구조와 3가지 pre-training objective(ITC, ITM, LM).</em>
</div>
<br>

BLIP의 핵심은 하나의 텍스트 Transformer를 “모드 전환”하여 3가지 역할로 쓰는 것이다.

### 2.1 구성 요소

- **Image Encoder**: ViT 기반. 이미지를 패치 토큰으로 변환해 시각 임베딩 시퀀스를 만든다.
- **Text Transformer**: BERT 계열 구조를 기반으로 하되, 사용 모드에 따라 attention mask / cross-attention 구성이 달라진다.
- 텍스트 입력에는 **[CLS] 토큰**을 붙여 문장 대표 임베딩을 얻고, ITM에서는 주로 [CLS]를 분류 헤드에 넣는다.

### 2.2 MED의 3가지 모드

BLIP 논문에서 MED는 아래 3 모드를 파라미터 공유로 묶는다.

1. **Unimodal Text Encoder**
   - 텍스트만 입력
   - **Bi-directional self-attention** (BERT 방식)

2. **Image-grounded Text Encoder (fusion encoder)**
   - 텍스트 self-attention은 bidirectional
   - 텍스트가 이미지 토큰에 **cross-attention**
   - 이미지-텍스트 결합 표현(understanding)에 유리
   - **ITM(매칭 분류)** 에서 사용

3. **Image-grounded Text Decoder (encoder-decoder)**
   - 텍스트 self-attention은 **causal mask** (auto-regressive)
   - 텍스트가 이미지 토큰에 cross-attention
   - **LM(캡션/설명 생성)** 에서 사용

---

## 3. Pre-training Objectives

BLIP은 아래 3개의 objective를 joint하게 학습한다.

- **(I) ITC**: Image-Text Contrastive Learning
- **(II) ITM**: Image-Text Matching
- **(III) LM**: Image-conditional Language Modeling

---

## 3.1 (I) Image-Text Contrastive Loss (ITC)

BLIP의 ITC는 CLIP-style 대조학습을 기반으로 하며, 학습 안정성과 노이즈 강건성을 위해 **ALBEF의 momentum distillation + queue** 방식을 따른다.

### 3.1.1 ALBEF-style ITC 핵심 구성

- **Student(online) encoder**: gradient로 업데이트
- **Teacher(momentum) encoder**: student의 EMA(Exponential Moving Average)로 업데이트 (gradient 사용 X)
- Teacher가 만든 feature를 **queue**에 저장해 큰 negative set을 구성
- hard one-hot 타깃 대신 **Teacher가 만든 soft target 분포**로 student를 학습

웹 캡션은 노이즈가 있어 표면적으로는 negative인 샘플도 의미적으로는 어느 정도 관련이 있을 수 있다. hard one-hot만 사용하면 과도한 밀어내기가 발생할 수 있어, soft target distillation이 이를 완화한다.

---

### 3.1.2 Notation

- 이미지 임베딩: $$\mathbf{v} \in \mathbb{R}^d$$  
- 텍스트 임베딩: $$\mathbf{t} \in \mathbb{R}^d$$  

두 벡터는 같은 차원 $$d$$ 로 매핑되며, 학습을 통해 정답 이미지-텍스트 쌍은 가깝게, 오답은 멀게 배치되도록 한다.

배치 크기를 $$B$$ 라고 할 때, 배치 내 이미지/텍스트를 각각 $$\{\mathbf{v}_i\}_{i=1}^B$$, $$\{\mathbf{t}_i\}_{i=1}^B$$ 로 둔다.

---

### 3.1.3 Similarity (정규화 후 내적)

정규화:

- $$\hat{\mathbf{v}}_i = \frac{\mathbf{v}_i}{\|\mathbf{v}_i\|}, \quad \hat{\mathbf{t}}_j = \frac{\mathbf{t}_j}{\|\mathbf{t}_j\|}$$

temperature $$\tau$$ 를 두고 유사도(logit)를

- $$s_{ij} = \frac{\hat{\mathbf{v}}_i^\top \hat{\mathbf{t}}_j}{\tau}$$

로 정의한다.

(실제로는 batch 밖의 negative를 늘리기 위해 queue에 저장된 임베딩도 후보 집합에 포함한다.)

---

### 3.1.4 Student 확률 분포 (i2t / t2i)

Image-to-Text 분포:

- $$p^{i2t}_{ij} = \frac{\exp(s_{ij})}{\sum_{k}\exp(s_{ik})}$$

Text-to-Image 분포:

- $$p^{t2i}_{ij} = \frac{\exp(s_{ji})}{\sum_{k}\exp(s_{jk})}$$

여기서 분모의 합은 “현재 step에서의 후보 텍스트/이미지 집합” 전체에 대해 수행되며, 구현에서는 queue까지 포함해 더 큰 집합으로 softmax를 취한다.

---

### 3.1.5 Teacher soft target 분포 (momentum encoder)

Momentum encoder(teacher)로부터 얻은 임베딩을 사용해 teacher 유사도 $$s^{m}_{ij}$$ 를 계산한다.  
그 후 teacher의 soft target을

- $$q^{i2t}_{ij} = \frac{\exp(s^{m}_{ij})}{\sum_{k}\exp(s^{m}_{ik})}$$
- $$q^{t2i}_{ij} = \frac{\exp(s^{m}_{ji})}{\sum_{k}\exp(s^{m}_{jk})}$$

로 둔다.

---

### 3.1.6 Hard target (정답 인덱스)

배치에서 $$i$$번째 이미지와 $$i$$번째 텍스트가 정답(pair)이라고 할 때, hard target은 one-hot으로

- $$y^{i2t}_{ij} = \mathbb{1}[j=i]$$
- $$y^{t2i}_{ij} = \mathbb{1}[j=i]$$

로 둘 수 있다.

---

### 3.1.7 ITC Loss (hard + soft target)

Cross entropy를 $$H(\mathbf{a}, \mathbf{b})$$ 로 표기하고, distillation 비중을 $$\alpha$$ 라고 두면 ITC 손실은

- $$\mathcal{L}_{ITC} = \frac{1}{2}\Big( (1-\alpha)\,H(\mathbf{y}^{i2t}_i,\mathbf{p}^{i2t}_i) + \alpha\,H(\mathbf{q}^{i2t}_i,\mathbf{p}^{i2t}_i) \;+\; (1-\alpha)\,H(\mathbf{y}^{t2i}_i,\mathbf{p}^{t2i}_i) + \alpha\,H(\mathbf{q}^{t2i}_i,\mathbf{p}^{t2i}_i) \Big)$$

여기서 $$\mathbf{p}^{i2t}_i$$ 는 $$i$$번째 이미지에 대한 분포(길이: 후보 텍스트 수), $$\mathbf{p}^{t2i}_i$$ 는 $$i$$번째 텍스트에 대한 분포(길이: 후보 이미지 수)이며, $$\mathbf{q}$$ 도 동일한 의미로 teacher 분포를 나타낸다.

---

## 3.2 (II) Image-Text Matching Loss (ITM)

ITM은 이미지-텍스트 쌍이 매칭인지 아닌지를 맞히는 이진 분류(binary classification) 문제다.  
BLIP에서는 **Image-grounded Text Encoder(fusion encoder 모드)** 를 사용해 멀티모달 융합 표현을 만든다.

### 3.2.1 입력/출력

- 입력: 이미지 토큰 + 텍스트 토큰
- 텍스트 토큰은 이미지 토큰에 cross-attention 가능
- 출력: [CLS] 임베딩을 2-class linear classifier에 넣어 match logit 계산

---

### 3.2.2 Hard Negative Mining

ITM에서 negative를 랜덤으로만 만들면 너무 쉬워 학습 신호가 약해진다.  
BLIP은 **ITC 유사도 기반으로 헷갈리는 negative** 를 샘플링한다.

배치에서 이미지 $$I_i$$ 에 대해, 정답 텍스트 $$T_i$$ 를 제외한 텍스트들 $$T_j$$ 중에서 ITC 유사도 $$s_{ij}$$ 가 큰 샘플을 negative 후보로 선택한다.  
텍스트 $$T_i$$ 에 대해서도 동일하게, 정답 이미지 $$I_i$$ 를 제외한 이미지들 $$I_j$$ 중에서 $$s_{ji}$$ 가 큰 샘플을 negative로 선택한다.

이렇게 구성한 양성/음성 쌍에 대해 이진 cross entropy로 학습한다.

---

## 3.3 (III) Language Modeling Loss (LM)

LM은 **Image-grounded Text Decoder 모드**에서 수행한다.

- 텍스트 self-attention은 causal mask로 auto-regressive 생성
- 각 토큰은 이미지 토큰에 cross-attention
- 학습 안정화를 위해 label smoothing을 사용

시퀀스 길이를 $$L$$, vocabulary를 $$\mathcal{V}$$ 라고 할 때,

- $$\mathcal{L}_{LM} = -\sum_{t=1}^{L}\sum_{w \in \mathcal{V}} y^{LS}_{t}(w)\,\log p_t(w)$$

여기서 $$y^{LS}_{t}(w)$$ 는 label smoothing이 적용된 타깃 분포이고, $$p_t(w)$$ 는 t번째 토큰에서 단어 $$w$$ 의 예측 확률이다.

---

## 4. CapFilt: Captioning and Filtering

<div style="text-align: center;">
  <img src="/assets/images/BLIP/CapFilt.png" alt="BLIP CapFilt pipeline" width="100%">
  <br>
  <em>Figure 2. CapFilt 파이프라인: captioner로 캡션을 생성하고, filter로 이미지-텍스트 페어 정합성을 판별해 노이즈를 줄인다.</em>
</div>
<br>

웹에서 수집한 이미지-텍스트 페어는 노이즈가 많다. BLIP은 이를 줄이기 위해 **Captioner + Filter**로 구성된 CapFilt를 제안한다.

### 4.1 Captioner

- 이미지 $$I$$ 를 입력 받아 캡션 $$\hat{T}$$ 를 생성한다.
- 원본 웹 캡션 $$T_{web}$$ 과 별도로 “모델이 생성한 캡션”을 추가로 확보한다.

### 4.2 Filter

- (이미지, 텍스트) 쌍이 의미적으로 맞는지 판별한다.
- 일반적으로 ITM과 유사하게 $$P(\text{match}\mid I,T)$$ 를 출력하는 분류기로 볼 수 있다.

### 4.3 데이터 재구성

1. 원본 웹 페어 $$ (I, T_{web}) $$ 가 주어진다.
2. Captioner로 $$\hat{T}$$ 를 생성한다.
3. Filter에 $$ (I, T_{web}) $$, $$ (I, \hat{T}) $$ 등을 넣어 통과/제거한다.
4. Filter를 통과한 텍스트들과 사람이 만든 주석 데이터를 합쳐 최종 학습 데이터를 구성한다.

---

## 5. Reference

- BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation (Li et al.)
- ALBEF: Align before Fuse (Li et al.)