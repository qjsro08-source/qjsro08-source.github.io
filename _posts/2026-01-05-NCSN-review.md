---
title:  "[Paper Review] Generative Modeling by Estimating Gradients of the Data Distribution (NCSN)"
excerpt: "NCSN."
categories:
  - Generative Model
tags:
  - Deep Learning
  - Generative Model
  - NCSN
toc: true
toc_sticky: true
author_profile: true
mathjax: true
---

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

# Generative Modeling by Estimating Gradients of the Data Distribution (NCSN) 정리

## 기본 용어

- **IID (Independent and Identically Distributed)**  
  데이터 샘플 $$\{x_i\}_{i=1}^n$$ 이 서로 독립이고(Independent), 동일한 분포 $$p_{\text{data}}(x)$$에서 샘플링(Identically Distributed)되었다는 가정.

- **Divergence (발산)**  
  벡터장 $$s(x)=(s_1(x),\dots,s_D(x))$$에 대해  
  $$\mathrm{div}\, s(x)=\sum_{i=1}^D \frac{\partial s_i(x)}{\partial x_i}.$$

- **Jacobian (야코비안)**  
  벡터값 함수 $$s:\mathbb R^D\to\mathbb R^D$$의 편미분 행렬  
  $$J_s(x)=\frac{\partial s(x)}{\partial x^\top},\qquad (J_s)_{ij}=\frac{\partial s_i(x)}{\partial x_j}.$$
  그리고  
  $$\mathrm{tr}(J_s(x))=\mathrm{div}\, s(x).$$

---

## Score-based Generative Model

**Score**는 데이터 분포의 로그밀도 기울기:  
$$\nabla_x \log p(x)$$  
이다. 직관적으로 어떤 위치 $$x$$에서 **log-density가 증가하는 방향**을 가리키므로, 이 방향으로 움직이면 “더 그럴듯한(확률이 높은)” 영역으로 가게 된다.

목표는 데이터 $$x\sim p_{\text{data}}(x)$$가 주어졌을 때 score를 근사하는 모델 $$s_\theta(x)$$를 학습하는 것:  
$$s_\theta(x)\approx \nabla_x \log p_{\text{data}}(x).$$

---

## Score Matching (Hyvärinen)

이상적으로는 다음 MSE를 최소화하고 싶다:  
$$\frac12\mathbb E_{p_{\text{data}}}\left[\left\|s_\theta(x)-\nabla_x\log p_{\text{data}}(x)\right\|_2^2\right].$$  
하지만 $$\nabla_x\log p_{\text{data}}(x)$$는 알 수 없으므로 그대로는 최적화가 어렵다.

이를 전개하면  
$$\frac12\mathbb E\big[\|s_\theta(x)\|^2\big]
+\frac12\mathbb E\big[\|\nabla_x\log p(x)\|^2\big]
-\mathbb E\big[s_\theta(x)^\top\nabla_x\log p(x)\big].$$  
여기서 두 번째 항은 $$\theta$$와 무관한 상수이므로 무시 가능하다. 문제는 마지막 교차항인데,

$$\mathbb E_{p}\big[s_\theta(x)^\top\nabla_x\log p(x)\big]
=\int p(x)\, s_\theta(x)^\top \frac{\nabla_x p(x)}{p(x)}dx
=\int s_\theta(x)^\top \nabla_x p(x)\,dx.$$

적절한 경계 조건(무한 원에서 경계항이 0)이 성립한다고 하면 부분적분으로  
$$\int s_\theta(x)^\top \nabla_x p(x)\,dx
= -\int p(x)\,\mathrm{div}\, s_\theta(x)\,dx.$$

따라서 원래 목적함수는 (상수항 제외) 다음과 동치:  
$$\mathcal L_{\text{SM}}(\theta)
=\mathbb E_{p_{\text{data}}}\left[\frac12\|s_\theta(x)\|_2^2+\mathrm{div}\, s_\theta(x)\right]
=\mathbb E_{p_{\text{data}}}\left[\frac12\|s_\theta(x)\|_2^2+\mathrm{tr}\big(\nabla_x s_\theta(x)\big)\right].$$

> **중요:** 여기서 trace/divergence는 **$$\theta$$**가 아니라 **입력 x**에 대한 미분이다.

---

## 왜 노이즈를 섞나? (Denoising Score Matching)

실제 데이터는 고차원 공간에서 “얇은 manifold 근처”에 몰려 있는 경우가 많다.  
이때 $$p_{\text{data}}(x)$$의 support 밖에서는 score가 불안정/정의 곤란해질 수 있다.

그래서 원본 데이터에 가우시안 노이즈를 섞은 perturbed 샘플  
$$\tilde x = x+\sigma\epsilon,\qquad \epsilon\sim\mathcal N(0,I)$$  
을 사용한다. 조건부분포는  
$$q_\sigma(\tilde x\mid x)=\mathcal N(\tilde x;\,x,\sigma^2 I).$$

이때 가우시안의 score는 닫힌형태로 계산 가능:  
$$\nabla_{\tilde x}\log q_\sigma(\tilde x\mid x)=\frac{x-\tilde x}{\sigma^2}.$$

### **(추가) 계산 과정:** $$\nabla_{\tilde x}\log q_\sigma(\tilde x\mid x)=\frac{x-\tilde x}{\sigma^2}$$

조건부분포  
$$q_\sigma(\tilde x\mid x)=\mathcal N(\tilde x;\,x,\sigma^2 I)$$  
의 밀도함수는  
$$q_\sigma(\tilde x\mid x)
=\frac{1}{(2\pi\sigma^2)^{D/2}}
\exp\!\left(-\frac{1}{2\sigma^2}\|\tilde x-x\|_2^2\right).$$

로그를 취하면  
$$\log q_\sigma(\tilde x\mid x)
= -\frac{D}{2}\log(2\pi\sigma^2)\;-\;\frac{1}{2\sigma^2}\|\tilde x-x\|_2^2.$$

여기서 $$\tilde x$$에 대한 gradient를 구할 때, 첫 항 $$-\frac{D}{2}\log(2\pi\sigma^2)$$는 $$\tilde x$$와 무관한 상수라서 미분하면 0이다.  
따라서  
$$\nabla_{\tilde x}\log q_\sigma(\tilde x\mid x)
= \nabla_{\tilde x}\left(-\frac{1}{2\sigma^2}\|\tilde x-x\|_2^2\right).$$

노름 제곱을 전개하면  
$$\|\tilde x-x\|_2^2=(\tilde x-x)^\top(\tilde x-x).$$  
그리고 표준 미분 결과로  
$$\nabla_{\tilde x}\,(\tilde x-x)^\top(\tilde x-x)=2(\tilde x-x).$$

이를 대입하면  
$$\nabla_{\tilde x}\log q_\sigma(\tilde x\mid x)
= -\frac{1}{2\sigma^2}\cdot 2(\tilde x-x)
= -\frac{\tilde x-x}{\sigma^2}
= \frac{x-\tilde x}{\sigma^2}.$$

(성분별로 보면, 각 $$i$$에 대해 $$\frac{\partial}{\partial \tilde x_i}\log q = -\frac{1}{\sigma^2}(\tilde x_i-x_i)$$ 이므로 벡터로 모으면 동일하게 $$\frac{x-\tilde x}{\sigma^2}$$가 된다.)

따라서 denoising score matching 목적함수는  
$$\mathcal L_{\text{DSM}}(\theta)
=\frac12\,\mathbb E_{x\sim p_{\text{data}},\,\tilde x\sim q_\sigma(\cdot\mid x)}
\left[\left\|s_\theta(\tilde x)-\frac{x-\tilde x}{\sigma^2}\right\|_2^2\right].$$  
이제 타겟(우변)이 **모두 계산 가능**해지므로 학습이 가능하다.

---

## Multi-level Noise (NCSN의 핵심 트릭)

$$\sigma$$를 하나만 쓰지 않고  
$$\sigma_1>\sigma_2>\cdots>\sigma_L$$  
처럼 여러 노이즈 레벨을 둔다(annealing).

- **큰 $$\sigma$$**: 분포가 많이 스무딩 → 전역 구조 학습/샘플링이 쉬움  
- **작은 $$\sigma$$**: 세밀한 디테일 복원

실제로는 하나의 네트워크가 $$\sigma$$를 조건으로 받아  
$$s_\theta(x,\sigma_i)$$  
를 출력하도록 학습한다.

---

## Sampling: Annealed Langevin Dynamics

학습된 score를 이용해 Langevin dynamics로 샘플링한다. (**노이즈 레벨을 큰 것에서 작은 것으로 내려가며 반복**)

각 $$\sigma_i$$에서 $$K$$번 반복:  
$$x_{t+1}=x_t+\frac{\alpha_i}{2}s_\theta(x_t,\sigma_i)+\sqrt{\alpha_i}\,z_t,
\qquad z_t\sim\mathcal N(0,I).$$

그리고 $$\sigma_1\to\sigma_2\to\cdots\to\sigma_L$$ 순서로 진행하면, 거친 형태에서 시작해 점점 디테일을 채워가며 최종 샘플을 얻는다.
