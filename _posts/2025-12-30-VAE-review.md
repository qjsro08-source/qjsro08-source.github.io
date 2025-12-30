---
title:  "[Paper Review] Auto-Encoding Variational Bayes"
excerpt: "VAE"
categories:
  - Paper Review
tags:
  - Deep Learning
  - Generative Model
  - VAE
toc: true
toc_sticky: true
author_profile: true
mathjax: true
---
단순히 입력을 압축했다가 복원하는 Autoencoder(AE)와 달리, **Variational Autoencoder(VAE)**는 새로운 데이터를 만들어내는 **생성 모델(Generative Model)**임.

이번 글에서는 VAE가 왜 Encoder를 필요로 하는지, **ELBO(Evidence Lower Bound)** 수식의 유도 과정, 그리고 **Reparameterization Trick**이 필요한 이유를 정리함.

---

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## 1. VAE의 목적과 난제

VAE의 목표는 데이터 $$x$$가 주어졌을 때, 이를 잘 설명하는 잠재 변수(Latent Variable) $$z$$를 찾는 것임. 즉, 우리가 알고 싶은 것은 **Posterior 분포 $$p(z|x)$$**임.

$$p(z|x) = \frac{p(x|z)p(z)}{p(x)}$$

하지만 베이즈 정리에 의한 위 식에서 분모인 **Evidence $p(x)$**를 계산하는 것이 문제임.

$$p(x) = \int p(x|z)p(z) dz$$

이 적분은 $$z$$의 차원이 높을수록 모든 경우의 수를 계산하는 것이 불가능(Intractable)함. 따라서 $$p(z|x)$$를 직접 구할 수 없음.

### 해결책: 변분 추론 (Variational Inference)

계산할 수 없는 $$p(z|x)$$ 대신, 이를 가장 잘 근사(Approximate)하는 다루기 쉬운 분포 **$q_{\phi}(z|x)$**를 도입함. 이것이 바로 **Encoder(Inference Model)**의 역할임.

<div style="center">
  <img src="/assets/images/VAE/main.png" alt="VAE Architecture and Reparameterization Trick" width="100%">
  <p><em>그림 1: VAE의 전체 아키텍처와 Reparameterization Trick 도식</em></p>
</div>
<br>

* **Encoder ($q_{\phi}(z|x)$):** 데이터 $$x$$를 보고 $$z$$의 분포(평균 $\mu$, 분산 $$\sigma^2$$)를 예측함.
* **Decoder ($p_{\theta}(x|z)$):** 잠재 변수 $$z$$로부터 데이터 $x$를 복원함.

---

## 2. 목적 함수 유도: ELBO

우리의 최종 목표는 관측 데이터의 우도(Log-Likelihood), 즉 $$\log p(x)$$를 최대화하는 것임. $$p(z|x)$$를 모르는데 어떻게 최대화할 수 있을까? 수식을 전개해보면 다음과 같음.

양변에 $$\int q_{\phi}(z|x) dz = 1$$을 곱해도 값은 변하지 않음.

$$
\begin{aligned}
\log p(x) &= \int q_{\phi}(z|x) \log p(x) dz \\
&= \int q_{\phi}(z|x) \log \left( \frac{p(x, z)}{p(z|x)} \right) dz \quad (\because p(x) = \frac{p(x,z)}{p(z|x)}) \\
&= \int q_{\phi}(z|x) \log \left( \frac{p(x, z)}{q_{\phi}(z|x)} \cdot \frac{q_{\phi}(z|x)}{p(z|x)} \right) dz \\
&= \underbrace{\int q_{\phi}(z|x) \log \left( \frac{p(x, z)}{q_{\phi}(z|x)} \right) dz}_{\text{ELBO}} + \underbrace{\int q_{\phi}(z|x) \log \left( \frac{q_{\phi}(z|x)}{p(z|x)} \right) dz}_{\text{KL}(q_{\phi}(z|x) || p(z|x))}
\end{aligned}
$$

위 식은 두 부분으로 나뉨.
1.  **ELBO (Evidence Lower Bound):** 우리가 최대화해야 할 대상.
2.  **KL Divergence:** $$q(z|x)$$와 $$p(z|x)$$ 사이의 거리. $$p(z|x)$$를 모르니 계산 불가능하지만, KL의 성질에 의해 **항상 0 이상**임.

즉, $$\log p(x) \ge \text{ELBO}$$ 가 성립함. 따라서 **ELBO를 최대화하면, 전체 Log-Likelihood인 $\log p(x)$도 하한(Lower Bound)이 올라가며 자연스럽게 최대화됨.**

### 최종 Loss Function
딥러닝에서는 보통 Minimize 문제를 풀기 때문에, ELBO에 마이너스를 붙인 것을 Loss로 사용함.

$$
\begin{aligned}
\text{Loss} &= - \text{ELBO} \\
&= - \mathbb{E}_{q_{\phi}(z|x)} [\log p_{\theta}(x|z)] + D_{KL}(q_{\phi}(z|x) || p(z))
\end{aligned}
$$

1.  **Reconstruction Loss (첫 번째 항):** Decoder가 $$z$$를 받아 $$x$$를 얼마나 잘 복원했는가? (MSE or BCE)
2.  **Regularization Loss (두 번째 항):** Encoder가 만든 $$z$$의 분포가 사전 분포 $$p(z)$$(보통 $$N(0, I)$$)와 얼마나 유사한가?

> **Q. 왜 $z$를 가우시안 분포로 가정하는가?**
> KL Divergence 항을 닫힌 형태(Closed-form)로 미분 가능하게 계산할 수 있기 때문임. 또한, 잠재 공간을 원점 근처로 모아주어 데이터의 Manifold를 부드럽게(Continuous) 형성하도록 도움.

---

## 3. 미분 불가능 문제의 해결: Reparameterization Trick

Encoder는 입력 $$x$$에 대해 확률 분포의 파라미터인 평균($$\mu$$)과 분산($$\sigma^2$$)을 출력함. 여기서 $$z$$를 하나 뽑아야(Sampling) Decoder에 넣을 수 있음.

$$z \sim N(\mu, \sigma^2)$$

하지만 **"샘플링(Sampling)"이라는 연산은 미분이 불가능함.** 즉, Backpropagation을 할 때 그래디언트가 Encoder로 흐르지 못해 학습이 불가능해짐.

이를 해결하기 위해 **Reparameterization Trick**을 사용함. 위 도식(그림 1)의 중앙 부분과 같이, 무작위성(Randomness)을 외부 변수 $\epsilon$으로 분리하는 기법임.

$$
\begin{aligned}
\epsilon &\sim N(0, I) \\
z &= \mu + \sigma \odot \epsilon
\end{aligned}
$$

* **$$\mu, \sigma$$:** Encoder가 출력한 값 (Deterministic, **미분 가능**)
* **$$\epsilon$$:** 외부에서 주입된 노이즈 (Stochastic, 미분 불필요)

이제 $$z$$는 $$\mu$$와 $$\sigma$$에 대한 함수가 되므로, 역전파를 통해 Encoder의 파라미터를 정상적으로 업데이트할 수 있게 됨.

---

### 요약

1.  **Encoder의 역할:** Intractable한 $$p(z|x)$$를 근사하는 $$q(z|x)$$를 모델링함.
2.  **학습 원리:** $$\log p(x)$$를 직접 최대화할 수 없으니, 하한선인 **ELBO**를 최대화함.
3.  **구현 핵심:** 샘플링 과정의 미분 불가능 문제를 해결하기 위해 **Reparameterization Trick** ($$z = \mu + \sigma \odot \epsilon$$)을 사용함.