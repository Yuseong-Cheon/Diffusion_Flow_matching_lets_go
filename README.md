# Generative Model Explainer Studio

브라우저에서 **Flow Matching의 조건부 직선 경로**와 **Diffusion reverse SDE의 확률적 경로**를 나란히 보는 2D 교육용 시각화입니다.

> 이 프로젝트는 실제 이미지 생성 모델이나 학습 코드를 포함하지 않습니다. 고양이·강아지 군집은 두 방법의 경로 차이를 직관적으로 보여주기 위한 Gaussian-mixture toy distribution입니다.

왼쪽 카드는 `R,G,B ∼ N(0,1)`인 Gaussian RGB 샘플입니다. 중앙 2D 경로는 같은 값의 직교 투영 `x=(R−G)/√2`, `y=(R+G−2B)/√6`을 사용합니다.

## 화면에서 비교하는 것

### Flow Matching

선택한 노이즈–데이터 한 쌍에 대해

```text
xτ = (1 − τ)z₀ + τx₁
uτ = x₁ − z₀
```

를 표시합니다. `uτ`가 일정한 것은 이 **조건부 선형 경로 한 쌍**에 대한 설명이며, 학습된 주변 속도장 `vθ(x, τ)` 전체가 항상 일정하다는 뜻은 아닙니다.

### Diffusion

20개 군집으로 구성된 2D Gaussian mixture의 analytic score를 계산하고, VP SDE의 역과정을 Euler–Maruyama 방식으로 적분합니다.

```text
dXτ = [−f(Xτ,s) + g(s)² ∇log pₛ(Xτ)]dτ + g(s)dW
s = 1 − τ
```

따라서 경로 변화와 mode 선택은 임의 품종 전환이 아니라 현재 위치의 score와 Gaussian Brownian increment에서 나옵니다.

## 포함하지 않는 것

- 신경망 학습, U-Net, 실제 이미지 데이터셋
- DDPM/CFM training loss 실행
- 실제 이미지 모델의 likelihood 또는 생성 확률
- Optimal Transport coupling 및 reflow

논문 학습용 기준:

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)

## 실행

```bash
python -m http.server 8080
```

브라우저에서 `http://localhost:8080`을 엽니다.
