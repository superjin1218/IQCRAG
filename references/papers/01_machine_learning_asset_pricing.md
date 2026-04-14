# 머신러닝·딥러닝 기반 자산가격결정

비선형 예측, 차원축소, no-arbitrage 제약 학습 등 ML/DL 기법을 자산가격결정에 접목한 논문들.

| # | 제목 | 저자 | 연도 | 핵심 기여 | 링크 |
|---|---|---|---:|---|---|
| 1 | **Empirical Asset Pricing via Machine Learning** | Shihao Gu, Bryan Kelly, Dacheng Xiu | 2020 | 자산가격결정 예측 문제에서 다양한 ML 기법을 비교하며 **비선형 상호작용이 예측력·경제적 성과에 기여**할 수 있음을 분석. 트리·신경망 계열이 전통 선형모형을 크게 능가. | https://dachxiu.chicagobooth.edu/download/ML.pdf |
| 2 | **Autoencoder Asset Pricing Models** | Shihao Gu, Bryan Kelly, Dacheng Xiu | 2021 | **조건부 오토인코더**로 특성(characteristics) 기반 비선형 요인/익스포저를 학습. 차원축소 + 비선형을 결합해 전통 선형 요인모형 대비 out-of-sample 성능 개선. | https://www.sciencedirect.com/science/article/abs/pii/S0304407620301998 |
| 3 | **Deep Learning in Asset Pricing** | Luyang Chen, Markus Pelger, Jason Zhu | 2019 | **No-arbitrage 조건을 제약으로 반영한 딥러닝** 자산가격결정 접근(working paper). SDF를 GAN 스타일로 학습하며 이후 저널 버전 존재. | https://arxiv.org/abs/1904.00745 |

## 시사점 (WorldQuant BRAIN 전략 설계 관점)

- 단일 선형 z-score 결합보다 **다중 특성·비선형 조합**이 정보를 더 뽑아낼 수 있음 — 단, 플랫폼 외부 리서치/검증에서 사용해야 함.
- `rank` / `ts_zscore` 후 다중 특성을 결합하는 알파는 오토인코더식 "축소된 요인" 근사로 해석 가능.
- No-arbitrage 제약은 펀더멘털·리스크 팩터 결합 시 롱숏 중립성(`group_neutralize`, dollar-neutral) 유지로 경험적 투영 가능.
