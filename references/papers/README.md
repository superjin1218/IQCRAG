# 퀀트 리서치 논문 인덱스

출처: `backup_20260321/deep-research-report (1).md` 의 "논문 15편" 테이블을 주제별로 분할.

총 15편을 6개 카테고리로 분류했습니다.

| 파일 | 카테고리 | 편수 |
|---|---|---:|
| [01_machine_learning_asset_pricing.md](01_machine_learning_asset_pricing.md) | 머신러닝/딥러닝 기반 자산가격결정 | 3 |
| [02_volatility_options.md](02_volatility_options.md) | 변동성·옵션 | 2 |
| [03_momentum_trend.md](03_momentum_trend.md) | 모멘텀·추세추종 | 3 |
| [04_factor_models_and_style.md](04_factor_models_and_style.md) | 팩터모델·스타일(퀄리티 등)·고차원 횡단면 | 4 |
| [05_text_nlp.md](05_text_nlp.md) | 텍스트·NLP 기반 예측 | 1 |
| [06_methodology_replication.md](06_methodology_replication.md) | 다중검정·재현성·방법론 | 2 |

## 카테고리 분류 기준

- **ML/DL**: 비선형 예측 모델, 오토인코더, 딥러닝 no-arbitrage 등 구조적 학습 기법
- **변동성·옵션**: 변동성 타겟팅, IV 기반 리스크 지표 (NVIX 포함)
- **모멘텀·추세**: time-series momentum, factor momentum, 모멘텀 크래시
- **팩터모델·스타일**: Quality, 고차원 shrinkage, factor zoo 검정, IPCA 계열
- **텍스트·NLP**: 뉴스 텍스트 감성 기반 수익률 예측
- **방법론**: 다중검정 허들, 이상현상 재현 연구

## 주의

원문 보고서에는 이 논문들의 성과 수치를 그대로 차용한 전략의 Sharpe/수익률이 "모의치(설계 목표 가정치)"로만 제시돼 있습니다. 실제 채택 시에는 플랫폼 시뮬레이션 + 워크포워드 + 다중가설 통제로 검증이 필요합니다.
