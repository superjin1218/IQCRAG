# IQC 알파 전략 분석 리포트

> **분석 대상**: IQC Sample Expression (2025.04.24 기준, 46개 알파)
> **공통 설정**: Universe TOP3000 · Pasteurization ON · NaN Handling OFF · Truncation 0.08 · Delay 1

---

## 1. 성과 요약 (Top Performers)

### Sharpe 기준 Top 10

| ID | Sharpe | Margin(bp) | Decay | Neutralize | 핵심 시그널 |
|----|--------|------------|-------|------------|-----------|
| 31 | 2.01 | 10.09 | 7 | I | `scl12_sentvec` `nws12_prez_120_min` `pv13_page_rank` |
| 44 | 1.98 | 9.33 | 10 | I | `scl12_buzz` `anl4_fcf_flag` `returns` |
| 5  | 1.94 | 5.98 | 5 | I | `anl4_bvps_flag` `dividend` `pv13_custretsig` |
| 6  | 1.84 | 9.30 | 5 | I | `anl4_totassets_flag` `nws12_prez_result2` `IV_skew_150` |
| 16 | 1.84 | 6.02 | 10 | SUB | `snt_social_value` `returns` `IV_skew_60` `IV_skew_150` |
| 14 | 1.79 | 8.12 | 6 | S | `scl12_buzz` `returns` `IV_skew_180` |
| 25 | 1.78 | 6.98 | 2 | M | `scl12_buzz` `anl4_dez1afv4_est` `anl4_fcf_flag` |
| 28 | 1.77 | 6.84 | 7 | SUB | `scl12_buzzvec` `pv13_revere_level` `IV_skew_150` |
| 9  | 1.76 | 8.37 | 7 | S | `nws12_prez_result2` `IV_skew_150` `returns` |
| 4  | 1.74 | 11.87 | 7 | I | `anl4_bvps_flag` `IV_skew_180` `snt_buzz` `anl4_totassets` |

### Margin 기준 Top 10

| ID | Margin(bp) | Sharpe | Decay | Neutralize | 핵심 시그널 |
|----|------------|--------|-------|------------|-----------|
| 33 | 45.07 | 1.28 | 0 | M | `snt_buzz` `anl4_netprofit_flag` `nws12_prez_prevday` |
| 35 | 19.51 | 1.30 | 8 | M | `scl12_buzz` `anl4_netprofit_flag` `scl12_sentvec` |
| 38 | 18.37 | 1.30 | 6 | I | `scl12_buzz` `IV_skew_60` `IV_skew_270` `anl4_netprofit` |
| 34 | 16.66 | 1.54 | 6 | M | `snt_social_value` `returns` `scl12_buzz` `anl4_ptpr` |
| 42 | 14.34 | 1.49 | 4 | M | `scl12_sentvec` `dividend` `anl4_netprofit_flag` |
| 36 | 13.69 | 1.65 | 5 | I | `scl12_buzz` `dividend` `anl4_ptpr_number` |
| 37 | 13.49 | 1.53 | 10 | S | `scl12_sentvec` `anl4_bvps_flag` `snt_buzz` |
| 1  | 12.03 | 1.61 | 6 | S | `anl4_totassets_flag` `anl4_dez1afv4_est` `snt_buzz` |
| 45 | 12.12 | 1.48 | 5 | M | `scl12_buzz` `anl4_netprofit_flag` `returns` |
| 2  | 11.72 | 1.63 | 10 | I | `anl4_totassets_flag` `scl12_buzz` `nws12_prez_result2` |

### 핵심 관찰

- **Sharpe가 높은 알파는 Margin이 상대적으로 낮고, Margin이 높은 알파는 Sharpe가 낮다.** 이는 고수익 알파일수록 변동성(리스크)이 크다는 것을 의미한다.
- Sharpe ≥ 1.8인 알파 6개 중 5개가 **Industry(I) 또는 SubIndustry(SUB) neutralize**를 사용한다.
- Margin ≥ 15bp인 알파 4개 중 3개가 **Market(M) neutralize**를 사용한다.

---

## 2. 시그널(팩터) 사용 빈도 분석

### 가장 자주 사용된 데이터 필드

| 시그널 | 등장 횟수 | 카테고리 | 역할 |
|--------|----------|---------|------|
| `scl12_buzz` | **36회** | 소셜/뉴스 센티먼트 | 소셜 미디어 버즈 지표. 대부분 **음(-)** 방향으로 사용 → 과열 역발상 |
| `snt_buzz` | 18회 | 센티먼트 | 센티먼트 버즈. max() 또는 ts_backfill()로 활용 |
| `returns` | 16회 | 가격 | 주가 수익률. 주로 **음(-)** 방향 → 단기 반전(mean reversion) |
| `implied_volatility_mean_skew_*` | 30회+ | 옵션/변동성 | IV 스큐 (60/120/150/180/270/720일). 핵심 알파 드라이버 |
| `anl4_totassets_flag` | 10회 | 펀더멘탈 | 총자산 관련 플래그 |
| `anl4_bvps_flag` | 7회 | 펀더멘탈 | 장부가치(BPS) 관련 플래그 |
| `anl4_netprofit_flag` | 8회 | 펀더멘탈 | 순이익 관련 플래그 |
| `anl4_fcf_flag` | 4회 | 펀더멘탈 | 잉여현금흐름 플래그 |
| `nws12_prez_result2` | 6회 | 뉴스 | 뉴스 프레지던트 결과 지표 |
| `pv13_custretsig_retsig` | 7회 | 가격 | 커스텀 수익률 시그널 |
| `rp_ess_earnings` | 3회 | ESG/리스크 | ESG 어닝 리스크 프리미엄 |
| `scl12_alltype_buzzvec` | 8회 | 소셜 | 전체 유형 버즈 벡터 |
| `scl12_alltype_sentvec` | 5회 | 소셜 | 전체 유형 센티먼트 벡터 |
| `dividend` | 5회 | 펀더멘탈 | 배당 데이터 |

### 시그널 카테고리 분포

```
센티먼트/소셜 (scl12, snt)  ██████████████████████  ~35%
옵션/변동성 (IV skew)       ████████████████       ~25%
펀더멘탈 (anl4)             ██████████████         ~22%
가격/수익률 (returns, pv13) ████████               ~12%
뉴스 (nws12)               ████                   ~6%
```

**→ 센티먼트 + IV 스큐가 전체 시그널의 60%를 차지. 이 둘이 핵심 알파 엔진이다.**

---

## 3. 오퍼레이터(함수) 사용 패턴

### 빈도순 오퍼레이터 정리

| 오퍼레이터 | 빈도 | 용도 |
|-----------|------|------|
| `add()` | **46/46** (100%) | 모든 알파의 최외곽 래퍼. 여러 시그널을 합산 |
| `ts_backfill()` | 거의 모든 알파 | 결측값 보간. 이중 중첩 `ts_backfill(ts_backfill(...))` 패턴 빈출 |
| `zscore()` | ~80% | 표준화. 시그널 스케일 통일의 핵심 |
| `rank()` | ~40% | 순위 변환. 비선형 매핑으로 아웃라이어 제거 |
| `ts_mean()` | ~35% | 시계열 평균. 노이즈 감소 |
| `max() / min()` | ~25% | 조건부 시그널 선택 |
| `ts_zscore()` | ~15% | 시계열 기반 z-score |
| `ts_av_diff()` | ~15% | 시계열 평균 차이. 모멘텀/반전 탐지 |
| `ts_delta()` | ~15% | 시계열 변화량 |
| `vec_avg()` | ~40% | 벡터 평균. 다차원 데이터 압축 |
| `ts_rank()` | ~10% | 시계열 순위 |
| `ts_covariance()` | ~8% | 시계열 공분산 |
| `signed_power()` | ~6% | 부호 유지 거듭제곱. 비선형 증폭 |
| `log()` | ~4% | 로그 변환 |
| `ts_scale()` | ~4% | 시계열 스케일링 |
| `ts_corr()` | ~2% | 시계열 상관관계 |
| `ts_std_dev()` | ~2% | 시계열 표준편차 |
| `filter=True` | 100% | 필터링 항상 활성 |

### 핵심 패턴: 이중 ts_backfill

거의 모든 알파에서 `ts_backfill(-ts_backfill(X, n), n)` 패턴이 반복된다. 이는 결측값을 먼저 채운 후 부호를 반전시키고, 다시 한 번 보간하는 방식이다. 이 패턴은 **데이터 안정성 확보 + 역방향 시그널 생성**의 표준 관용구로 보인다.

---

## 4. Decay 파라미터 분석

| Decay 값 | 알파 수 | 평균 Sharpe | 평균 Margin(bp) |
|----------|--------|-----------|---------------|
| 0 | 1 | 1.28 | 45.07 |
| 2 | 4 | 1.65 | 9.03 |
| 4 | 1 | 1.49 | 14.34 |
| 5 | 7 | 1.74 | 8.53 |
| 6 | 10 | 1.64 | 9.73 |
| 7 | 8 | 1.70 | 9.62 |
| 8 | 5 | 1.59 | 11.08 |
| 9 | 4 | 1.52 | 9.91 |
| 10 | 10 | 1.62 | 9.90 |

### 해석

- **Decay 5~7이 Sharpe 최적 구간이다** (평균 1.70~1.74). 시그널의 반감기가 1주일 내외가 가장 효과적임을 시사한다.
- Decay 0 (id 33)은 극단적 Margin(45bp)을 보이지만 Sharpe 1.28로 리스크 대비 효율이 낮다. 빠른 턴오버 전략.
- Decay가 높을수록(8~10) 안정성이 떨어지는 경향.

---

## 5. Neutralize 방식 분석

| Neutralize | 알파 수 | 평균 Sharpe | 평균 Margin(bp) |
|------------|--------|-----------|---------------|
| M (Market) | 12 | 1.55 | 14.56 |
| S (Sector) | 13 | 1.60 | 10.10 |
| I (Industry) | 17 | 1.67 | 10.04 |
| SUB (SubIndustry) | 4 | 1.68 | 7.10 |

### 해석

- **세분화된 중립화(I, SUB)일수록 Sharpe가 높다.** Industry/SubIndustry 중립화가 cross-sectional 노이즈를 더 잘 제거한다.
- **Market 중립화는 Margin이 가장 높지만 Sharpe가 가장 낮다.** 큰 베팅을 하되 리스크도 큰 구조.
- 1b)에서 언급된 "10 year sharpe & margin" 개선 여부를 고려하면, **I neutralize + Decay 5~7 조합**이 가장 안정적으로 높은 점수를 얻을 가능성이 높다.

---

## 6. 반복되는 알파 구성 공식 (Blueprint)

대부분의 고성과 알파는 다음 4가지 컴포넌트를 `add()`로 합산하는 구조이다:

```
add(
    [A] 센티먼트 역발상 시그널,
    [B] 옵션 IV 스큐 시그널,
    [C] 펀더멘탈 품질 시그널,
    [D] 가격 반전 또는 뉴스 시그널,
    filter=True
)
```

### [A] 센티먼트 역발상 (Contrarian Sentiment)

가장 빈번한 패턴:
```
zscore(-ts_backfill(scl12_buzz, 25))           # 소셜 버즈 역방향
ts_mean(-ts_backfill(scl12_buzz, 25), 25)      # 소셜 버즈 평균의 역방향
max(-ts_backfill(scl12_buzz, 25), snt_buzz)    # 소셜 버즈 역방향 vs 센티먼트 중 큰 값
ts_backfill(-ts_backfill(scl12_buzz, 25), 25)  # 이중 백필 역방향
```

**핵심 인사이트**: scl12_buzz는 거의 항상 **음(-)** 부호로 사용된다. 소셜 미디어에서 관심이 과열된 종목을 피하거나 숏하는 역발상 전략이 IQC에서 가장 강력한 단일 팩터다.

### [B] IV 스큐 (Implied Volatility Skew)

```
rank(ts_backfill(implied_volatility_mean_skew_180, 3))     # 180일 스큐의 순위
zscore(ts_backfill(implied_volatility_mean_skew_150, 43))  # 150일 스큐 표준화
ts_mean(ts_backfill(implied_volatility_mean_skew_270, 57), 57)  # 270일 스큐 평균
```

**핵심 인사이트**: IV 스큐는 옵션 시장의 비대칭 기대를 포착한다. 여러 기간(60/120/150/180/270/720일)의 스큐를 사용하되, **150일과 180일 스큐가 가장 인기**가 많다. 주로 양(+) 방향으로 사용 → 스큐가 높은(풋옵션 프리미엄이 높은) 종목을 선호.

### [C] 펀더멘탈 품질 (Fundamental Quality)

```
zscore(ts_backfill(anl4_totassets_flag, 3))     # 총자산 플래그
zscore(ts_backfill(anl4_bvps_flag, 26))         # 장부가치 플래그
ts_mean(ts_backfill(anl4_netprofit_flag, 500), 500)  # 순이익 (장기 평균)
log(ts_backfill(anl4_fcf_flag, 500))            # 잉여현금흐름 (로그)
signed_power(ts_backfill(anl4_ptpr_number, 8), 2)    # 목표가 수 (비선형 증폭)
```

**핵심 인사이트**: 펀더멘탈 시그널은 주로 **양(+) 방향**. 재무 건전성이 좋은 종목에 롱 포지션. 장기(500일) 백필을 사용하여 안정적인 펀더멘탈 시그널을 확보한다.

### [D] 가격 반전 / 뉴스 (Price Reversal / News)

```
zscore(-ts_backfill(returns, 155))              # 155일 수익률 역방향 (중기 반전)
zscore(-ts_backfill(returns, 3))                # 3일 수익률 역방향 (단기 반전)
ts_av_diff(-ts_backfill(pv13_custretsig_retsig, 272), 272)  # 커스텀 리턴 변화
rank(ts_backfill(vec_avg(nws12_prez_result2), 60))          # 뉴스 결과 순위
```

**핵심 인사이트**: returns는 155일과 3일이 가장 자주 사용되며 둘 다 **음(-)** 방향이다. 중기(~6개월) 또는 초단기(3일) 하락 후 반등을 포착하는 **mean reversion** 전략.

---

## 7. 최적 전략 조합 가이드라인

### Do's (권장)

1. **4개 컴포넌트 합산 구조를 유지하라**
   - 모든 알파가 `add(A, B, C, D, filter=True)` 형태
   - 4개 시그널은 각각 다른 카테고리에서 가져와 분산 효과 확보

2. **scl12_buzz를 역방향(-)으로 반드시 포함하라**
   - 46개 알파 중 36개가 이 시그널 사용
   - 가장 일관된 알파 소스

3. **IV 스큐를 150일 또는 180일 기준으로 포함하라**
   - 양(+) 방향, rank() 또는 zscore()로 래핑

4. **Decay를 5~7로 설정하라**
   - Sharpe 최적 구간

5. **Industry(I) neutralize를 기본으로 사용하라**
   - 평균 Sharpe 1.67로 가장 안정적
   - Sharpe와 Margin의 균형이 가장 좋음

6. **ts_backfill()을 모든 원시 데이터에 적용하라**
   - 결측값 처리가 안정적 백테스트의 기본

7. **zscore()와 rank()를 적절히 혼용하라**
   - zscore: 정규분포 가정이 가능한 시그널
   - rank: 분포가 불규칙한 시그널 (아웃라이어 방지)

### Don'ts (비권장)


2. **Decay 0은 피하라**
   - Margin은 높을 수 있으나 Sharpe가 급격히 낮아짐
   - 턴오버 70% 초과 위험

3. **Market(M) neutralize에만 의존하지 마라**
   - Margin은 크지만 리스크도 크고, 많은 참가자가 비슷한 접근을 사용할 가능성

4. **returns를 양(+) 방향으로 사용하지 마라**
   - 이 데이터셋에서 모멘텀(양방향)보다 반전(음방향)이 압도적으로 우세

---

## 8. 제출 전략 제안

### 포트폴리오 구성 (25+ 제출 필요)

문서 주요 공지에 따라 **25번 이상 시뮬레이션으로 10,000점**을 채워야 한다. 다음 전략으로 제출을 분배하는 것을 추천:

| 유형 | 비율 | Neutralize | Decay 범위 | 목표 |
|------|------|-----------|-----------|------|
| 고Sharpe 안정형 | 50% | I / SUB | 5~7 | 점수 안정 확보 |
| 고Margin 공격형 | 25% | M / S | 2~6 | 점수 상한 돌파 |
| 실험형 (새 조합) | 25% | 혼합 | 다양 | 새 시그널 탐색 |

### 일일 제출 규칙

- 하루 1~2개 제출 (문서 권장 사항)
- 같은 팀원과 시그널 중복 확인 필수 (self correlation 체크)
- 턴오버 1~70% 범위 확인 (초과 시 decay 조정)

---

## 9. 요약: "잘 먹히는 전략"의 공통점

```
1. 센티먼트 역발상     →  소셜 버즈가 높으면 피하고, 낮으면 매수
2. IV 스큐 팩터        →  풋옵션 프리미엄이 높은 종목 선호 (꼬리 위험 보상)
3. 펀더멘탈 품질       →  재무 건전성이 확인된 종목에 롱 포지션
4. 가격 반전           →  중기(155일) 또는 단기(3일) 하락 후 반등 포착
5. 적절한 중립화       →  Industry 수준에서 cross-sectional 노이즈 제거
6. 중간 Decay          →  5~7일의 시그널 반감기가 최적
```

이 6가지 요소를 갖춘 알파가 IQC에서 가장 높은 리스크 조정 수익률을 기록한다.
