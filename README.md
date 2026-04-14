# IQCRAG — Field Graph 파이프라인

WorldQuant BRAIN / IQC 알파 연구를 위해 **필드(데이터 시그널)를 대규모로 시뮬레이션하고, 필드 간 상관관계를 계산해 지식 그래프를 구축하는 엔드투엔드 파이프라인**입니다. 시각화 산출물은 [superjin1218/IQC-SITE](https://github.com/superjin1218/IQC-SITE) 에 정적 사이트로 배포되어 있으며, 본 저장소에는 그 데이터를 생성하기 위해 사용된 **모든 스크립트, 중간 산출물, 원본 데이터, 설정 파일**이 포함되어 있습니다.

---

## 목차

1. [프로젝트 배경](#프로젝트-배경)
2. [전체 파이프라인 개요](#전체-파이프라인-개요)
3. [단계별 상세 설명](#단계별-상세-설명)
4. [핵심 수식 · 알고리즘](#핵심-수식--알고리즘)
5. [데이터 파일 명세](#데이터-파일-명세)
6. [설정 (config.yaml)](#설정-configyaml)
7. [실행 방법](#실행-방법)
8. [결과 요약](#결과-요약)
9. [폴더 구조](#폴더-구조)

---

## 프로젝트 배경

WorldQuant BRAIN 플랫폼에는 수천 개의 데이터 필드(가격, 펀더멘탈, 분석가 추정, 옵션, 뉴스 등)가 있고, 이를 조합해 알파(예측 시그널)를 만듭니다. 문제는:

- 필드 개수가 너무 많아 **탐색 공간이 지수적**
- 서로 다른 필드여도 **실제 PnL 상관이 높은 경우**가 많아 중복 알파가 대량 발생
- 이미 낸 알파들과 **상관 낮은 새 알파**를 찾기 어려워짐 (IQC 기준 통과 실패)

이 프로젝트는 **1,008개의 유용한 필드**를 전수 시뮬레이션해서, 필드들끼리의 **실제 PnL 상관 매트릭스**를 계산하고, 이를 바탕으로:

1. 행동이 유사한 필드들을 **클러스터링** → 대표 필드만 탐색
2. 필드 간 **그래프** 구축 → 시각적 탐색
3. 주어진 임계값 기준으로 **그룹 분류** → 각 그룹의 fitness 최상위 필드만 추출 (다양화된 대표 알파 묶음)
4. 제출할 알파와 상관 낮은 **diversifier** 필드 자동 추천

---

## 전체 파이프라인 개요

```
  references/wq_brain_all_fields.csv (2,094 fields)
        │
        ▼
  [01] Corpus 빌드 ──────────── 1,008 kept / 1,086 dropped
        │ 필드 ID + description + dataset + category 를 하나의 텍스트로
        ▼
  [02] Embeddings 생성 ──────── 854 × dim 1024 (BAAI/bge-m3)
        │ 필드 설명을 임베딩 → ChromaDB 저장
        ▼
  [03] Cluster (HDBSCAN) ────── 97 clusters / 231 noise
        │ 임베딩 거리로 의미 기반 1차 클러스터링
        ▼
  [04] Representatives 선정 ──── 97개 (centroid 근접도 0.7 + log(alphaCount) 0.3)
        │
        ▼
  [05b] Single-field probe ───── 1,008 개의 "rank(ts_backfill(X, 20))" 알파 수식 생성
        │
        ▼
  [06] Step 4 가정 검증 ──────── intra=0.73, inter=0.80 → INTER_HIGH 통과
        │ "임베딩 유사도 ≈ 실제 상관" 가정 검증
        ▼
  [07b] Main Simulation ──────── 1,008/1,008 완료 (pass=993, fail=15)
        │ 각 필드 단일 probe 알파를 WQB 에 시뮬하고 daily-pnl 수집
        │ 3-shard 병렬 처리 (shard 0/1/2)
        ▼
  [08a] Similarity Matrix ────── (840, 840) combined similarity
        │ PnL cross-sectional demean → 피어슨 상관 → |·|
        │ 텍스트 코사인도 계산
        │ combined = 0.85·|pnl| + 0.15·text
        ▼
  [08b] Behavior Recluster ───── 76 behavior clusters / 402 noise
        │ combined distance 로 HDBSCAN 재클러스터링
        ▼
  [08] NetworkX Graph ────────── 840 노드 / 7,546 엣지
        │
        ▼
  [09a] PyVis HTML (그래프 뷰)
  [09b] Plotly UMAP (맵 뷰)       ─ 정적 사이트 assets
  [09c] Seaborn Heatmap
  [09d] 사이트 조립 ──────────── output/site/ (IQC-SITE 로 배포)
        │  ├ 검색 인덱스 (data.json)
        │  ├ threshold별 그룹 분류 (groups.json)
        │  └ 필드별 이웃 상관 top-500 (neighbors/*.json)
        ▼
  [10] Diversifier Finder
        제출 알파 expression → 상관 낮은 필드 top-N 추천
```

---

## 단계별 상세 설명

### 01. `scripts/01_build_corpus.py` — 코퍼스 빌드

- **입력**: `references/wq_brain_all_fields.csv` (2,094 필드)
- **출력**: `data/field_corpus.jsonl` — 필드당 1줄
- **필터**:
  - `usable_fields.json` 에 정의된 "사용 가능한 dataset" 에만 속한 필드만 통과
  - coverage 0.0 이상
  - alpha_count 0 이상 (제한 없음)
- **결과**: 1,008 kept / 1,086 dropped
- **각 레코드 구조**:
  ```json
  {
    "field_id": "anl4_afv4_eps_mean",
    "text": "anl4_afv4_eps_mean | Earnings per share - mean ... | dataset=...",
    "description": "Earnings per share - mean of estimations for annual frequency",
    "dataset_name": "Analyst Estimate Data for Equity",
    "category_name": "Analyst",
    "subcategory_name": "Analyst Estimates",
    "coverage": 0.95,
    "alpha_count": 156,
    "user_count": 42,
    "type": "MATRIX"
  }
  ```

### 02. `scripts/02_build_embeddings.py` — 임베딩 생성

- **모델**: `BAAI/bge-m3` (dim=1024, multilingual, dense retrieval용)
- **입력**: `field_corpus.jsonl` 의 `text` 필드 (field_id + description + dataset + category 합친 자연어)
- **출력**:
  - `data/embeddings.npy` — (N, 1024) float32
  - `data/field_ids.json` — 인덱스 순서 매핑
  - `data/chroma_db/` — ChromaDB 벡터 스토어 (유사도 검색용, 본 저장소에선 제외)
- **normalize**: 코사인 유사도 ↔ 유클리드 거리 등가화를 위해 L2 정규화
- **결과**: 854개 필드가 임베딩됨 (corpus 1,008 중 일부는 중복 제거됨)

### 03. `scripts/03_cluster_fields.py` — HDBSCAN 클러스터링

- **입력**: `embeddings.npy`
- **알고리즘**: HDBSCAN (euclidean metric, min_cluster_size=5, min_samples=2)
- **출력**: `data/clusters.json` — `{field_id: cluster_id, -1 = noise}`
- **결과**: 97 clusters / 231 noise
- 의미 기반(semantic) 1차 분류 — "이름이 비슷한 필드끼리" 모음

### 04. `scripts/04_pick_representatives.py` — 대표 필드 선정

- **점수식**:
  ```
  score = 0.7 × centroid_proximity + 0.3 × log(alpha_count + 1)
  ```
- 각 클러스터에서 score 최대 필드를 대표로 선정
- **출력**: `data/representatives.json` — `{cluster_id: field_id}`
- **결과**: 97 대표 필드

### 05b. `scripts/05b_build_single_field_alphas.py` — 단일 필드 probe 알파 수식 생성

- 각 필드에 대해 가장 단순한 알파 수식 생성:
  ```
  rank(ts_backfill(X, 20))
  ```
- **출력**: `data/probe_alphas.jsonl`
- **결과**: 1,008 개의 수식

> (`05_build_probe_alphas.py` 는 쌍별 probe 생성용이며 이번 파이프라인에서는 미사용)

### 06. `scripts/06_step4_validate.py` — 가정 검증

핵심 가정:

> **"임베딩 유사도가 높으면 실제 PnL 상관도 높다"**

이를 50개 intra-cluster 쌍 + 50개 inter-cluster 쌍으로 검증.

- **intra 평균 similarity**: 0.73 (기준 ≥ 0.5 → PASS)
- **inter 평균 similarity**: 0.80 (기준 ≤ 0.3 → **FAIL**, 너무 높음)
- 결과: `INTER_HIGH` — intra 만 통과

> inter 가 높게 나온 것은 WQ 필드들이 전반적으로 "펀더멘탈" 이라는 하나의 거대 메가클러스터에 속해 있기 때문. 독립성을 확보하려면 뒤의 combined-similarity 기반 재클러스터링이 필요.

### 07b. `scripts/07b_run_single_sims_and_pnl.py` — 본 시뮬레이션

- 각 필드의 단일 probe 알파를 WQ BRAIN 에 시뮬 → 결과 메타 + **daily-pnl 벡터** 수집
- **WQB 설정** (config.yaml):
  - region: USA, universe: TOP3000
  - delay: 1, decay: 4
  - neutralization: SUBINDUSTRY, truncation: 0.08
- **재개 지원**: 이미 완료된 field_id 는 skip (`single_field_meta.jsonl` 기반)
- **3-shard 병렬**: `--shard 0/3`, `1/3`, `2/3` 으로 동시 실행 (WQB 슬롯 한도 고려)
- **출력**:
  - `data/single_field_meta.jsonl` — 필드별 시뮬 메타 (sharpe, fitness, turnover, returns, drawdown, alpha_id, status)
  - `data/pnl_records.jsonl` — 필드별 일별 PnL 레코드 (long format, 543k+ 줄)
  - `data/failures.jsonl` — 시뮬 실패 기록
- **결과**: **1,008/1,008 완료, pass=993, fail=15, pnl_ok=991**

### 08a. `scripts/08a_build_similarity_matrix.py` — 유사도 매트릭스

```python
# 1. pnl_records → (N, T) wide matrix
# 2. cross-sectional demean (각 날짜의 알파 평균 제거)
# 3. np.corrcoef → (N, N) PnL Pearson correlation
# 4. embeddings.npy → (N, N) text cosine
# 5. combined = 0.85 × |pnl_corr| + 0.15 × text_cos
```

- **출력**:
  - `data/similarity_pnl.npy` — (N, N) float32
  - `data/similarity_text.npy`
  - `data/similarity_combined.npy`
  - `data/similarity_field_ids.json` — 인덱스 순서
  - `data/similarity_meta.json` — 설정 · 통계

### 08b. `scripts/08b_behavior_recluster.py` — 행동 기반 재클러스터링

- 거리 = 1 - combined_similarity 로 HDBSCAN 재실행
- 의미가 아닌 **실제 시뮬 결과 행동** 기준으로 클러스터링
- **출력**: `data/clusters_behavior.json`
  ```json
  {
    "n_clusters": 76,
    "n_noise": 402,
    "assignments": {field_id: cluster_id},
    "centroids": {cluster_id: representative_field_id}
  }
  ```

### 08. `scripts/08_build_graph.py` — NetworkX 그래프

- 노드 = 필드, 엣지 = 강한 상관 페어
- centroid, cluster, coverage, alpha_count, sharpe 등을 노드 속성으로
- **출력**:
  - `output/field_graph.gpickle` — NetworkX 직렬화
  - `output/field_graph_edges.csv` — 엣지 목록 (엑셀 확인용)
- **결과**: 840 nodes, 7,546 edges

### 09a. `scripts/09a_build_graph_html.py` — PyVis 그래프 HTML

- 다크 모노크롬 테마, 엣지 밝기 = 유사도
- centroid 는 라벨 표시, 나머지는 hover 시 영문 description 툴팁
- 부모 프레임에서 postMessage 로 특정 필드로 focus + 노란 하이라이트 가능
- **출력**: `output/site/views/graph.html`

### 09b. `scripts/09b_build_map_html.py` — Plotly UMAP 맵 HTML

- `umap-learn` 의 `UMAP(metric='precomputed')` 으로 2D 프로젝션
- 거리 매트릭스 = 1 - combined
- 클러스터별 그레이 5단계, centroid 는 흰 테두리 링
- **출력**:
  - `output/site/views/map.html`
  - `data/umap_coords.json` — 다른 뷰에서 좌표 재사용을 위해 별도 저장

### 09c. `scripts/09c_build_heatmap_png.py` — Seaborn 히트맵

- 필드 × 필드 combined similarity 히트맵 PNG
- **출력**: `output/site/views/heatmap.png`

### 09d. `scripts/09d_build_site.py` — 정적 사이트 조립

- `output/site/` 에 SPA 스타일 사이트 빌드 (서버 불필요)
- 구성:
  - `index.html` + `detail.html` + `style.css` + `main.js` + `detail.js`
  - `data.json` — 검색 인덱스
  - `groups.json` — threshold 별 그룹 분류
  - `neighbors/<field_id>.json` — 필드별 top-500 이웃 상관 (840개 파일)
- 포함 기능:
  - **Agglomerative clustering (complete linkage)** 으로 4가지 threshold 에서 그룹 생성 (≥0.10 / 0.25 / 0.35 / 0.50)
  - 각 그룹 라벨은 지배적 subcategory 또는 category 기반 자동 생성
  - 클라이언트사이드에서 top-N / group CSV 내보내기 (expr 포함)
- **결과**: 사이트가 [IQC-SITE repo](https://github.com/superjin1218/IQC-SITE) 로 복사되어 Vercel 배포

### 10. `scripts/10_find_diversifier.py` — Diversifier 추천

- 입력: 제출할 알파의 expression 또는 field_id
- 출력: 상관관계 낮은 상위 N개 필드 추천 (제출 알파와 orthogonal 한 후보)

---

## 핵심 수식 · 알고리즘

### PnL 유사도 (cross-sectional demean 적용)

```python
# Y: (N_fields, T_days) wide PnL matrix
Y_demeaned = Y - Y.mean(axis=0)  # 각 날짜의 전체 평균 제거
# → "전반적으로 시장이 오르면 같이 오른다" 효과 제거
pnl_corr = np.corrcoef(Y_demeaned)  # (N, N) 피어슨 상관
```

### Combined similarity

```
combined[i,j] = 0.85 × |pnl_corr[i,j]| + 0.15 × text_cos[i,j]
```

- 부호 무시(절대값) → "반대 방향으로 움직이지만 같은 정보원" 인 페어도 유사로 간주
- 텍스트 유사도는 희소한 시뮬 실패 필드를 보완하는 역할 (15%만)

### 그룹 분류 (정적 사이트용, 09d 에서 수행)

```python
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

d = 1.0 - combined  # 거리 매트릭스
Z = linkage(squareform(d), method="complete")
for thr in [0.10, 0.25, 0.35, 0.50]:
    labels = fcluster(Z, t=1-thr, criterion="distance")
    # → 그룹 내 모든 페어의 combined similarity ≥ thr 보장
```

**Complete linkage + 거리 컷오프**는 "그룹 내 최악의 페어도 최소한 X 이상 유사"를 보장하므로, "같은 그룹 = 같은 행동" 이라는 해석이 엄격하게 성립합니다.

### 그룹 한 단어 라벨링

```
if 가장 많은 subcategory 가 그룹의 60% 이상 → 그 subcategory 사용
elif 가장 많은 category 가 60% 이상 → category 사용
else → "top1 / top2" mixed 표기
```

---

## 데이터 파일 명세

### 입력 (references/)

| 파일 | 크기 | 설명 |
|------|------|------|
| `wq_brain_all_fields.csv` | ~2,094 rows | WQ BRAIN 전체 필드 메타 덤프 |
| `wq_brain_top_fields.csv` | subset | 많이 쓰이는 필드만 |
| `usable_fields.json` | - | 사용 가능한 dataset 리스트 |
| `datasets_compact.json` | - | 데이터셋 메타 |
| `worldquant_operators.csv` | - | 연산자 명세 |
| `operators_compact.json` | - | 연산자 압축 메타 |
| `expression_templates.json` | - | 알파 expression 템플릿 |
| `field_families.json` | - | 필드 패밀리 (유사도 1차 필터용) |
| `regime_alpha_strategies.json` | - | 레짐별 전략 예시 |
| `IQC_Alpha_Strategy_Analysis (1).md` | - | 46개 PASS 알파 분석 문서 |
| `brain_navigator.md` | - | WQ BRAIN UI 가이드 |

### 중간 산출물 (data/)

| 파일 | 크기 | 설명 |
|------|------|------|
| `field_corpus.jsonl` | 460KB | 01 출력 — 필드 텍스트 + 메타 |
| `embeddings.npy` | 3.4MB | 02 출력 — (854, 1024) float32 |
| `field_ids.json` | - | 02 출력 — embedding 인덱스 매핑 |
| `clusters.json` | - | 03 출력 — HDBSCAN 초기 클러스터 |
| `representatives.json` | - | 04 출력 — 클러스터 대표 필드 |
| `probe_alphas.jsonl` | 108KB | 05b 출력 — 단일 필드 알파 수식 |
| `single_field_meta.jsonl` | 316KB | 07b 출력 — 필드별 시뮬 결과 메타 |
| `pnl_records.jsonl` | 88MB | 07b 출력 — 필드별 일별 PnL (long format) |
| `failures.jsonl` | - | 07b 출력 — 시뮬 실패 기록 |
| `similarity_pnl.npy` | 2.7MB | 08a — (840, 840) PnL 피어슨 상관 |
| `similarity_text.npy` | 2.7MB | 08a — 텍스트 코사인 |
| `similarity_combined.npy` | 2.7MB | 08a — 블렌드된 최종 유사도 |
| `similarity_field_ids.json` | - | 08a — 행/열 순서 매핑 |
| `similarity_meta.json` | - | 08a — 빌드 설정/통계 |
| `clusters_behavior.json` | - | 08b — 행동 기반 재클러스터링 |
| `umap_coords.json` | - | 09b — 맵 뷰용 2D 좌표 |
| `cluster_comparison.json` | - | 초기 vs 행동 클러스터 비교 |

### 결과물 (output/)

| 파일 | 크기 | 설명 |
|------|------|------|
| `field_graph.gpickle` | 636KB | NetworkX 그래프 직렬화 |
| `field_graph_edges.csv` | 740KB | 엣지 목록 CSV |

> `output/site/` 는 크기가 크고 [IQC-SITE](https://github.com/superjin1218/IQC-SITE) 저장소에 별도 배포되므로 본 저장소에선 제외했습니다.

### 로그 (logs/)

- `progress.json` — 파이프라인 단계별 상태
- `step{NN}.log` — 각 단계 실행 로그
- `post_sim_*.log` — 후처리(08a~09d) 체인 로그

---

## 설정 (config.yaml)

주요 항목:

```yaml
embedding:
  model: BAAI/bge-m3          # 또는 intfloat/multilingual-e5-small (경량)
  batch_size: 32
  normalize: true

filter:
  use_usable_fields_only: true
  min_coverage: 0.0
  min_alpha_count: 0

cluster:
  algorithm: hdbscan
  target_clusters: 100
  min_cluster_size: 5
  min_samples: 2
  metric: euclidean

representative:
  centroid_weight: 0.7
  alphacount_weight: 0.3

probe:
  template: subtract          # 이번 파이프라인에선 단일 필드 probe 를 써서 미사용
  lookback: 20

step4:
  intra_samples: 50
  inter_samples: 50
  min_intra_similarity: 0.5
  max_inter_similarity: 0.3

wqb:
  region: USA
  universe: TOP3000
  delay: 1
  decay: 4
  neutralization: SUBINDUSTRY
  truncation: 0.08
  max_wait_sec: 600
  env_file: ../.env          # WQB_USERNAME / WQB_PASSWORD 담긴 파일
```

---

## 실행 방법

### 0. 사전 준비

```bash
# WQB 계정 환경변수 파일 (저장소 상위 경로에)
cat > ../.env <<EOF
WQB_USERNAME=your_username
WQB_PASSWORD=your_password
EOF
```

### 1. 의존성 설치

```bash
bash install.sh
# 또는
pip install -r requirements.txt
```

### 2. 파이프라인 실행

**전체 실행**:
```bash
bash run_pipeline.sh       # 01 → 06 (시뮬 전)
bash run_post_sim.sh       # 08a → 09d (시뮬 후 후처리 체인)
```

**개별 실행**:
```bash
python scripts/01_build_corpus.py
python scripts/02_build_embeddings.py
python scripts/03_cluster_fields.py
python scripts/04_pick_representatives.py
python scripts/05b_build_single_field_alphas.py
python scripts/06_step4_validate.py
# ↓ 본 시뮬 (재개 지원, 3-shard 병렬 권장)
python scripts/07b_run_single_sims_and_pnl.py --shard 0/3 &
python scripts/07b_run_single_sims_and_pnl.py --shard 1/3 &
python scripts/07b_run_single_sims_and_pnl.py --shard 2/3 &
wait
# ↓ 후처리
python scripts/08a_build_similarity_matrix.py
python scripts/08b_behavior_recluster.py
python scripts/08_build_graph.py
python scripts/09a_build_graph_html.py
python scripts/09b_build_map_html.py
python scripts/09c_build_heatmap_png.py
python scripts/09d_build_site.py
```

### 3. 진행률 대시보드 (선택)

```bash
python scripts/dashboard.py
# → 별도 터미널에서 rich 기반 실시간 진행률 + 통계
```

### 4. 사이트 미리보기

```bash
cd output/site && python3 -m http.server 8080
# 브라우저 → http://localhost:8080
```

---

## 결과 요약

| 지표 | 값 |
|------|-----|
| 원본 필드 수 | ~2,094 |
| Corpus 통과 | 1,008 |
| Embedding | 854 × 1024 |
| 초기 HDBSCAN 클러스터 | 97 clusters / 231 noise |
| Step 4 intra 유사도 (평균) | 0.73 (PASS) |
| Step 4 inter 유사도 (평균) | 0.80 (INTER_HIGH) |
| Main 시뮬 완료 | 1,008 / 1,008 |
| Main 시뮬 PASS | 993 |
| Main 시뮬 FAIL | 15 |
| Daily-PnL 수집 성공 | 991 |
| 최종 그래프 | 840 nodes / 7,546 edges |
| 행동 기반 클러스터 | 76 / 402 noise |
| 그룹 분류 (≥0.10) | 14 groups |
| 그룹 분류 (≥0.25) | 44 groups |
| 그룹 분류 (≥0.35) | 66 groups |
| 그룹 분류 (≥0.50) | 119 groups |

---

## 폴더 구조

```
IQCRAG/
├── README.md                         본 문서
├── config.yaml                       파이프라인 전체 설정
├── requirements.txt                  Python 의존성
├── install.sh                        설치 스크립트
├── run_pipeline.sh                   01 ~ 06 실행
├── run_post_sim.sh                   08a ~ 09d 체인 실행
│
├── scripts/                          모든 파이프라인 스크립트
│   ├── common.py                     경로/진행률/WQB 클라이언트 유틸
│   ├── preflight.py                  환경 점검
│   ├── dashboard.py                  rich 기반 실시간 진행률
│   ├── 01_build_corpus.py
│   ├── 02_build_embeddings.py
│   ├── 03_cluster_fields.py
│   ├── 04_pick_representatives.py
│   ├── 05_build_probe_alphas.py      (쌍 probe — 이번엔 미사용)
│   ├── 05b_build_single_field_alphas.py  (단일 probe — 사용)
│   ├── 06_step4_validate.py
│   ├── 07_run_main_simulation.py     (쌍 시뮬 — 이번엔 미사용)
│   ├── 07b_run_single_sims_and_pnl.py  (단일 시뮬 — 사용)
│   ├── 08_build_graph.py
│   ├── 08a_build_similarity_matrix.py
│   ├── 08b_behavior_recluster.py
│   ├── 09_visualize_graph.py         (구버전, 09a~09d 로 대체됨)
│   ├── 09a_build_graph_html.py
│   ├── 09b_build_map_html.py
│   ├── 09c_build_heatmap_png.py
│   ├── 09d_build_site.py
│   └── 10_find_diversifier.py
│
├── references/                       입력 메타 · 전략 문서
│   ├── wq_brain_all_fields.csv
│   ├── wq_brain_top_fields.csv
│   ├── usable_fields.json
│   ├── datasets_compact.json
│   ├── operators_compact.json
│   ├── worldquant_operators.csv
│   ├── expression_templates.json
│   ├── field_families.json
│   ├── regime_alpha_strategies.json
│   ├── brain_navigator.md
│   ├── IQC_Alpha_Strategy_Analysis (1).md
│   ├── papers/
│   └── research_log_by_category/
│
├── data/                             중간 산출물
│   ├── field_corpus.jsonl
│   ├── embeddings.npy
│   ├── field_ids.json
│   ├── clusters.json
│   ├── representatives.json
│   ├── probe_alphas.jsonl
│   ├── single_field_meta.jsonl
│   ├── pnl_records.jsonl
│   ├── failures.jsonl
│   ├── similarity_combined.npy
│   ├── similarity_pnl.npy
│   ├── similarity_text.npy
│   ├── similarity_field_ids.json
│   ├── similarity_meta.json
│   ├── clusters_behavior.json
│   ├── umap_coords.json
│   └── cluster_comparison.json
│
├── output/                           최종 산출물 (정적 사이트 제외)
│   ├── field_graph.gpickle
│   └── field_graph_edges.csv
│
└── logs/                             파이프라인 로그
    ├── progress.json
    └── step*.log
```

---

## 관련 저장소

- **[IQC-SITE](https://github.com/superjin1218/IQC-SITE)** — 본 파이프라인이 생성한 정적 시각화 사이트 (Vercel 배포용)

## 라이선스

개인 연구용 — 데이터/필드 메타는 WorldQuant BRAIN 플랫폼 기반이며, 시뮬 결과 및 분석은 본인의 알파 연구 산출물입니다.
