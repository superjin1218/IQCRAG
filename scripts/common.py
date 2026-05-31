"""
Field Graph 파이프라인 공통 유틸리티.

- config.yaml 로드
- 경로 헬퍼
- progress.json 업데이트 (대시보드가 읽음)
- WQBSimulatorClient 로더 (기존 wqb_simulator.py 재사용)
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# field_graph 루트 경로
FIELD_GRAPH_ROOT = Path(__file__).resolve().parent.parent

# 상위 프로젝트의 scripts/ 를 PYTHONPATH 에 추가해서 wqb_simulator 를 가져다 쓸 수 있게
_PARENT_SCRIPTS = FIELD_GRAPH_ROOT.parent / "scripts"
if _PARENT_SCRIPTS.exists():
    sys.path.insert(0, str(_PARENT_SCRIPTS))


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """config.yaml 을 로드한다."""
    if path is None:
        path = FIELD_GRAPH_ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(rel: str) -> Path:
    """config 의 상대경로를 field_graph 루트 기준 절대경로로 변환."""
    p = Path(rel)
    if p.is_absolute():
        return p
    return (FIELD_GRAPH_ROOT / p).resolve()


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ── 진행률 업데이트 ──────────────────────────────────────────────────────────
# logs/progress.json 에 파이프라인의 현재 상태를 기록한다.
# 대시보드(scripts/dashboard.py)가 이 파일을 읽어서 실시간으로 표시한다.
#
# 스키마:
#   {
#     "updated_at": "2026-04-11T12:34:56",
#     "steps": {
#       "01_corpus":       {"status": "done",    "progress": 1.0, "message": "...", "started_at": ..., "finished_at": ...},
#       "02_embeddings":   {"status": "running", "progress": 0.42, "message": "batch 13/31", ...},
#       "03_cluster":      {"status": "pending", "progress": 0.0,  ...},
#       ...
#     },
#     "sim_stats": {  # 시뮬 스크립트가 주기적으로 업데이트
#       "total_planned": 4950,
#       "completed": 123,
#       "passed": 120,
#       "failed": 3,
#       "current_pair": "fnd6_fopo x anl4_eps_mean",
#       "throughput_per_hour": 180,
#       "eta_hours": 27.5
#     }
#   }
# ────────────────────────────────────────────────────────────────────────────

STEP_ORDER = [
    "01_corpus",
    "02_embeddings",
    "03_cluster",
    "04_representatives",
    "05_probe_alphas",
    "06_step4",
    "07_main_sim",
    "08_graph",
    "09_visualize",
]


def _progress_path(config: Dict[str, Any]) -> Path:
    return resolve_path(config["paths"]["progress_file"])


def load_progress(config: Dict[str, Any]) -> Dict[str, Any]:
    path = _progress_path(config)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "steps": {s: {"status": "pending", "progress": 0.0, "message": ""} for s in STEP_ORDER},
        "sim_stats": {},
    }


def save_progress(config: Dict[str, Any], progress: Dict[str, Any]) -> None:
    path = ensure_parent(_progress_path(config))
    progress["updated_at"] = datetime.now().isoformat(timespec="seconds")
    # atomic write — PID로 구분해서 여러 shard 동시 실행 시 충돌 방지
    tmp = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except FileNotFoundError:
        # 다른 프로세스가 동시에 rename 중이면 스킵 (다음 tick에 다시 시도)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def update_step(
    config: Dict[str, Any],
    step: str,
    status: Optional[str] = None,
    progress_value: Optional[float] = None,
    message: Optional[str] = None,
    sim_stats: Optional[Dict[str, Any]] = None,
) -> None:
    """단일 스텝 상태를 업데이트한다. 필요한 필드만 전달하면 된다."""
    prog = load_progress(config)
    step_state = prog["steps"].setdefault(step, {"status": "pending", "progress": 0.0, "message": ""})
    if status:
        step_state["status"] = status
        if status == "running" and "started_at" not in step_state:
            step_state["started_at"] = datetime.now().isoformat(timespec="seconds")
        if status in ("done", "failed"):
            step_state["finished_at"] = datetime.now().isoformat(timespec="seconds")
    if progress_value is not None:
        step_state["progress"] = float(max(0.0, min(1.0, progress_value)))
    if message is not None:
        step_state["message"] = message
    if sim_stats is not None:
        prog["sim_stats"] = sim_stats
    save_progress(config, prog)


# ── 로깅 헬퍼 ────────────────────────────────────────────────────────────────

def log_line(step: str, msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{step}] {msg}", flush=True)


# ── WQB 클라이언트 로더 (기존 wqb_simulator 재사용) ──────────────────────────

def _parse_env_file(env_path: Path) -> Dict[str, str]:
    """단순 KEY=VALUE env 파일 파서 (주석/빈줄/공백 처리)."""
    out: Dict[str, str] = {}
    if not env_path.exists():
        return out
    for line in env_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip()
        # 따옴표 제거
        if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
            v = v[1:-1]
        if k:
            out[k] = v
    return out


def make_wqb_client(config: Dict[str, Any], account_id: Optional[str] = None):
    """WQBSimulatorClient 로드. 단일 .env 파일에서 계정별 credentials 추출.

    Env 스키마 (config.wqb.env_file 단일 파일):
      WQ_EMAIL_A1, WQ_PASSWORD_A1
      WQ_EMAIL_A2, WQ_PASSWORD_A2
      ... (a3, a4 동일 패턴)
      WQ_BASE_URL (옵션, 모든 계정 공유)

    - account_id="a1" → WQ_EMAIL_A1 / WQ_PASSWORD_A1 을 WQ_EMAIL / WQ_PASSWORD 로 주입 후 인증
    - account_id=None → WQ_EMAIL / WQ_PASSWORD 그대로 (레거시 단일 계정 모드)

    process 단위로 os.environ 을 override 하므로 워커 프로세스 간 cred 간섭 없음.
    """
    try:
        from wqb_simulator import WQBSimulatorClient  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "wqb_simulator 모듈을 찾을 수 없습니다. "
            f"{_PARENT_SCRIPTS}/wqb_simulator.py 가 있는지 확인하세요."
        ) from e

    env_file = config["wqb"].get("env_file", "../.env")
    env_path = resolve_path(env_file)
    if not env_path.exists():
        raise FileNotFoundError(
            f"env 파일을 찾을 수 없습니다: {env_path}\n"
            f"템플릿을 복사해서 채우세요: cp ../.env.example ../.env"
        )

    env_vars = _parse_env_file(env_path)

    if account_id:
        sfx = account_id.upper()
        email_key = f"WQ_EMAIL_{sfx}"
        pw_key = f"WQ_PASSWORD_{sfx}"
        email = env_vars.get(email_key) or os.environ.get(email_key, "")
        password = env_vars.get(pw_key) or os.environ.get(pw_key, "")
        if not email or not password:
            raise ValueError(
                f"{email_key}/{pw_key} 가 {env_path} 또는 환경변수에서 발견되지 않음. "
                f"account_id={account_id!r}"
            )
        # 같은 프로세스의 다음 from_env 호출이 우리가 주입한 값을 읽도록 강제 (overwrite).
        os.environ["WQ_EMAIL"] = email
        os.environ["WQ_PASSWORD"] = password
    # else: 레거시 — env 파일에 WQ_EMAIL/WQ_PASSWORD 가 직접 있어야 함

    # 공통 옵션 (있으면 setdefault)
    if env_vars.get("WQ_BASE_URL"):
        os.environ.setdefault("WQ_BASE_URL", env_vars["WQ_BASE_URL"])

    client = WQBSimulatorClient.from_env(env_file=str(env_path))
    client.authenticate()
    return client


def build_wqb_settings(client, config: Dict[str, Any]) -> Dict[str, Any]:
    w = config["wqb"]
    return client.build_settings(
        region=w.get("region", "USA"),
        universe=w.get("universe", "TOP3000"),
        delay=w.get("delay", 1),
        decay=w.get("decay", 4),
        neutralization=w.get("neutralization", "SUBINDUSTRY"),
        truncation=w.get("truncation", 0.08),
    )
