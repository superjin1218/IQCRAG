"""
Field Graph 진행률 대시보드.

rich 기반 터미널 대시보드. logs/progress.json 을 1초마다 읽어서 표시한다.
별도 터미널에서 실행:
  python3 scripts/dashboard.py

표시 내용:
  - 9개 스텝별 상태 (pending / running / done / failed), 진행률 바
  - 현재 스텝 메시지
  - 시뮬 통계 (step4 / main_sim 단계에서만): 완료/계획, 쓰루풋, ETA, 현재 쌍
  - 마지막 업데이트 시간

Ctrl+C 로 종료.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, resolve_path, STEP_ORDER

try:
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.console import Console, Group
    from rich.progress import Progress, BarColumn, TextColumn
    from rich.text import Text
    from rich.align import Align
except ImportError:
    print("rich 가 설치돼 있지 않습니다. pip install rich")
    sys.exit(1)


STEP_LABELS = {
    "01_corpus":          "01 · Corpus",
    "02_embeddings":      "02 · Embeddings",
    "03_cluster":         "03 · Cluster",
    "04_representatives": "04 · Representatives",
    "05_probe_alphas":    "05 · Probe Alphas",
    "06_step4":           "06 · Step 4 Validation",
    "07_main_sim":        "07 · Main Simulation",
    "08_graph":           "08 · Build Graph",
    "09_visualize":       "09 · Visualize",
}

STATUS_STYLES = {
    "pending": "dim",
    "running": "bold yellow",
    "done":    "bold green",
    "failed":  "bold red",
}


def _read_progress(progress_path: Path) -> dict:
    if not progress_path.exists():
        return {"updated_at": "-", "steps": {}, "sim_stats": {}}
    try:
        prog = json.loads(progress_path.read_text(encoding="utf-8"))
    except Exception:
        return {"updated_at": "-", "steps": {}, "sim_stats": {}}

    # shard 파일 합산: single_field_meta*.jsonl 에서 실제 완료 수 계산
    data_dir = progress_path.parent.parent / "data"
    completed_ids = set()
    for meta_file in data_dir.glob("single_field_meta*.jsonl"):
        try:
            for line in meta_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    completed_ids.add(json.loads(line)["field_id"])
        except Exception:
            pass

    if completed_ids:
        total = prog.get("sim_stats", {}).get("total_planned", 984)
        done = len(completed_ids)
        pct = done / max(1, total)
        tph = prog.get("sim_stats", {}).get("throughput_per_hour", 0)
        # shard 수 감지
        n_shards = len(list(data_dir.glob("single_field_meta_s*.jsonl")))
        if n_shards > 0 and tph > 0:
            tph_total = tph * n_shards
        else:
            tph_total = tph
        remaining = total - done
        eta = remaining / tph_total if tph_total > 0 else 0

        prog.setdefault("sim_stats", {})
        prog["sim_stats"]["completed"] = done
        prog["sim_stats"]["total_planned"] = total
        prog["sim_stats"]["throughput_per_hour"] = round(tph_total, 1)
        prog["sim_stats"]["eta_hours"] = round(eta, 1)
        prog["sim_stats"]["shards_active"] = n_shards

        step07 = prog.get("steps", {}).get("07_main_sim", {})
        step07["progress"] = pct
        step07["message"] = f"{done}/{total}  shards={n_shards}  tph={tph_total:.0f}  eta={eta:.1f}h"

    return prog


def _steps_table(progress: dict) -> Table:
    t = Table(title="Pipeline Steps", expand=True, show_lines=False, header_style="bold cyan")
    t.add_column("Step", style="bold")
    t.add_column("Status", justify="left")
    t.add_column("Progress", justify="left", ratio=3)
    t.add_column("Message", justify="left", ratio=4)

    steps = progress.get("steps", {})
    for key in STEP_ORDER:
        label = STEP_LABELS.get(key, key)
        s = steps.get(key, {"status": "pending", "progress": 0.0, "message": ""})
        status = s.get("status", "pending")
        pv = float(s.get("progress", 0.0))
        msg = s.get("message", "")

        style = STATUS_STYLES.get(status, "")
        status_cell = Text(status, style=style)

        # 간단한 막대
        total_w = 30
        filled = int(round(pv * total_w))
        bar = Text()
        if status == "done":
            bar.append("█" * total_w, style="green")
        elif status == "failed":
            bar.append("█" * total_w, style="red")
        else:
            bar.append("█" * filled, style="yellow" if status == "running" else "grey35")
            bar.append("░" * (total_w - filled), style="grey23")
        bar.append(f"  {pv*100:5.1f}%", style="dim")

        t.add_row(label, status_cell, bar, Text(msg, style="white"))
    return t


def _sim_panel(progress: dict) -> Panel:
    stats = progress.get("sim_stats", {}) or {}
    if not stats:
        body = Text("현재 시뮬레이션 통계 없음", style="dim")
        return Panel(body, title="Simulation Stats", border_style="grey50")

    lines = []
    phase = stats.get("phase", "-")
    total = stats.get("total_planned", 0)
    done = stats.get("completed", 0)
    passed = stats.get("passed", 0)
    failed = stats.get("failed", 0)
    skipped = stats.get("skipped", 0)
    cur = stats.get("current_pair", "-")
    tph = stats.get("throughput_per_hour", 0)
    eta = stats.get("eta_hours", 0)

    pct = (done / total * 100.0) if total else 0.0

    t = Table.grid(padding=(0, 2))
    t.add_column(justify="right", style="bold")
    t.add_column(justify="left")
    shards = stats.get("shards_active", 0)
    t.add_row("Phase:", f"[cyan]{phase}[/cyan]")
    t.add_row("Completed:", f"{done} / {total}   ({pct:.1f}%)")
    if shards:
        t.add_row("Shards:", f"[bold cyan]{shards}[/bold cyan] concurrent sessions")
    t.add_row("Skipped:", f"{skipped}")
    t.add_row("Passed / Failed:", f"[green]{passed}[/green] / [red]{failed}[/red]")
    t.add_row("Current pair:", f"[white]{cur}[/white]")
    t.add_row("Throughput:", f"{tph} sim/hour")
    t.add_row("ETA:", f"{eta} hours" if eta else "-")

    return Panel(t, title="Simulation Stats", border_style="cyan")


def _header(progress: dict) -> Panel:
    updated = progress.get("updated_at", "-")
    now = datetime.now().strftime("%H:%M:%S")
    text = Text(f"Field Graph Pipeline Dashboard   |  updated: {updated}  |  now: {now}",
                justify="center", style="bold white")
    return Panel(text, style="on grey15")


def build_layout(progress: dict):
    return Group(
        _header(progress),
        _steps_table(progress),
        _sim_panel(progress),
    )


def main():
    config = load_config()
    progress_path = resolve_path(config["paths"]["progress_file"])

    console = Console()
    with Live(build_layout(_read_progress(progress_path)), console=console, refresh_per_second=2, screen=False) as live:
        try:
            while True:
                p = _read_progress(progress_path)
                live.update(build_layout(p))
                time.sleep(1.0)
        except KeyboardInterrupt:
            console.print("\n[dim]대시보드 종료[/dim]")


if __name__ == "__main__":
    main()
