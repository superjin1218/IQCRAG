"""
파이프라인 실행 전 환경 점검.

실행: python3 scripts/preflight.py
  → 의존성/파일/설정이 갖춰졌는지 리스트로 보여준다.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, resolve_path

OK = "\033[92m[OK]  \033[0m"
MISS = "\033[91m[MISS]\033[0m"
WARN = "\033[93m[WARN]\033[0m"


def check_imports():
    print("# 파이썬 모듈")
    mods = [
        ("numpy", True),
        ("pandas", True),
        ("yaml", True),
        ("rich", False),
        ("sklearn", True),
        ("hdbscan", True),
        ("sentence_transformers", True),
        ("chromadb", False),  # 실패해도 npy fallback 가능
        ("networkx", True),
        ("pyvis", True),
        ("torch", True),
    ]
    missing_required = []
    for name, required in mods:
        try:
            importlib.import_module(name)
            print(f"  {OK} {name}")
        except ImportError:
            tag = MISS if required else WARN
            print(f"  {tag} {name}" + ("  (필수)" if required else "  (선택)"))
            if required:
                missing_required.append(name)
    return missing_required


def check_inputs():
    print("\n# 입력 파일")
    config = load_config()
    csv_p = resolve_path(config["input"]["all_fields_csv"])
    usable_p = resolve_path(config["input"]["usable_fields_json"])
    for p, req in [(csv_p, True), (usable_p, False)]:
        if p.exists():
            print(f"  {OK} {p}")
        else:
            print(f"  {MISS if req else WARN} {p}")


def check_outputs():
    print("\n# 중간 산출물 (있으면 이어 실행 가능)")
    config = load_config()
    keys = [
        ("corpus_file",        "01 corpus"),
        ("embeddings_file",    "02 embeddings"),
        ("clusters_file",      "03 cluster"),
        ("representatives_file","04 representatives"),
        ("probe_alphas_file",  "05 probe"),
        ("step4_results_file", "06 step4"),
        ("pair_results_file",  "07 main sim"),
        ("graph_pickle",       "08 graph"),
        ("graph_html",         "09 visualize"),
    ]
    for key, label in keys:
        p = resolve_path(config["paths"][key])
        if key == "pair_results_file":
            p = p.with_suffix(".jsonl")
        if p.exists():
            if p.is_file():
                size = p.stat().st_size
                print(f"  {OK} {label:25s}  {p.name} ({size:,} B)")
            else:
                print(f"  {OK} {label:25s}  {p.name}/")
        else:
            print(f"  {WARN} {label:25s}  (없음)")


def check_wqb_env():
    print("\n# WQB 인증 (.env)")
    config = load_config()
    env_path = resolve_path(config["wqb"].get("env_file", "../.env"))
    if not env_path.exists():
        print(f"  {MISS} {env_path} 없음")
        return
    content = env_path.read_text(encoding="utf-8")
    has_email = "WQ_EMAIL=" in content and "WQ_EMAIL=\n" not in content
    has_pw = "WQ_PASSWORD=" in content and "WQ_PASSWORD=\n" not in content
    print(f"  {OK if has_email else MISS} WQ_EMAIL")
    print(f"  {OK if has_pw else MISS} WQ_PASSWORD")


def main():
    print("=== Field Graph Preflight Check ===\n")
    missing = check_imports()
    check_inputs()
    check_outputs()
    check_wqb_env()

    print()
    if missing:
        print("\033[91m필수 모듈 누락:\033[0m " + ", ".join(missing))
        print("설치:")
        print("  bash install.sh")
        print("또는:")
        print("  pip install --user --break-system-packages -r requirements.txt")
        sys.exit(1)
    else:
        print("\033[92mpreflight OK — 파이프라인 실행 가능\033[0m")


if __name__ == "__main__":
    main()
