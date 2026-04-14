#!/bin/bash
# Field Graph 의존성 설치

set -e
cd "$(dirname "$0")"

echo "=== Field Graph 의존성 설치 ==="
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

echo ""
echo "=== 설치 완료 ==="
echo "다음 명령으로 파이프라인을 실행할 수 있습니다:"
echo "  bash run_pipeline.sh"
echo "진행률 대시보드:"
echo "  python3 scripts/dashboard.py"
