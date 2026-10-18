#!/usr/bin/env bash
# B70 PC에서 학생 zip들을 풀고, 형식을 점검하고, 하나의 데이터셋으로 병합한 뒤 zip으로 묶는다.
#
# 사용법:
#   ./merge_class_datasets.sh ~/class/inbox class_merged
#
# 전제:
#   - lerobot(0.6.x)이 설치된 Python 환경에서 실행 (Studio 컨테이너 안 또는 별도 venv)
#   - inbox에 s01.zip, s02.zip ... 형태로 학생 zip이 모여 있음
#
# ⚠️ 리허설 필요: lerobot-edit-dataset의 merge가 로컬 데이터셋을 HF_LEROBOT_HOME/<repo_id>에서
#    찾는다는 가정으로 작성했다. 실제 Export zip 구조로 한 번 검증할 것.
set -euo pipefail

INBOX="${1:?inbox 폴더를 지정하세요 (예: ~/class/inbox)}"
MERGED="${2:-class_merged}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LEROBOT_HOME="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"
DEST_ROOT="$LEROBOT_HOME/local"
mkdir -p "$DEST_ROOT"

ids=()
for zip in "$INBOX"/*.zip; do
  name="$(basename "$zip" .zip)"
  tmp="$(mktemp -d)"
  unzip -q "$zip" -d "$tmp"
  # zip 최상위에 meta/가 바로 있을 수도, 폴더 하나 아래에 있을 수도 있다
  info="$(find "$tmp" -path '*/meta/info.json' -print -quit)"
  if [[ -z "$info" ]]; then
    echo "[건너뜀] $name: meta/info.json 없음 (LeRobot v3 zip이 아님)"
    rm -rf "$tmp"
    continue
  fi
  root="$(dirname "$(dirname "$info")")"
  rm -rf "$DEST_ROOT/$name"
  mv "$root" "$DEST_ROOT/$name"
  rm -rf "$tmp"
  ids+=("$name")
  echo "[풀기] $name → $DEST_ROOT/$name"
done

if (( ${#ids[@]} < 2 )); then
  echo "병합할 데이터셋이 2개 미만입니다." >&2
  exit 1
fi

echo
echo "== 형식 점검 =="
paths=()
for id in "${ids[@]}"; do paths+=("$DEST_ROOT/$id"); done
if ! python3 "$SCRIPT_DIR/check_datasets.py" "${paths[@]}"; then
  echo
  echo "불일치 데이터셋을 inbox에서 빼고 다시 실행하세요." >&2
  exit 1
fi

repo_list="["
for id in "${ids[@]}"; do repo_list+="'local/$id', "; done
repo_list="${repo_list%, }]"

echo
echo "== 병합: $repo_list → local/$MERGED =="
rm -rf "$DEST_ROOT/$MERGED"
lerobot-edit-dataset \
  --repo_id "local/$MERGED" \
  --operation.type merge \
  --operation.repo_ids "$repo_list"

python3 - "$DEST_ROOT/$MERGED" <<'EOF'
import json, sys
info = json.load(open(f"{sys.argv[1]}/meta/info.json"))
print(f"병합 결과: episodes={info['total_episodes']} frames={info['total_frames']} fps={info['fps']}")
EOF

out="$PWD/$MERGED.zip"
(cd "$DEST_ROOT/$MERGED" && zip -qr "$out" .)
echo "완료: $out  → Studio에서 Import dataset으로 가져오세요."
