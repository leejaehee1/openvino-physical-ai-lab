#!/usr/bin/env bash
# B70 PC에서 학생 zip들을 풀고, 형식을 점검하고, 하나의 데이터셋으로 병합한 뒤 zip으로 묶는다.
#
# 사용법:
#   ./merge_class_datasets.sh ~/class/inbox class_merged [작업 폴더]
#
# 전제:
#   - lerobot(0.6.x, 'lerobot[dataset]')이 설치된 Python 환경에서 실행 (Studio 컨테이너 안 또는 별도 venv)
#   - inbox에 s01.zip, s02.zip ... 형태로 학생 zip이 모여 있음
#
# 검증: lerobot 0.6.1에서 LeRobotDataset.create로 만든 v3 데이터셋 3개를 이 스크립트로 병합해 확인함.
#       실제 Studio Export zip으로는 리허설에서 한 번 더 확인할 것.
set -euo pipefail

INBOX="${1:?inbox 폴더를 지정하세요 (예: ~/class/inbox)}"
MERGED="${2:-class_merged}"
WORK="${3:-$HOME/class}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$WORK/src"
OUT="$WORK/merged/$MERGED"
mkdir -p "$SRC" "$WORK/merged"

ids=()
roots=()
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
  rm -rf "${SRC:?}/$name"
  mv "$(dirname "$(dirname "$info")")" "$SRC/$name"
  rm -rf "$tmp"
  ids+=("$name")
  roots+=("$SRC/$name")
  echo "[풀기] $name → $SRC/$name"
done

if (( ${#ids[@]} < 2 )); then
  echo "병합할 데이터셋이 2개 미만입니다." >&2
  exit 1
fi

echo
echo "== 형식 점검 =="
if ! python3 "$SCRIPT_DIR/check_datasets.py" "${roots[@]}"; then
  echo
  echo "불일치 데이터셋을 inbox에서 빼고 다시 실행하세요." >&2
  exit 1
fi

to_list() { local out="[" x; for x in "$@"; do out+="'$x', "; done; echo "${out%, }]"; }

echo
echo "== 병합 → $OUT =="
rm -rf "$OUT"
lerobot-edit-dataset \
  --new_repo_id "local/$MERGED" \
  --new_root "$OUT" \
  --operation.type merge \
  --operation.repo_ids "$(to_list "${ids[@]}")" \
  --operation.roots "$(to_list "${roots[@]}")"

python3 - "$OUT" <<'EOF'
import json, sys
info = json.load(open(f"{sys.argv[1]}/meta/info.json"))
print(f"병합 결과: episodes={info['total_episodes']} frames={info['total_frames']} fps={info['fps']}")
EOF

zipfile="$WORK/merged/$MERGED.zip"
rm -f "$zipfile"
(cd "$OUT" && zip -qr "$zipfile" .)
echo "완료: $zipfile  → Studio에서 Import dataset으로 가져오세요."
