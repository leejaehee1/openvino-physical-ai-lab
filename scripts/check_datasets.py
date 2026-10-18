#!/usr/bin/env python3
"""학생별 LeRobot 데이터셋의 meta/info.json을 비교해 병합 가능 여부를 점검한다.

사용법:
    python3 check_datasets.py ~/.cache/huggingface/lerobot/local/s01 ~/.cache/huggingface/lerobot/local/s02 ...

첫 번째 데이터셋을 기준으로 fps, robot_type, features(키·shape·dtype)가 다른 데이터셋을 표시한다.
"""
import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    info = json.loads((path / "meta" / "info.json").read_text())
    feats = {
        k: (tuple(v.get("shape", [])), v.get("dtype"), tuple(v.get("names") or []) if isinstance(v.get("names"), list) else None)
        for k, v in info.get("features", {}).items()
    }
    return {
        "fps": info.get("fps"),
        "robot_type": info.get("robot_type"),
        "episodes": info.get("total_episodes"),
        "features": feats,
    }


def main() -> int:
    paths = [Path(p).expanduser() for p in sys.argv[1:]]
    if len(paths) < 2:
        print(__doc__)
        return 2

    base_path, *rest = paths
    base = load(base_path)
    print(f"기준: {base_path.name}  fps={base['fps']}  robot={base['robot_type']}  episodes={base['episodes']}")
    print("  features:", ", ".join(sorted(base["features"])))

    bad = []
    total = base["episodes"] or 0
    for p in rest:
        cur = load(p)
        total += cur["episodes"] or 0
        problems = []
        for key in ("fps", "robot_type"):
            if cur[key] != base[key]:
                problems.append(f"{key}: {cur[key]} (기준 {base[key]})")
        missing = set(base["features"]) - set(cur["features"])
        extra = set(cur["features"]) - set(base["features"])
        if missing:
            problems.append(f"없는 키: {sorted(missing)}")
        if extra:
            problems.append(f"추가된 키: {sorted(extra)}")
        for k in set(base["features"]) & set(cur["features"]):
            if cur["features"][k] != base["features"][k]:
                problems.append(f"{k} 형식 다름: {cur['features'][k]} (기준 {base['features'][k]})")
        status = "OK" if not problems else "불일치"
        print(f"[{status}] {p.name}  episodes={cur['episodes']}")
        for msg in problems:
            print("    -", msg)
        if problems:
            bad.append(p.name)

    print(f"\n총 에피소드(전체 합): {total}")
    if bad:
        print("병합에서 제외하거나 다시 녹화할 데이터셋:", ", ".join(bad))
        return 1
    print("모든 데이터셋 형식이 같습니다. 병합을 진행해도 됩니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
