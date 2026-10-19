#!/usr/bin/env python3
"""LeRobot v3 데이터셋의 에피소드 품질을 숫자로 점검한다.

사용법:
    python3 analyze_episodes.py <데이터셋 폴더> [--idle 0.0005]

데이터셋 폴더는 Studio에서 Export한 zip을 푼 곳(meta/, data/ 가 있는 폴더)이다.
필요 패키지: pandas, pyarrow

에피소드마다 아래 지표를 계산하고, 다른 에피소드와 크게 다르면 "확인"으로 표시한다.
  - 길이(초)       : 프레임 수 / fps
  - 시작 편차      : 첫 프레임 관절값이 전체 에피소드 시작 자세(중앙값)와 얼마나 다른가
  - 정지 비율(%)   : 관절 변화가 거의 없는 프레임의 비율 (멈칫, 대기)
  - 추종 오차      : |action - observation.state| 평균 (리더와 팔로워의 차이)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load(root: Path):
    info = json.loads((root / "meta" / "info.json").read_text())
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        sys.exit(f"{root}/data 아래에 parquet 파일이 없습니다.")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    for col in ("episode_index", "frame_index", "observation.state", "action"):
        if col not in df.columns:
            sys.exit(f"'{col}' 열이 없습니다. LeRobot v3 데이터셋이 맞는지 확인하세요.")
    return info, df.sort_values(["episode_index", "frame_index"])


def robust_outlier(values: np.ndarray, k: float = 3.0) -> np.ndarray:
    """중앙값에서 MAD 기준 k배 이상 벗어난 값을 True로 표시한다."""
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    # 에피소드들이 거의 같으면 MAD가 0에 가까워 사소한 차이도 튀어 보이므로 하한을 둔다
    mad = max(mad, 0.05 * abs(med), 1e-9)
    return np.abs(values - med) / (1.4826 * mad) > k


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="데이터셋 폴더 (meta/, data/ 포함)")
    ap.add_argument("--idle", type=float, default=0.0005,
                    help="정지로 볼 프레임 간 변화량 (관절별 전체 범위 대비 비율, 기본 0.0005)")
    args = ap.parse_args()

    info, df = load(args.root.expanduser())
    fps = info.get("fps", 30)
    names = info.get("features", {}).get("observation.state", {}).get("names")

    state_all = np.stack(df["observation.state"].to_numpy())
    joint_range = state_all.max(axis=0) - state_all.min(axis=0)
    joint_range[joint_range == 0] = 1.0

    starts = np.stack(df.groupby("episode_index")["observation.state"].first().to_numpy())
    start_ref = np.median(starts, axis=0)

    rows = []
    for ep, g in df.groupby("episode_index"):
        state = np.stack(g["observation.state"].to_numpy())
        action = np.stack(g["action"].to_numpy())
        step = np.abs(np.diff(state, axis=0)) / joint_range
        idle = (step.max(axis=1) < args.idle).mean() * 100 if len(step) else 0.0
        start_dev = np.linalg.norm((state[0] - start_ref) / joint_range)
        track = np.abs(action - state).mean()
        rows.append({"에피소드": int(ep), "길이(초)": len(g) / fps, "시작 편차": start_dev,
                     "정지 비율(%)": idle, "추종 오차": track})

    res = pd.DataFrame(rows)
    flags = []
    for col in ("길이(초)", "시작 편차", "정지 비율(%)", "추종 오차"):
        out = robust_outlier(res[col].to_numpy())
        flags.append(np.where(out, col.split("(")[0], ""))
    res["확인"] = [", ".join(f for f in fs if f) for fs in zip(*flags)]

    print(f"데이터셋: {args.root}  fps={fps}  에피소드={len(res)}  프레임={len(df)}")
    if names:
        print("관절:", ", ".join(names) if isinstance(names, list) else names)
    print()
    with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 120):
        print(res.to_string(index=False))

    flagged = res[res["확인"] != ""]
    print()
    if flagged.empty:
        print("크게 튀는 에피소드가 없습니다.")
    else:
        print("다시 재생해 볼 에피소드:", ", ".join(str(e) for e in flagged["에피소드"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
