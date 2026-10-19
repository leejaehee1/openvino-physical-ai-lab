#!/usr/bin/env python3
"""LeRobot v3 데이터셋을 메타데이터부터 따라가며 들여다본다.

사용법:
    python3 explore_dataset.py <데이터셋 폴더>                                   # 요약 + 에피소드 목차
    python3 explore_dataset.py <데이터셋 폴더> --episode 3 --csv ep3.csv         # 관절값·행동을 CSV로
    python3 explore_dataset.py <데이터셋 폴더> --episode 3 --frame 13 --png f.png  # 영상 프레임을 이미지로

필요 패키지: pandas, pyarrow / 프레임 추출은 PyAV(av) 또는 OpenCV
LeRobot 0.6 기본 영상 코덱은 AV1이라 OpenCV로 안 열릴 수 있다. 그때는 `pip install av`.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_meta(root: Path):
    info = json.loads((root / "meta" / "info.json").read_text())
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    eps = pd.concat([pd.read_parquet(f) for f in sorted((root / "meta" / "episodes").rglob("*.parquet"))],
                    ignore_index=True).sort_values("episode_index")
    return info, tasks, eps


def video_keys(info: dict) -> list[str]:
    return [k for k, v in info["features"].items() if v.get("dtype") == "video"]


def summary(root: Path, info: dict, tasks: pd.DataFrame, eps: pd.DataFrame) -> None:
    print(f"데이터셋 : {root}")
    print(f"형식     : {info.get('codebase_version')}  로봇: {info.get('robot_type')}  fps: {info['fps']}")
    print(f"규모     : 에피소드 {info['total_episodes']}개, 프레임 {info['total_frames']}개, "
          f"약 {info['total_frames'] / info['fps']:.1f}초")
    print("\n[features]")
    for k, v in info["features"].items():
        names = v.get("names")
        extra = f"  {names}" if isinstance(names, list) and k in ("observation.state", "action") else ""
        print(f"  {k:28s} {v['dtype']:8s} {tuple(v['shape'])}{extra}")
    print("\n[tasks.parquet]")
    for task, row in tasks.iterrows():
        print(f"  {row['task_index']}: {task}")

    cols = ["episode_index", "length", "dataset_from_index", "dataset_to_index"]
    table = eps[cols].copy()
    table["초"] = table["length"] / info["fps"]
    for vk in video_keys(info):
        short = vk.split(".")[-1]
        table[f"{short} 파일"] = (eps[f"videos/{vk}/chunk_index"].astype(str) + "/"
                                + eps[f"videos/{vk}/file_index"].astype(str))
        table[f"{short} 구간(초)"] = (eps[f"videos/{vk}/from_timestamp"].round(2).astype(str) + "–"
                                   + eps[f"videos/{vk}/to_timestamp"].round(2).astype(str))
    print("\n[meta/episodes — 에피소드 목차]")
    with pd.option_context("display.width", 160, "display.float_format", "{:.2f}".format):
        print(table.to_string(index=False))

    # 목차가 끊김 없이 이어지는지 확인
    gaps = (eps["dataset_from_index"].to_numpy()[1:] != eps["dataset_to_index"].to_numpy()[:-1]).sum()
    print("\n목차 연속성:", "정상 (앞 에피소드 끝 = 다음 에피소드 시작)" if gaps == 0 else f"끊긴 곳 {gaps}군데")


def episode_frames(root: Path, info: dict, ep: pd.Series) -> pd.DataFrame:
    path = root / info["data_path"].format(chunk_index=int(ep["data/chunk_index"]),
                                           file_index=int(ep["data/file_index"]))
    df = pd.read_parquet(path)
    lo, hi = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
    return df[(df["index"] >= lo) & (df["index"] < hi)].sort_values("frame_index")


def save_csv(df: pd.DataFrame, info: dict, out: Path) -> None:
    rows = {"frame_index": df["frame_index"].to_numpy(), "timestamp": df["timestamp"].to_numpy()}
    for key in ("observation.state", "action"):
        names = info["features"][key].get("names") or [f"{i}" for i in range(info["features"][key]["shape"][0])]
        arr = np.stack(df[key].to_numpy())
        prefix = "state" if key == "observation.state" else "action"
        for i, n in enumerate(names):
            rows[f"{prefix}.{n}"] = arr[:, i]
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"CSV 저장: {out} ({len(df)}행)")


def read_frame(video: Path, t: float, fps: float):
    try:
        import av
        with av.open(str(video)) as c:
            stream = c.streams.video[0]
            for frame in c.decode(stream):
                if frame.time is not None and frame.time >= t - 0.5 / fps:
                    return frame.to_ndarray(format="rgb24")
    except ImportError:
        pass
    import cv2
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        sys.exit("프레임을 읽지 못했습니다. 영상 코덱(AV1 등) 때문일 수 있으니 `pip install av` 후 다시 실행하세요.")
    return bgr[:, :, ::-1]


def save_png(root: Path, info: dict, ep: pd.Series, frame: int, out: Path, key: str | None) -> None:
    vk = key or video_keys(info)[0]
    video = root / info["video_path"].format(video_key=vk, chunk_index=int(ep[f"videos/{vk}/chunk_index"]),
                                             file_index=int(ep[f"videos/{vk}/file_index"]))
    t = float(ep[f"videos/{vk}/from_timestamp"]) + frame / info["fps"]
    print(f"영상 파일 {video.relative_to(root)} 의 {t:.3f}초 위치 "
          f"(= 에피소드 시작 {ep[f'videos/{vk}/from_timestamp']:.3f}초 + {frame}/{info['fps']})")
    rgb = read_frame(video, t, info["fps"])
    try:
        from PIL import Image
        Image.fromarray(rgb).save(out)
    except ImportError:
        import cv2
        cv2.imwrite(str(out), rgb[:, :, ::-1])
    print(f"이미지 저장: {out} {rgb.shape}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--episode", type=int, help="자세히 볼 에피소드 번호")
    ap.add_argument("--csv", type=Path, help="에피소드 관절값·행동을 저장할 CSV 경로")
    ap.add_argument("--frame", type=int, help="이미지로 뽑을 프레임 번호 (에피소드 안 기준)")
    ap.add_argument("--png", type=Path, help="프레임 이미지를 저장할 경로")
    ap.add_argument("--camera", help="카메라 키 (기본: 첫 번째 카메라)")
    args = ap.parse_args()

    root = args.root.expanduser()
    info, tasks, eps = load_meta(root)
    if args.episode is None:
        summary(root, info, tasks, eps)
        return 0

    sel = eps[eps["episode_index"] == args.episode]
    if sel.empty:
        sys.exit(f"에피소드 {args.episode}이(가) 없습니다. (0–{int(eps['episode_index'].max())})")
    ep = sel.iloc[0]
    df = episode_frames(root, info, ep)
    print(f"에피소드 {args.episode}: 프레임 {len(df)}개, 작업 {list(ep['tasks'])}")
    print(f"  data 파일 {int(ep['data/chunk_index'])}/{int(ep['data/file_index'])}, "
          f"index {int(ep['dataset_from_index'])}–{int(ep['dataset_to_index']) - 1}")
    if args.csv:
        save_csv(df, info, args.csv)
    if args.frame is not None:
        if not 0 <= args.frame < len(df):
            sys.exit(f"프레임 번호는 0–{len(df) - 1} 사이여야 합니다.")
        save_png(root, info, ep, args.frame, args.png or Path(f"ep{args.episode}_f{args.frame}.png"), args.camera)
    return 0


if __name__ == "__main__":
    sys.exit(main())
