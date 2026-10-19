#!/usr/bin/env python3
"""정책 하나를 CPU · GPU · NPU에서 돌려 추론 지연 시간을 비교한다 (로봇 없이 가능).

사용법:
    # 1) 인텔이 공개한 테스트용 ACT(무작위 가중치)로 바로 해 보기
    python3 bench_policy.py

    # 2) 우리 반 모델(runtime export의 exports/openvino 폴더)로
    python3 bench_policy.py --model ~/inference/exports/openvino \
        --obs '{"images.front": [1, 3, 480, 640], "state": [1, 6]}'

필요 패키지: physicalai (pip install physicalai)
관찰값 키와 모양은 모델마다 다르다. 우리 모델의 키는 export 폴더의 manifest/metadata를 보고 맞춘다.
기본값은 테스트용 ACT(OpenVINO/act-fp16-ov)의 입력: 카메라 2대 256×256 + 상태 8개.
"""
import argparse
import json

import numpy as np

DEFAULT_MODEL = "OpenVINO/act-fp16-ov"
DEFAULT_OBS = {"images.image": [1, 3, 256, 256], "images.image2": [1, 3, 256, 256], "state": [1, 8]}


def make_obs(spec: dict) -> dict:
    """키 → 모양 사전으로 가짜 관찰값을 만든다. 이미지는 [0, 1] float32 (LeRobot 규칙)."""
    rng = np.random.default_rng(0)
    return {k: rng.random(shape, dtype=np.float32) if k.startswith("images") else np.zeros(shape, np.float32)
            for k, shape in spec.items()}


def load(model: str, device: str):
    from physicalai.inference import InferenceModel
    if "/" in model and not model.startswith((".", "/", "~")):
        return InferenceModel.from_pretrained(model, device=device)
    from pathlib import Path
    return InferenceModel(Path(model).expanduser(), device=device)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="export 폴더 또는 Hugging Face 모델 ID")
    ap.add_argument("--obs", type=json.loads, default=DEFAULT_OBS, help="관찰값 키 → 모양 (JSON)")
    ap.add_argument("--devices", default="CPU,GPU,NPU", help="비교할 디바이스 (쉼표로 구분)")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    from physicalai.benchmark.performance import InferenceLatencyBenchmark

    obs = make_obs(args.obs)
    rows = []
    for device in args.devices.split(","):
        try:
            model = load(args.model, device)
            chunk = model.predict_action_chunk(obs)
            model.reset()
            inputs = (obs for _ in range(args.iters + 3))
            m = InferenceLatencyBenchmark(max_iters=args.iters, warmup_iters=3, max_duration=60000).run(model, inputs)
            if hasattr(model, "close"):  # 0.3.0에는 없고 main에는 있음
                model.close()
            rows.append((device, "OK", m["median_iter_time"] * 1000, m["mean_iter_time"] * 1000,
                         m["avg_warmup_iter_time"] * 1000, tuple(np.asarray(chunk).shape)))
        except Exception as e:  # 디바이스가 없거나 지원하지 않는 경우
            rows.append((device, f"실패: {type(e).__name__}: {str(e)[:60]}", None, None, None, None))

    print(f"\n모델: {args.model}")
    print(f"{'디바이스':6s} {'중앙값(ms)':>10s} {'평균(ms)':>9s} {'워밍업(ms)':>10s}  행동 묶음 모양")
    for device, status, med, mean, warm, shape in rows:
        if med is None:
            print(f"{device:6s} {status}")
        else:
            print(f"{device:6s} {med:10.1f} {mean:9.1f} {warm:10.1f}  {shape}")
    print("\n참고: 행동 묶음 1회 추론 시간입니다. 30fps 제어에서 한 스텝은 약 33ms이고, 묶음을 쓰면 매 스텝 추론하지 않아도 됩니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
