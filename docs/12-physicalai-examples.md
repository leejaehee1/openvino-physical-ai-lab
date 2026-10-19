# 12. OpenVINO physicalai로 해 보는 예제 모음

> Studio 화면 밖에서 **카메라 · 로봇 · 정책 추론 · 제어 루프** 를 Python으로 직접 다뤄 보는 예제들입니다.
> 13일차의 `physicalai run` 한 줄이 내부에서 무엇을 하는지 이해하고, M6 프로젝트에서 부품처럼 가져다 쓰는 것이 목표입니다.
> 대상: 경험자, M6에서 Python으로 시스템을 묶을 조.

## 기준 버전과 검증 상태

| 항목 | 내용 |
|---|---|
| 패키지 | `physicalai` 0.3.0 (PyPI), OpenVINO 2026.4.1 / 문서는 GitHub `main` (2026-10-19) |
| 표시 | ✅ 이 저장소에서 실제로 실행해 확인 · 🔶 API(클래스·인자)까지 확인, 로봇·카메라가 있어야 실행 · ⚠️ 현장에서 키 이름 등 확인 필요 |
| 주의 | 개발이 매우 빠른 패키지입니다. `main`에는 있지만 0.3.0에는 없는 기능이 있습니다(예: `InferenceModel.close()`). 버전에 따라 이름이 다르면 `help(클래스)`로 확인합니다. |

### 설치와 실행 위치

```bash
# 노트북에 별도 가상환경으로 (Studio 컨테이너와 따로)
python3 -m venv ~/pai && source ~/pai/bin/activate
pip install physicalai               # 추론·런타임
pip install "physicalai[so101-plugin]"   # SO-101 드라이버 (scservo_sdk 포함)
```

- Studio 컨테이너 안에는 이미 설치되어 있습니다 (`docker exec -it -u appuser physical-ai-studio-xpu bash`).
- ⚠️ **로봇 포트는 한 프로그램만 잡을 수 있습니다.** 예제를 돌리기 전에 Studio의 원격조작·녹화·추론 세션을 모두 끄고, 컨테이너 밖에서 실행할 때는 `docker compose down`으로 Studio를 내립니다.

## 예제 지도

| # | 예제 | 로봇 | 카메라 | 상태 | 이어지는 곳 |
|:---:|---|:---:|:---:|:---:|---|
| 1 | 카메라 찾기와 한 장 찍기 | | ✔ | 🔶 | 04 카메라 등록 |
| 2 | 로봇 관절값 읽기 | ✔ | | 🔶 | 03 캘리브레이션 확인 |
| 3 | **디바이스별 추론 속도 비교** | | | ✅ | M2 디바이스 비교, 06 더 해보기 |
| 4 | 행동 묶음(action chunk) 직접 보기 | | | ✅ | 06 ACT 원리, 11 delta_timestamps |
| 5 | 녹화 데이터로 오프라인 검증 | | | ⚠️ | 05a 데이터 품질, 06 성능 평가 |
| 6 | Python으로 제어 루프 실행 (`physicalai run`의 속) | ✔ | ✔ | 🔶 | 06 추론 |
| 7 | 동기 vs 비동기 실행 | ✔ | ✔ | 🔶 | 07a 동적 환경 |
| 8 | Studio 없이 텔레오퍼레이션 | ✔✔ | | 🔶 | 04 텔레오퍼레이션 |
| 9 | 네트워크로 로봇 공유 (`robot serve`) | ✔ | | 🔶 | 2인 1로봇 운영 |
| 10 | INT8 가중치 압축 후 속도 비교 | | | 🔶 | M2 양자화, 10 경량화 |
| 11 | M6: 인식 결과로 정책 고르기 | ✔ | ✔ | 🔶 | 07 · 07a 판단 계층 |

---

## 예제 1. 카메라 찾기와 한 장 찍기 🔶

```python
from physicalai.capture import UVCCamera, discover_all

for cam_type, devices in discover_all().items():          # 연결된 카메라 전부
    for d in devices:
        print(cam_type, d.device_id, d.name)

with UVCCamera(device="/dev/video0", width=640, height=480, fps=30) as cam:
    frame = cam.read_latest()                               # 가장 최근 프레임
    print(frame.data.shape, frame.timestamp)               # (480, 640, 3), 단조 증가 시각
```

**확인 포인트**: 04에서 Studio에 등록한 카메라와 같은 장치인가? 해상도·fps가 반 기준(640×480, 30)과 같은가? 내장 웹캠은 보통 `/dev/video0`이므로 USB 카메라 번호를 잘 고릅니다.

---

## 예제 2. 로봇 관절값 읽기 🔶

```python
from physicalai.robot import SO101

robot = SO101(port="/dev/ttyACM0", calibration="calibration.json", role="follower")
robot.connect()
obs = robot.get_observation()
print(robot.joint_names)       # ['shoulder_pan', 'shoulder_lift', ...]
print(obs.joint_positions)     # 관절값 6개 (기본 단위: normalized)
robot.disconnect()
```

- 생성자: `SO101(port, calibration=None, baudrate=1_000_000, role="follower", unit="normalized", allow_uncalibrated=False)`.
- 저장소의 `examples/so101/read_joints.py`는 토크를 끈 채 값을 계속 보여 줍니다. **손으로 팔을 움직이며 값이 따라 변하는지** 보면 캘리브레이션 확인이 됩니다.
- ⚠️ 캘리브레이션 파일은 **그 팔의 것** 만 씁니다(03). Studio에 저장된 캘리브레이션을 쓰려면 노트북별 `runtime.yaml`에 들어 있는 값을 사용합니다.

---

## 예제 3. 디바이스별 추론 속도 비교 ✅ (로봇 없이)

인텔이 Hugging Face에 공개한 **테스트용 ACT 패키지**(`OpenVINO/act-fp16-ov`, 가중치는 무작위)로, 지금 바로 CPU·GPU·NPU 속도를 비교할 수 있습니다.

```bash
python3 scripts/bench_policy.py                    # 테스트용 ACT, CPU·GPU·NPU
python3 scripts/bench_policy.py --devices CPU,GPU  # 일부만
```

우리 반 모델로 하려면 export 폴더와 관찰값 모양을 지정합니다.

```bash
python3 scripts/bench_policy.py --model ~/inference/exports/openvino \
  --obs '{"images.front": [1, 3, 480, 640], "state": [1, 6]}'
```

**실제 실행 결과 (개발용 Mac CPU, 참고값)**

```text
디바이스  중앙값(ms)  평균(ms)  워밍업(ms)  행동 묶음 모양
CPU          43.7      45.3       50.0  (100, 7)
```

수업 노트북(Meteor Lake)에서 아래 표를 채워 봅니다.

| 디바이스 | 중앙값(ms) | 첫 실행(컴파일) 시간 | 메모 |
|---|---|---|---|
| CPU | | | |
| GPU (Arc iGPU) | | | 첫 실행에 커널 컴파일로 오래 걸림 |
| NPU | | | 지원 안 되면 오류 → 그것도 결과 |

> **생각해 볼 것**: 30fps 제어에서 한 스텝은 약 33ms입니다. 추론이 40ms 걸려도 ACT가 **100스텝 묶음** 을 한 번에 내놓기 때문에 매 스텝 추론할 필요가 없습니다. 이것이 action chunking이 엣지에서 유리한 또 하나의 이유입니다.

---

## 예제 4. 행동 묶음 직접 보기 ✅

```python
import numpy as np
from physicalai.inference import InferenceModel

model = InferenceModel.from_pretrained("OpenVINO/act-fp16-ov", device="CPU")
obs = {   # 이미지는 LeRobot 규칙: float32, [0, 1], (배치, 채널, 높이, 너비)
    "images.image":  np.random.rand(1, 3, 256, 256).astype(np.float32),
    "images.image2": np.random.rand(1, 3, 256, 256).astype(np.float32),
    "state": np.zeros((1, 8), dtype=np.float32),
}

chunk = model.predict_action_chunk(obs)   # (100, 7) — 앞으로 100스텝 × 행동 7개
model.reset()
a = model.select_action(obs)              # (7,)    — 지금 실행할 한 스텝
```

| 메서드 | 돌려주는 것 | 언제 |
|---|---|---|
| `predict_action_chunk` | 묶음 전체 (100, 행동 수) | 런타임이 큐에 넣고 꺼내 쓸 때 |
| `select_action` | 한 스텝 | 간단한 루프. 내부에서 묶음을 받아 하나씩 꺼내 줌 |
| `reset` | — | 새 에피소드 시작 때 큐·상태 비우기 |

**연결**: 11의 `delta_timestamps={"action": [i/30 for i in range(100)]}`로 꺼낸 `(100, 6)`과 같은 모양입니다. 학습 때 본 "미래 100스텝"을 추론 때 그대로 내놓는 것입니다. (테스트 모델은 LIBERO 시뮬레이션 형식이라 행동이 7개, SO-101 모델은 6개)

---

## 예제 5. 녹화 데이터로 오프라인 검증 ⚠️

로봇을 움직이기 **전에**, 녹화한 에피소드의 장면을 정책에 넣어 **사람이 한 동작(action)과 정책이 낸 동작이 얼마나 비슷한지** 봅니다. 위험 없이 모델이 제대로 배웠는지 1차로 확인하는 방법입니다(open-loop 평가).

```python
import numpy as np
from lerobot.datasets import LeRobotDataset
from physicalai.inference import InferenceModel

ds = LeRobotDataset("local/class_merged", root="~/class/merged/class_merged")
model = InferenceModel("~/inference/exports/openvino", device="CPU")

# ⚠️ 키 이름: 왼쪽 = 모델 입력 키(export의 manifest/metadata 확인), 오른쪽 = 데이터셋 키
KEYS = {"images.front": "observation.images.front", "state": "observation.state"}

ep = ds.meta.episodes[3]                                  # 에피소드 3
lo, hi = ep["dataset_from_index"], ep["dataset_to_index"]
errors = []
for idx in range(lo, hi, 10):                             # 10프레임마다
    s = ds[idx]
    obs = {k: s[v].numpy()[None].astype(np.float32) for k, v in KEYS.items()}
    model.reset()
    pred = model.select_action(obs)
    errors.append(np.abs(pred - s["action"].numpy()).mean())
print("평균 오차:", np.mean(errors))
```

- 오차가 **에피소드 초반에 크고 후반에 작다** → 시작 자세 불일치(05a 조건 ③)
- **특정 에피소드만 크다** → 그 에피소드가 불량이거나 학습 데이터와 장면이 다름
- ⚠️ 정규화는 export 패키지 안에서 처리되는지(전처리 포함 여부)를 manifest에서 확인합니다. 오차 단위가 이상하게 크면 정규화 문제를 먼저 의심합니다.

---

## 예제 6. Python으로 제어 루프 실행 — `physicalai run`의 속 🔶

13일차의 `physicalai run --config runtime.yaml`은 아래 코드와 같은 일을 합니다.

```python
from physicalai.capture import UVCCamera
from physicalai.inference import InferenceModel
from physicalai.robot import SO101
from physicalai.runtime import PolicySource, RobotRuntime

runtime = RobotRuntime(
    robot=SO101(port="/dev/ttyACM0", calibration="calibration.json"),
    action_source=PolicySource(model=InferenceModel("./exports/openvino", device="GPU")),
    fps=30,
    cameras={"front": UVCCamera(device="/dev/video2", width=640, height=480, fps=30)},
)

with runtime:                    # 연결 → (끝나면) 해제
    runtime.run(duration_s=10)   # ⚠️ 처음엔 10초, 한 손은 전원 스위치에
```

| 런타임이 매 틱(1/30초)마다 하는 일 |
|---|
| 1. 카메라 프레임 읽기 → 2. 로봇 관절값 읽기 → 3. 관찰값 딕셔너리 만들기 → 4. 행동 큐가 비면 정책 추론 → 5. 다음 행동을 로봇에 보내기 |

- `cameras`의 키(`front`)는 **학습 데이터의 카메라 이름과 같아야** 합니다. 04에서 이름을 통일한 이유가 여기서 다시 나옵니다.
- YAML로 쓰면 `RobotRuntime.from_config("runtime.yaml")`. `class_path` + `init_args` 형식이라 위 코드와 1:1로 대응됩니다.

---

## 예제 7. 동기 vs 비동기 실행 🔶

```python
from physicalai.runtime import AsyncExecution, PolicySource, SyncExecution

sync_src  = PolicySource(model=model, execution=SyncExecution())    # 추론이 끝날 때까지 루프가 기다림
async_src = PolicySource(model=model, execution=AsyncExecution())   # 추론은 뒤에서, 루프는 30fps 유지
```

| | 동기 (Sync) | 비동기 (Async) |
|---|---|---|
| 동작 | 큐가 비면 그 자리에서 추론하고 기다림 | 추론을 백그라운드 스레드에서, 제어 루프는 일정 주기 유지 |
| 장점 | 단순, 디버깅 쉬움 | 추론이 느려도 움직임이 끊기지 않음 |
| 관찰 포인트 | 묶음이 바뀌는 순간 **잠깐 멈칫** | 멈칫이 줄어듦 |

예제 3에서 잰 추론 시간이 33ms보다 길다면 비동기의 효과가 눈에 보입니다. 07a에서 본 **동적 환경**(판단하는 동안에도 세상이 움직임)을 직접 체험하는 실험입니다.

---

## 예제 8. Studio 없이 텔레오퍼레이션 🔶

```python
from physicalai.robot import SO101
from physicalai.runtime import RobotRuntime, TeleopSource

follower = SO101(port="/dev/ttyACM0", calibration="follower.json", role="follower")
leader   = SO101(port="/dev/ttyACM1", calibration="leader.json",   role="leader")

runtime = RobotRuntime(robot=follower, action_source=TeleopSource(leader=leader), fps=30)
with runtime:
    runtime.run(duration_s=30)
```

- **같은 제어 루프에 "행동을 어디서 가져오나"만 바꾼 것** 입니다. 예제 6의 `PolicySource`(정책) 자리에 `TeleopSource`(리더 팔)를 끼웠습니다.
- 이 구조 덕분에 M6에서 "평소엔 정책, 문제가 생기면 사람이 리더로 개입" 같은 전환을 설계할 수 있습니다.

---

## 예제 9. 네트워크로 로봇 공유 — `physicalai robot serve` 🔶

로봇이 꽂힌 노트북에서 로봇을 **서비스로 띄우고**, 다른 컴퓨터에서 이름으로 찾아 쓰는 기능입니다(Zenoh 기반).

```yaml
# serve.yaml (로봇이 연결된 노트북)
name: follower-arm-03
robot:
  class_path: physicalai.robot.SO101
  init_args:
    port: /dev/ttyACM0
    calibration: /path/to/calibration.json
    role: follower
allow_remote: false     # 기본값: 같은 컴퓨터에서만. 다른 컴퓨터에서 쓰려면 true
rate_hz: 100.0
```

```bash
physicalai robot serve --config serve.yaml   # 로봇 쪽
physicalai robot discover                    # 찾는 쪽
```

- 2인 1로봇 수업에서 "로봇은 한 노트북에 꽂아 두고, 짝꿍 노트북에서 정책을 돌린다"는 구성을 실험해 볼 수 있습니다. USB를 옮겨 꽂고 Docker를 내렸다 올리는 수고가 줄어듭니다.
- ⚠️ `allow_remote: true`는 **같은 망의 누구나 로봇을 움직일 수 있게** 됩니다. 교실 망에서는 강사 확인 후, 실험할 때만 켭니다.

---

## 예제 10. INT8 가중치 압축 후 속도 비교 🔶 (B70 · 강사)

학습 라이브러리(`physicalai-train`)로 export할 때 압축 훅을 붙이면 OpenVINO 모델 가중치가 INT8로 줄어듭니다.

```bash
pip install "physicalai-train[nncf]"
```

```python
from physicalai.export import compress_weights_openvino_int8_sym
from physicalai.policies import ACT

policy = ACT.load_from_checkpoint("checkpoints/best.ckpt")
policy.export("./exports_int8", backend="openvino",
              post_export_hooks=[compress_weights_openvino_int8_sym])
```

그다음 예제 3으로 `./exports`와 `./exports_int8`의 속도를, 예제 5로 오차를 비교합니다.

| | FP | INT8 |
|---|---|---|
| 모델 크기 | | |
| CPU 중앙값(ms) | | |
| GPU 중앙값(ms) | | |
| 오프라인 평균 오차 | | |

M2에서 YOLO에 했던 양자화 실험을 **로봇 정책** 에 그대로 해 보는 것입니다. 속도가 빨라진 대신 정확도가 얼마나 떨어지는지가 핵심 관찰 포인트입니다.

---

## 예제 11. M6: 인식 결과로 정책 고르기 🔶

07a의 층 구조(인식 → 판단 → 행동)를 코드로 옮긴 뼈대입니다.

```python
from physicalai.inference import InferenceModel
from physicalai.runtime import PolicySource, RobotRuntime

POLICIES = {
    "red":  InferenceModel("./exports/pick_red",  device="GPU"),
    "blue": InferenceModel("./exports/pick_blue", device="GPU"),
}

def detect_color(frame) -> str | None:
    """인식 계층 — M3의 YOLO(OpenVINO)로 큐브 색을 찾는다."""
    ...

def decide(color, hand_visible) -> str | None:
    """판단 계층 — 07a 6-2의 규칙. 안전 규칙이 먼저."""
    if hand_visible:
        return None
    return color

def act(name: str, robot, cameras):
    """행동 계층 — 고른 정책으로 정해진 시간만 실행."""
    rt = RobotRuntime(robot=robot, action_source=PolicySource(model=POLICIES[name]),
                      fps=30, cameras=cameras)
    with rt:
        rt.run(duration_s=15)
```

- 인식은 CPU·NPU, 정책은 GPU처럼 **층마다 디바이스를 나눠 배치** 하고, 예제 3으로 각각의 시간을 재서 발표 근거로 씁니다.
- 정책을 바꿀 때마다 `reset()`이 되도록 새 `PolicySource`를 만드는 것이 안전합니다.

---

## 더 보기

- OpenVINO physicalai — <https://github.com/openvinotoolkit/physicalai> (README의 Camera·Robot·Inference·Runtime 절, `examples/so101/`, `examples/tutorials/collect_train_deploy.ipynb`)
- Physical AI Studio 학습 라이브러리 — `library/docs/how-to/export/export_inference.md`
- 테스트용 ACT 모델 — <https://huggingface.co/OpenVINO/act-fp16-ov> (가중치 무작위, 테스트 전용)
