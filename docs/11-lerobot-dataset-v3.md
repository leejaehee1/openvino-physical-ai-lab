# 11. LeRobot v3 데이터셋 깊이 보기 — 구조와 다양한 에피소드 설계

> [05](05-data-collection.md)에서 zip을 한 번 열어 봤다면, 이 문서는 **파일 하나하나가 무엇을 담고, 서로 어떻게 연결되는지** 를 다룹니다.
> 후반부에서는 작업 목적에 따라 **에피소드를 어떻게 다르게 설계하는지** 와, 그것이 데이터셋에 **어떻게 남는지** 를 예제로 봅니다.
> 기준: LeRobot `main` 소스 코드(2026-10-19, 데이터셋 형식 `v3.0`). 공식 문서와 다른 부분은 소스 코드를 따랐습니다.
> 이 문서의 구조 설명, 코드, 명령은 LeRobot 0.6.1로 실제 v3 데이터셋을 만들어 병합·로딩까지 돌려 확인했습니다.

## 학습 목표

- v3 데이터셋의 세 기둥(표 데이터 · 영상 · 메타데이터)과 파일 경로 규칙을 설명할 수 있다.
- `info.json`, `stats.json`, `tasks.parquet`, `meta/episodes/`의 역할과 주요 필드를 안다.
- "에피소드 3번의 관절값과 영상이 어디에 있나"를 메타데이터만 보고 찾아갈 수 있다.
- 작업 목적(위치 다양화, 다중 작업, 다중 카메라, 복귀 시연 등)에 맞게 에피소드를 설계하고, 그 결과가 데이터셋 어디에 나타나는지 말할 수 있다.
- `lerobot-edit-dataset`으로 에피소드 삭제·분할·병합·작업 문구 수정을 할 수 있다.

---

## 1. v3의 핵심 아이디어 — "에피소드별 파일"에서 "큰 파일 + 목차"로

| | v2.1 (예전) | **v3.0 (지금)** |
|---|---|---|
| 저장 단위 | 에피소드 하나 = parquet 하나, 카메라마다 mp4 하나 | **여러 에피소드를 큰 파일 하나에 이어 붙임** |
| 에피소드 경계 | 파일 이름 (`episode_000003.parquet`) | **메타데이터의 시작·끝 위치** |
| 파일 수 | 에피소드 1만 개면 파일 수만 개 | 수백 MB 단위의 파일 몇 개 |
| 장점 | 단순 | 대규모에서 빠르고, Hub에서 바로 스트리밍 가능 |

> 핵심: **저장은 효율적으로(큰 파일 몇 개), 사용은 직관적으로(에피소드 단위)**. 그 사이를 이어 주는 것이 `meta/episodes/`의 "목차"입니다.

### 세 기둥

| 기둥 | 형식 | 담는 것 |
|---|---|---|
| **표 데이터** | Parquet | 관절값, 행동, 시간, 번호처럼 작고 자주 기록되는 값 |
| **영상** | MP4 (카메라별) | 카메라 프레임. 같은 카메라의 여러 에피소드가 한 파일에 |
| **메타데이터** | JSON · Parquet | 형식 정의, 통계, 작업 문구, 에피소드 목차 |

---

## 2. 폴더 구조와 경로 규칙

```text
s05_pick/
├── meta/
│   ├── info.json                         ← 형식 정의 (가장 먼저 읽는 파일)
│   ├── stats.json                        ← 정규화 통계
│   ├── tasks.parquet                     ← 작업 문구 ↔ 번호
│   └── episodes/
│       └── chunk-000/file-000.parquet    ← 에피소드 목차
├── data/
│   └── chunk-000/file-000.parquet        ← 모든 에피소드의 프레임 표
└── videos/
    └── observation.images.front/
        └── chunk-000/file-000.mp4        ← 이 카메라의 모든 에피소드 영상
```

경로는 `info.json`에 **템플릿** 으로 적혀 있습니다.

| 템플릿 | 값 |
|---|---|
| `data_path` | `data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet` |
| `video_path` | `videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4` |
| 에피소드 목차 | `meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet` |

### 파일은 언제 나뉘나

| 설정 (`info.json`) | 기본값 | 의미 |
|---|---|---|
| `data_files_size_in_mb` | 100 | parquet 파일 하나가 이 크기를 넘으면 다음 `file-001`로 |
| `video_files_size_in_mb` | 200 | mp4 파일 하나가 이 크기를 넘으면 다음 파일로 |
| `chunks_size` | 1000 | 폴더(chunk) 하나에 들어가는 최대 파일 수 |

우리 수업처럼 1인 10개(약 2–3분 분량)면 **data·video 모두 `file-000` 하나** 에 다 들어갑니다. 반 전체를 병합하면 영상이 200MB를 넘어 `file-001`이 생길 수 있습니다.

---

## 3. `meta/info.json` — 데이터셋의 설계도

SO-101 + 전면 카메라 1대로 녹화한 데이터셋의 예입니다(값은 예시).

```json
{
  "codebase_version": "v3.0",
  "robot_type": "so101_follower",
  "fps": 30,
  "total_episodes": 10,
  "total_frames": 4500,
  "total_tasks": 1,
  "chunks_size": 1000,
  "data_files_size_in_mb": 100,
  "video_files_size_in_mb": 200,
  "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
  "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
  "splits": {"train": "0:10"},
  "features": {
    "action":            {"dtype": "float32", "shape": [6], "names": ["shoulder_pan.pos", "shoulder_lift.pos", "..."]},
    "observation.state": {"dtype": "float32", "shape": [6], "names": ["shoulder_pan.pos", "shoulder_lift.pos", "..."]},
    "observation.images.front": {"dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channels"], "info": {"video.fps": 30, "video.codec": "..."}},
    "timestamp":     {"dtype": "float32", "shape": [1]},
    "frame_index":   {"dtype": "int64",   "shape": [1]},
    "episode_index": {"dtype": "int64",   "shape": [1]},
    "index":         {"dtype": "int64",   "shape": [1]},
    "task_index":    {"dtype": "int64",   "shape": [1]}
  }
}
```

### `features` 읽는 법

| 키 | 누가 채우나 | 뜻 |
|---|---|---|
| `observation.state` | 녹화 프로그램 (팔로워) | 지금 로봇 자세 |
| `observation.images.<카메라>` | 녹화 프로그램 (카메라) | `dtype: video` → 실제 픽셀은 parquet가 아니라 **mp4에** |
| `action` | 녹화 프로그램 (리더) | 사람이 시킨 동작 = 정답 |
| `timestamp` · `frame_index` · `episode_index` · `index` · `task_index` | **자동** | 시간, 에피소드 안 번호, 에피소드 번호, 전체 번호, 작업 번호 |

- `names`는 관절 이름 순서입니다. 병합할 데이터끼리 **순서까지 같아야** 합니다.
- `index`는 데이터셋 전체에서 몇 번째 프레임인지, `frame_index`는 에피소드 안에서 몇 번째인지입니다.
- 카메라 해상도·fps가 학생마다 다르면 `features`가 달라져 병합이 실패합니다 (`scripts/check_datasets.py`가 이것을 비교).
- 영상 `info`의 `video.codec`을 보면 압축 방식을 알 수 있습니다. LeRobot 0.6의 기본값은 **AV1** 이라 OpenCV로는 안 열릴 수 있습니다. 이때는 PyAV(`pip install av`)로 엽니다.

---

## 4. `meta/episodes/` — 에피소드 목차 (가장 중요)

에피소드마다 한 행이 있고, **"이 에피소드의 데이터와 영상이 어느 파일의 어디부터 어디까지인가"** 를 적어 둡니다.

| 열 | 예 (에피소드 3) | 뜻 |
|---|---|---|
| `episode_index` | 3 | 에피소드 번호 |
| `tasks` | `["Pick up the red cube…"]` | 이 에피소드의 작업 문구 |
| `length` | 450 | 프레임 수 |
| `data/chunk_index`, `data/file_index` | 0, 0 | 프레임 표가 있는 parquet 파일 |
| `dataset_from_index`, `dataset_to_index` | 1350, 1800 | 전체 프레임 번호(`index`) 기준 **시작(포함)·끝(미포함)** |
| `videos/observation.images.front/chunk_index`, `.../file_index` | 0, 0 | 이 카메라 영상이 있는 mp4 파일 |
| `videos/observation.images.front/from_timestamp`, `.../to_timestamp` | 45.0, 60.0 | **그 mp4 안에서** 몇 초부터 몇 초까지 |
| `meta/episodes/chunk_index`, `meta/episodes/file_index` | 0, 0 | 이 행 자체가 들어 있는 목차 파일 |
| `stats/<feature>/<min·max·mean·std·count·q01…q99>` | … | 이 에피소드만의 통계 |

### 직접 찾아가 보기 — "에피소드 3의 13번째 프레임"

```text
1) meta/episodes 에서 episode_index == 3 인 행을 찾는다
2) 관절값:  data/chunk-000/file-000.parquet 의 index == 1350 + 13 = 1363 인 행
3) 영상:    videos/observation.images.front/chunk-000/file-000.mp4 의
            45.0 + 13/30 ≈ 45.43초 위치 프레임
```

이것이 **"파일 경계가 아니라 메타데이터로 에피소드를 복원한다"** 는 v3의 원리입니다. 그래서 학생 zip을 그냥 복사하면 안 되는 것입니다. 모두 `dataset_from_index = 0`부터 시작하고 같은 파일 이름을 쓰기 때문에, 병합 도구가 이 숫자들을 **다시 계산** 해 줘야 합니다.

> 실습: [`scripts/explore_dataset.py`](../scripts/explore_dataset.py)가 이 목차를 표로 보여 주고, 특정 에피소드의 관절값을 CSV로 저장하거나 영상 프레임을 이미지로 뽑아 줍니다(8절).

---

## 5. `meta/tasks.parquet`와 `meta/stats.json`

### tasks.parquet — 작업 문구 사전

| task (인덱스) | task_index |
|---|---|
| Pick up the red cube and place it in the box | 0 |

- 프레임마다 문구를 반복 저장하지 않고 `task_index` 번호만 저장합니다. M4·09에서 본 **어휘사전과 같은 발상** 입니다.
- 한 데이터셋에 작업이 여러 개면 행이 늘어납니다(7절 예제 ③).
- 공식 문서 일부에는 `tasks.jsonl`로 적혀 있지만, 현재 코드는 **`tasks.parquet`** 를 씁니다.

### stats.json — 정규화 통계

```json
{
  "observation.state": {"mean": [...], "std": [...], "min": [...], "max": [...], "count": [4500],
                        "q01": [...], "q10": [...], "q50": [...], "q90": [...], "q99": [...]},
  "action": { ... },
  "observation.images.front": { ... }
}
```

- 정책은 학습·추론 때 이 값으로 입력·출력을 정규화합니다. 병합 후 **반드시 다시 계산** 해야 하는 이유입니다(05a 6-3).
- 분위수(q01–q99)는 이상치에 덜 민감한 정규화에 쓰입니다.
- 에피소드별 통계는 `meta/episodes`의 `stats/...` 열에, 전체 통계는 이 파일에 있습니다.

---

## 6. Python에서 불러 쓰기

### LeRobot으로

```python
from lerobot.datasets import LeRobotDataset

ds = LeRobotDataset("local/s05_pick", root="~/peek/s05_pick")   # 로컬 폴더 지정
print(ds.meta.fps, ds.meta.total_episodes, ds.num_frames)

sample = ds[100]                      # 100번째 프레임 → 텐서 딕셔너리
print(sample["observation.state"], sample["action"])
print(sample["observation.images.front"].shape)   # (3, 480, 640)
```

### `delta_timestamps` — 시간 창으로 꺼내기

```python
# 지금 프레임 기준으로, 행동은 앞으로 100스텝(30fps 기준 약 3.3초)을 한 번에
fps = 30
delta_timestamps = {"action": [i / fps for i in range(100)]}
ds = LeRobotDataset("local/s05_pick", root="~/peek/s05_pick", delta_timestamps=delta_timestamps)
print(ds[100]["action"].shape)        # (100, 6)  ← ACT가 배우는 action chunk 모양
```

ACT의 "앞으로 k스텝을 한 번에 예측"은 데이터셋 쪽에서 이렇게 **`action`을 미래 시간 창으로 묶어서** 구현됩니다. 에피소드 끝을 넘어가는 자리는 마지막 값으로 채워지고, 어느 칸이 채운 값인지가 `action_is_pad`(True/False 100개)로 함께 나옵니다. 학습할 때는 이 칸을 손실 계산에서 뺍니다.

### pandas로 (가볍게 볼 때)

```python
import pandas as pd
eps = pd.read_parquet("~/peek/s05_pick/meta/episodes/chunk-000/file-000.parquet")
print(eps[["episode_index", "length", "dataset_from_index", "dataset_to_index"]])
```

> Physical AI Studio의 학습 라이브러리(`physicalai-train`)도 같은 형식을 읽습니다: `LeRobotDataModule(repo_id=..., root=..., episodes=[...], delta_timestamps=...)`.

---

## 7. 다양한 에피소드 설계 — 목적에 따라 무엇을 바꾸고, 어디에 남나

같은 SO-101이라도 **무엇을 배우게 할지** 에 따라 에피소드를 다르게 모읍니다. 각 설계가 데이터셋의 어느 부분을 바꾸는지 함께 봅니다.

| # | 설계 | 목적 | 이렇게 녹화 | 데이터셋에 남는 곳 |
|:---:|---|---|---|---|
| ① | **기본** | 한 위치에서 한 작업 | 같은 시작 자세, 같은 물체 위치 | 작업 1개, 에피소드 길이 비슷 |
| ② | **위치 다양화** | 물체를 보고 찾아가기 | 표시 안에서 1–2cm씩 이동, 위치를 녹화 일지에 기록 | 데이터 형식은 같고 **영상 내용** 만 다름. 녹화 일지가 없으면 위치를 알 수 없음 → 일지 필수 |
| ③ | **다중 작업** | 문구에 따라 다른 동작 (SmolVLA 같은 언어 조건 정책) | "빨간 큐브를 왼쪽 상자에" 10개 + "파란 큐브를 오른쪽 상자에" 10개, **문구를 정확히 구분** | `tasks.parquet` 행 2개, 프레임의 `task_index`가 0/1. ACT는 문구를 입력으로 쓰지 않으므로 **ACT에는 섞지 않음** |
| ④ | **다중 카메라** | 가림에 강하게, 그리퍼 근처 정밀도 | 전면 + 손목 카메라 | `features`에 `observation.images.gripper` 추가, `videos/` 폴더 2개, 목차에 카메라별 열 |
| ⑤ | **복귀 시연** | 벗어난 상태에서 돌아오기 (05a 4절) | 일부러 살짝 빗나간 뒤 다시 집어 성공 | 형식은 같음. 별도 데이터셋(`s05_recovery`)으로 모았다가 **병합** 하면 비율 조절이 쉬움 |
| ⑥ | **실패 모음** | 성공/실패 분류기, 품질 분석용 | 실패도 Accept, 대신 별도 데이터셋 | 정책 학습에는 **넣지 않음**. 작업 문구를 `failed: ...`로 따로 두면 구분 쉬움 |
| ⑦ | **긴 작업 쪼개기** | 여러 단계 작업 | "집기"와 "놓기"를 각각 에피소드로, 또는 한 에피소드 + 단계별 문구 | 에피소드가 짧아지고 작업 수가 늘어남. M6 판단 계층이 단계 순서를 관리 |
| ⑧ | **사람별 데이터셋** | 사람 단위 검증 (05a 7-1) | 학생마다 별도 데이터셋 (지금 수업 방식) | 병합 전 `s01…s20`, 병합 후 `episode_index`로만 남으므로 **병합 전에 학생-에피소드 대응표를 기록** |

### 설계를 바꿀 때 지킬 것

- **형식(features)은 바꾸지 말고 내용만 바꾼다.** 카메라를 추가하는 ④는 형식 자체가 달라지므로, 기존 데이터셋과 **병합할 수 없습니다**. 처음부터 정하고 시작합니다.
- **작업 문구는 실험 설계의 일부** 입니다. ③처럼 문구로 작업을 구분할 계획이면 반 전체가 정확히 같은 문장을 씁니다. 반대로 ACT 하나만 학습할 거면 문구 하나로 통일합니다.
- **나중에 나눌 수 있게 기록한다.** 위치, 녹화자, 성공 여부 같은 정보는 데이터셋 안에 자동으로 남지 않습니다. 녹화 일지(05a 8절)에 에피소드 번호와 함께 적습니다.

### M6 프로젝트별 추천 설계

| M6 주제 | 추천 설계 |
|---|---|
| ★ 물체 집어 옮기기 | ① + ② (+ 소량 ⑤) |
| ★★ 색깔별 분류 | 위치별로 정책 2개(각각 ① + ②) 또는 ③ + VLA 도전 |
| ★★ 불량품 골라내기 | "치우기" 동작만 ① + ②. 불량 판정은 Anomalib(M3)이 담당 |
| ★★★ 말로 시키는 로봇 | 동작 단위로 ⑦처럼 쪼갠 정책 여러 개 + LLM이 순서 결정 |

---

## 8. 실습 — 데이터셋 들여다보기

[`scripts/explore_dataset.py`](../scripts/explore_dataset.py)는 pandas만으로(영상 프레임 추출은 OpenCV 추가) 데이터셋을 탐색합니다.

```bash
# 요약: info.json 핵심, 작업 목록, 에피소드 목차
python3 scripts/explore_dataset.py ~/peek/s05_pick

# 에피소드 3의 관절값·행동을 CSV로 (엑셀로 그래프 그려 보기)
python3 scripts/explore_dataset.py ~/peek/s05_pick --episode 3 --csv ep3.csv

# 에피소드 3의 13번째 프레임 영상을 이미지로 (4절의 계산을 코드로)
python3 scripts/explore_dataset.py ~/peek/s05_pick --episode 3 --frame 13 --png ep3_f13.png
```

| 확인할 것 | 질문 |
|---|---|
| 에피소드 목차 | `dataset_from_index`가 앞 에피소드의 `dataset_to_index`와 이어지나? |
| 영상 구간 | `from_timestamp`와 `to_timestamp`의 차이 × fps가 `length`와 같은가? |
| CSV | `action`이 `state`보다 앞서 움직이나? (추종 지연) |
| PNG | 13번째 프레임이 녹화 화면과 같은 장면인가? |

품질 점검은 [`scripts/analyze_episodes.py`](../scripts/analyze_episodes.py)(05a 7절)로 이어집니다.

---

## 9. 데이터셋 편집 — `lerobot-edit-dataset`

Studio 화면에서 못 하는 편집은 LeRobot 명령으로 합니다. **원본은 그대로 두고 `--new_root`로 새 폴더에 결과를 만드는 습관** 을 들입니다.

| 연산 (`--operation.type`) | 하는 일 | 예 |
|---|---|---|
| `info` | 요약 출력 | 형식 확인 |
| `delete_episodes` | 특정 에피소드 삭제 | 불량 에피소드 제거 |
| `split` | 비율 또는 번호로 나누기 | 학습/검증 분리 (에피소드 단위!) |
| `merge` | 여러 데이터셋 합치기 | 반 전체 병합 |
| `modify_tasks` | 작업 문구 바꾸기 | 오타 수정, 에피소드별 문구 지정 |
| `remove_feature` | 특정 키 삭제 | 안 쓰는 카메라 제거 |
| `recompute_stats` | 통계 다시 계산 | 편집 후 정규화 통계 갱신 |

### 자주 쓰는 형태

```bash
# 불량 에피소드 2, 7 삭제 → 새 폴더에 저장
lerobot-edit-dataset \
  --repo_id local/s05_pick --root ~/peek/s05_pick \
  --new_repo_id local/s05_pick_clean --new_root ~/peek/s05_pick_clean \
  --operation.type delete_episodes \
  --operation.episode_indices "[2, 7]"

# 학습 80% / 검증 20% (에피소드 단위로 나뉨)
lerobot-edit-dataset \
  --repo_id local/class_merged --root ~/class/merged/class_merged \
  --new_root ~/class/split \
  --operation.type split \
  --operation.splits '{"train": 0.8, "val": 0.2}'

# 로컬 폴더 여러 개 병합
lerobot-edit-dataset \
  --new_repo_id local/class_merged --new_root ~/class/merged/class_merged \
  --operation.type merge \
  --operation.repo_ids "['s01', 's02', 's03']" \
  --operation.roots "['/home/me/class/src/s01', '/home/me/class/src/s02', '/home/me/class/src/s03']"

# 작업 문구 오타 고치기
lerobot-edit-dataset \
  --repo_id local/s05_pick --root ~/peek/s05_pick \
  --operation.type modify_tasks \
  --operation.task_replacements '{"Pick up the red cub": "Pick up the red cube"}'
```

> ⚠️ 위 형태는 LeRobot `main`(2026-10) 기준입니다. Studio 컨테이너 안의 LeRobot 버전이 다르면 옵션 이름이 다를 수 있으니, 먼저 `lerobot-edit-dataset --help`로 확인합니다. 병합은 [`scripts/merge_class_datasets.sh`](../scripts/merge_class_datasets.sh)가 위 `merge` 형태로 처리합니다.

---

## 10. 자기점검

1. v3가 에피소드마다 파일을 따로 만들지 않고 큰 파일에 이어 붙이는 이유는?
2. `observation.images.front`의 실제 픽셀은 parquet와 mp4 중 어디에 있나요?
3. 에피소드 5의 영상이 mp4의 어디에 있는지 알려면 어떤 파일의 어떤 열을 보나요?
4. 병합할 때 `dataset_from_index`를 다시 계산해야 하는 이유는?
5. 다중 작업 데이터(③)를 ACT 학습에 섞으면 왜 문제가 되나요?
6. 손목 카메라를 중간에 추가한 데이터셋을 기존 데이터셋과 병합할 수 없는 이유는?
7. `delta_timestamps={"action": [i/30 for i in range(100)]}`로 꺼낸 `action`의 모양은? 이것은 ACT의 무엇에 해당하나요?

<details>
<summary>모범 답안</summary>

1. 파일 수가 너무 많아지면 열고 관리하는 비용이 커집니다. 큰 파일 몇 개로 모으면 빠르고, Hub에서 바로 스트리밍할 수도 있습니다. 에피소드 경계는 메타데이터로 복원합니다.
2. mp4입니다. `info.json`에서 `dtype: video`로 표시되고, parquet에는 픽셀이 없습니다.
3. `meta/episodes/`의 parquet에서 `episode_index == 5`인 행의 `videos/observation.images.front/chunk_index`, `file_index`, `from_timestamp`, `to_timestamp` 열을 봅니다.
4. 학생마다 데이터셋이 0번 프레임부터 시작하고 같은 파일 이름을 쓰기 때문입니다. 합친 뒤에는 전체 기준 위치가 달라지므로 목차를 새로 계산해야 합니다.
5. ACT는 작업 문구를 입력으로 받지 않습니다. 같은 장면에서 문구에 따라 다른 동작이 정답이 되면, 정책 입장에서는 같은 입력에 정답이 두 개인 다봉성 문제가 됩니다.
6. `features`(카메라 키)가 달라져 형식이 맞지 않기 때문입니다.
7. (100, 6)입니다. 앞으로 100스텝 × 관절 6개로, ACT가 한 번에 예측하는 action chunk에 해당합니다.

</details>

## 참고

- LeRobot 데이터셋 v3 문서 — <https://huggingface.co/docs/lerobot/lerobot-dataset-v3>
- LeRobot 소스 — `src/lerobot/datasets/` (경로 상수, 메타데이터 저장), `src/lerobot/scripts/lerobot_edit_dataset.py` (편집 명령)
