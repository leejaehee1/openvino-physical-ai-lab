# 06. ACT 정책 학습과 실물 로봇 추론 (101–104차시)

## 학습 목표

- ACT 정책이 어떻게 관측을 받아 동작 묶음(action chunk)을 출력하는지 설명할 수 있다.
- 수집(노트북) → 병합·학습(B70) → 배포·추론(노트북)의 전 주기를 직접 수행한다.
- 같은 모델을 CPU·GPU·NPU에서 돌려 보고 엣지 추론의 의미를 이해한다.

## 1. 전체 흐름

| 단계 | 장소 | 담당 | 방식 | Studio 기능 |
|---|---|---|---|---|
| ① 수집 | 학생 노트북 | 학생 | GUI | 녹화 · 에피소드 확인/삭제 ([05](05-data-collection.md)) |
| ② 전달 | 노트북 → PC | 학생 | GUI + 공유폴더 | Export dataset |
| ③ 병합 | B70 PC | 강사 | CLI | (Studio에 병합 기능 없음) → `lerobot-edit-dataset` |
| ④ 학습 | B70 PC | 강사 | GUI | Import dataset → Train model → OpenVINO 내보내기 |
| ⑤ 추론 | 학생 노트북 | 학생 | CLI 1줄 | Download runtime export → `physicalai run` |

③과 ⑤만 명령어인 이유: Studio의 학습은 **데이터셋 하나만** 받고 병합 메뉴가 없으며, 다른 PC에서 만든 모델을 Studio에 **다시 넣는 메뉴가 없습니다**. 대신 Studio가 공식 제공하는 runtime export를 `physicalai run`으로 실행하면 Studio 화면 밖에서 로봇 추론이 됩니다.

> Studio에는 노트북에서 학습 버튼을 누르면 GPU 서버에서 학습하는 **원격 학습** 기능도 있지만, GPU 1장에 노트북 수십 대의 학습이 줄을 서게 됩니다. 이 수업은 **반 전체 데이터로 한 번 학습해서 나눠 주는** 방식이라 원격 학습을 쓰지 않습니다.

## 2. ACT 원리 (학습 대기 시간에 설명)

**ACT = Action Chunking with Transformers** (Zhao et al., 2023, ALOHA 프로젝트)

```text
 카메라 이미지 ──▶ ResNet (이미지 특징 추출) ─┐
                                              ├─▶ Transformer 인코더 ─▶ Transformer 디코더 ─▶ 앞으로 k스텝의 관절 목표값
 현재 관절값 6개 ─────────────────────────────┘                                                (action chunk)
```

### 핵심 아이디어 1 — 한 번에 여러 스텝을 예측 (Action Chunking)

- 매 순간 다음 1스텝만 예측하면, 작은 오차가 쌓여서 데이터에서 본 적 없는 자세로 흘러갑니다 (**오차 누적, compounding error**).
- ACT는 **앞으로 수십 스텝을 한 묶음으로 예측** 합니다. 결정 횟수가 줄어 오차가 덜 쌓이고, 동작이 부드럽습니다.
- 사람의 시연은 중간에 멈칫하는 등 "1스텝 기준으로는 일관성이 없는" 부분이 많은데, 묶음 단위로 보면 의도가 더 잘 드러납니다.

### 핵심 아이디어 2 — 같은 작업도 사람마다 다르게 한다 (CVAE)

- 학습 때는 시연 궤적의 "스타일"을 잠재변수 z로 요약해 디코더에 넣어 줍니다 (조건부 VAE).
- 추론 때는 z를 평균값(0)으로 고정 → **가장 평균적인 방식** 으로 동작합니다.
- 우리 반처럼 **여러 사람이 시연한 데이터** 를 합칠 때 이 구조가 도움이 됩니다.

### 왜 엣지에 적합한가

- 수천만 파라미터 규모로 VLA(수억~수십억)보다 훨씬 작음 → 노트북 iGPU·NPU에서 실시간 추론 가능
- 언어 입력이 없어 "작업 하나"를 잘하는 데 집중 (여러 작업을 언어로 지시하려면 SmolVLA·Pi0.5)

### 생각해 볼 질문

- 우리 반 데이터(학생 수 × 10)가 1인 데이터(10개)보다 나은 점과 나쁜 점은? (양 vs 스타일 다양성·배치 차이)
- 카메라 위치가 조마다 3cm씩 다르면 모델에 어떤 영향이 있을까?

## 3. (강사) B70 PC — 데이터셋 병합

### 3-1. 학생 zip 모으기

공유폴더 `~/class/inbox/`에 `s01.zip`, `s02.zip` … 이 모였는지 확인합니다.

### 3-2. 병합 스크립트 실행

이 저장소의 [`scripts/merge_class_datasets.sh`](../scripts/merge_class_datasets.sh)가 아래를 한 번에 합니다.

1. zip을 `~/.cache/huggingface/lerobot/local/<학생번호>/`로 풀기
2. [`check_datasets.py`](../scripts/check_datasets.py)로 학생별 `meta/info.json`의 fps · 카메라 키 · 해상도 · robot_type 비교 → 다른 데이터는 알려 주고 중단
3. `lerobot-edit-dataset` 병합 (에피소드 번호 재부여, 메타·통계 재계산)
4. 결과를 `class_merged.zip`으로 압축

```bash
./scripts/merge_class_datasets.sh ~/class/inbox class_merged
```

스크립트가 내부에서 실행하는 병합 명령:

```bash
lerobot-edit-dataset \
  --repo_id local/class_merged \
  --operation.type merge \
  --operation.repo_ids "['local/s01', 'local/s02', 'local/s03']"
```

> ⚠️ **리허설 필수.** 로컬 폴더를 병합 대상으로 지정하는 방식은 실제 Export zip으로 아직 검증되지 않았습니다. lerobot이 Studio 컨테이너 안에 있는지, 별도 venv(`pip install lerobot==0.6.0`)가 필요한지도 현장에서 확인합니다.

### 3-3. Studio로 가져와 학습

1. B70 Studio(<http://localhost:7860>) → 학생과 같은 이름의 프로젝트 → Datasets → **Import dataset** → `class_merged.zip`
   (LeRobot v2/v3 자동 인식, 검토 화면에서 **사전에 만들어 둔 환경** 을 지정)
2. 에피소드 수가 **학생 수 × 10** 과 맞는지, 영상이 재생되는지 확인
3. Models → **Train model** → 데이터셋 `class_merged` · 정책 **ACT** · 장치 **XPU**
4. 진행은 Models 화면의 loss 곡선과 로그로 확인 (화면 연결이 끊겨도 학습은 계속됨)
5. 학습이 끝나면 PyTorch · OpenVINO · ONNX · ExecuTorch로 자동 내보내기

> B70 PC에도 **로봇 1대 + 카메라가 등록된 환경** 이 있어야 runtime export를 만들 수 있습니다. 실제 로봇을 연결할 필요는 없고, 카메라 이름만 학생과 똑같이 맞춥니다.

### 학습 중에 할 것 (대기 시간 활용)

- 2절 ACT 원리 설명
- 학생들이 자기 데이터를 열어 보는 [05의 "더 해보기"](05-data-collection.md#더-해보기-경험자)
- B70 화면을 프로젝터로 띄워 **loss 곡선이 내려가는 모습** 함께 보기
- 학습이 너무 오래 걸리면 → 중단해도 그 시점 체크포인트로 내보내기가 됩니다. 또는 **리허설 백업 모델** 사용.

## 4. (강사) 모델 배포

1. 학습된 모델 → **Download runtime export** → OpenVINO · 환경 · 장치 · 작업 문구 선택 → zip
2. 공유폴더로 배포

| 메뉴 | 내용 | 이 수업에서 |
|---|---|---|
| Download model | 체크포인트 + 내보낸 모델 전체 | 보관용 (노트북 Studio에 넣는 메뉴가 없음) |
| **Download runtime export** | OpenVINO 모델 + `runtime.yaml` + README | **이것을 사용** |

## 5. 학생 노트북 — 추론

### 5-1. zip 안의 runtime.yaml은 그대로 쓰면 위험

B70 PC에서 받은 `runtime.yaml`에는

- **B70 PC에 등록된 로봇의 포트와 캘리브레이션 값** 이 들어 있고 → 내 팔과 다른 값으로 움직임
- **device 값이 B70 기준** → 노트북에서 모델이 안 열림

그래서 강사가 미리 만들어 둔 **노트북 전용 `~/inference/runtime.yaml`** 을 쓰고, zip에서는 **모델 폴더만** 가져옵니다.
`runtime.yaml`은 모델을 `./exports/openvino` 상대경로로 찾기 때문에 모델 폴더만 바꿔 끼우면 됩니다.

> 노트북 전용 runtime.yaml 만드는 법(강사, 사전 1회): 각 노트북 Studio에서
> `http://localhost:7860/api/projects/<프로젝트ID>/environments/<환경ID>/runtime-config`로 그 노트북의 로봇·카메라 설정을 받고,
> runtime export의 `runtime.yaml`에서 로봇·카메라 부분만 교체(리더 항목 제외) + device를 노트북용으로 수정 → `~/inference/runtime.yaml`로 저장, 노트북·팔 번호로 라벨링.
> 작성법 참고: [openvinotoolkit/physicalai](https://github.com/openvinotoolkit/physicalai) `docs/how-to/config/write-runtime-config.md`

### 5-2. 실행

1. zip을 풀어 `exports/openvino` 폴더만 `~/inference/exports/`에 덮어쓰기 (zip 안의 runtime.yaml은 사용 안 함)
2. Studio 화면에서 로봇을 쓰는 세션(원격조작·녹화·추론)을 **모두 종료** — 같은 로봇을 두 곳에서 잡으면 잠금 때문에 시작되지 않습니다.
3. 컨테이너로 복사 후 실행

```bash
docker cp ~/inference physical-ai-studio-xpu:/app/storage/inference
docker exec -it -u appuser physical-ai-studio-xpu bash
```

컨테이너 안에서:

```bash
cd /app/storage/inference
physicalai run --config runtime.yaml --run.duration_s=10
```

### ⚠️ 첫 실행 안전 수칙

- 로봇 주변 물건을 치우고 **한 손은 전원 스위치에**.
- 처음엔 **10초** 만. 예상과 다르게 움직이면 **즉시 전원 차단** → 캘리브레이션·설정 파일 확인.
- 정상이면 `--run.duration_s=60`으로 늘립니다.

## 6. 성능 평가

조별로 **같은 조건에서 10회** 시도하고 기록합니다.

| 시도 | 물체 위치 | 집기 성공 | 상자 넣기 성공 | 비고 (실패 양상) |
|:---:|---|:---:|:---:|---|
| 1 | 표시 정중앙 | | | |
| 2 | 표시 +2cm 오른쪽 | | | |
| … | | | | |
| 성공률 | | /10 | /10 | |

토론 거리:

- 어떤 위치에서 실패가 많았나? 학습 데이터에 그런 위치가 있었나?
- 조명이나 배경이 바뀌면? (일부러 손전등을 비춰 보기)
- 실패 양상: 엉뚱한 곳으로 감 / 근처까지 가서 헛집음 / 집었는데 떨어뜨림 → 각각 무엇이 부족했을까?

## 더 해보기 (경험자) — OpenVINO 디바이스 비교

같은 모델을 `runtime.yaml`의 device만 바꿔 실행하며 반응 속도를 비교합니다. M2에서 했던 CPU·GPU·NPU 비교를 **로봇 정책** 으로 확장하는 실습입니다.

| device | 체감 반응 | 동작 부드러움 | 비고 |
|---|---|---|---|
| CPU | | | |
| GPU (Arc iGPU) | | | |
| NPU | | | 지원 여부 확인 |

> Meteor Lake의 GPU·NPU에 맞는 device 표기는 현장에서 확인 중입니다. 확인되면 이 표에 기록합니다.

## 체크포인트

- [ ] ACT의 "action chunking"이 왜 필요한지 한 문장으로 설명할 수 있다
- [ ] 내 노트북에서 `physicalai run`으로 10초 추론 성공
- [ ] 성공률 표 작성
- [ ] runtime.yaml을 그대로 쓰면 안 되는 이유 두 가지를 말할 수 있다
