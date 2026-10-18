# 08. 문제 해결 · FAQ

## 1. 문제 해결표

| 증상 | 주요 원인 | 해결 |
|---|---|---|
| Studio에서 로봇·카메라가 안 보임 | 컨테이너를 켠 **뒤에** USB를 꽂음 | `docker compose down` → USB 확인 → `docker compose up -d` (`restart`로는 안 됨) |
| `permission denied` (docker 명령) | `docker` 그룹 미적용 | `sudo usermod -aG docker $USER` 후 **재로그인** |
| `Permission denied` (로봇 포트) | 권한 외에도 포트 점유 · 오래된 장치 · 모터 무응답 | 케이블 → 보드 전원 → 어댑터 → 모터 순으로 하나씩 확인. `dialout` 그룹 확인 |
| 모터 ID 설정 중 에러 | 전원 빠짐, 3핀 케이블 헐거움, 모터 2개 이상 연결됨 | 보드에 **그 모터 하나만** 연결됐는지, 전원·USB 확인. Waveshare는 점퍼 B 채널 |
| 팔로워가 리더를 안 따라옴 | 캘리브레이션 불량, 다른 팔의 캘리브레이션 사용 | 해당 팔 재캘리브레이션 |
| `wrist_roll`이 특정 각도에서 튐 | 두 팔 손목 방향이 캘리브레이션 전에 안 맞았음 | 두 팔 손목을 물리적으로 맞춘 뒤 재캘리브레이션 (범위 값을 손으로 고치면 악화) |
| `wrist_roll` 값이 0–4095 | 정상 (360° 회전 관절) | — |
| 카메라 미리보기가 다른 카메라 | 내장 웹캠과 혼동 | 장치를 하나씩 골라 미리보기로 확인 |
| "no video encoder available" | 백엔드 PyAV/FFmpeg 코덱 문제 | 로그 확인 후 강사에게. 이미지 버전 확인 |
| Studio 접속 안 됨 | 컨테이너 미실행, 포트 충돌 | `docker ps`, `docker compose logs --tail 100` |
| 이미지 pull이 매우 느림 / 실패 | 동시 다운로드, 프록시 | 순서대로 받기. 프록시 환경이면 `.env`의 `HTTP_PROXY`·`HTTPS_PROXY` |
| 첫 학습·추론이 몇 분간 멈춘 듯함 | SYCL 커널 최초 컴파일 | 기다리기. 이후 캐시됨(`SYCL_CACHE_PERSISTENT=1`) |
| 병합 명령 실패 | 학생 간 카메라 키 · 해상도 · fps · 관절 이름 불일치 | `scripts/check_datasets.py`로 비교 → 다른 데이터 제외 |
| Import dataset 실패 | zip이 LeRobot v2/v3 구조가 아님, 디스크 부족 | zip 최상위에 `meta/` `data/` `videos/` 확인, 디스크 1GB 이상 |
| `physicalai run`이 시작 안 됨 | Studio 세션이 같은 로봇을 잡고 있음 | Studio의 원격조작·녹화·추론 세션 모두 종료 |
| 추론 중 로봇이 엉뚱하게 움직임 | B70 PC의 runtime.yaml 사용, 캘리브레이션 불량 | **즉시 전원 차단** → 노트북 전용 runtime.yaml인지, 캘리브레이션 확인 |
| 장치 오류로 모델이 안 열림 | `device`가 B70 기준으로 남음 | runtime.yaml의 device를 노트북용으로 |
| 추론이 데이터만큼 잘 안 됨 | 작업대·조명·카메라 위치 차이 | 배치 통일 · 불량 에피소드 제거 · 에피소드 추가 |
| sudo 명령이 중간에 건너뜀 | 여러 줄을 한꺼번에 붙여넣어 비밀번호 입력 중 끊김 | 한 줄씩 실행 |
| 데이터가 전부 사라짐 | `docker compose down -v` 실행 | 복구 불가. **`-v` 금지** |

### 로그 보는 법

```bash
cd ~/physical-ai-studio/application/docker
docker compose logs -f --tail 100
```

질문할 때는 **화면 캡처 + 로그 마지막 20줄** 을 함께 보여 주세요. (M1에서 배운 "AI에게 정확히 질문하기"와 같은 원칙)

---

## 2. FAQ

### CUDA와 Intel XPU는 뭐가 다른가요?

**한 줄**: CUDA는 NVIDIA GPU 전용 개발 플랫폼이고, XPU는 PyTorch가 인텔 GPU를 부르는 장치 이름입니다. 쓰는 입장에서는 `"cuda"` → `"xpu"`로 바꾸는 정도입니다.

| 계층 | NVIDIA | Intel |
|---|---|---|
| PyTorch 장치 이름 | `cuda` | `xpu` |
| 확인 코드 | `torch.cuda.is_available()` | `torch.xpu.is_available()` |
| 저수준 프로그래밍 | CUDA | oneAPI · SYCL (드라이버 계층: Level Zero) |
| 수학 라이브러리 | cuBLAS, cuDNN | oneMKL, oneDNN |
| 추론 최적화 엔진 | TensorRT (NVIDIA GPU 전용) | **OpenVINO** (CPU · GPU · NPU 공통) |
| Studio Docker 프로필 | `--cuda` | `--xpu` |

```python
import torch
device = "xpu" if torch.xpu.is_available() else "cpu"   # NVIDIA라면 "cuda"
model.to(device)
```

PyTorch는 2.5 버전부터 인텔 GPU(XPU)를 공식 지원합니다. 별도 확장 없이 표준 PyTorch로 동작합니다.

### Intel XPU의 장점이 뭔가요?

- **하나의 툴킷으로 CPU·GPU·NPU** — OpenVINO로 변환한 모델 하나를 device만 바꿔 어디서든 실행. M2에서 비교해 본 그것입니다.
- **iGPU는 시스템 메모리를 공유** — 노트북 Arc iGPU는 16GB 시스템 메모리를 함께 쓰므로, 작은 외장 GPU보다 큰 모델을 올릴 수 있는 경우가 있습니다 (속도는 외장 GPU보다 느림).
- **NPU** — 저전력 상시 추론용 전용 가속기. 배터리·발열이 중요한 엣지 기기에 유리합니다.
- **이미 깔려 있는 하드웨어** — 대부분의 업무용 노트북·산업용 PC가 인텔 CPU+iGPU. 추가 GPU 없이 AI를 돌릴 수 있다는 것이 엣지 배포에서 큰 장점입니다.
- **개방형 생태계** — oneAPI·SYCL은 개방 표준 기반, OpenVINO·Anomalib·Physical AI Studio 모두 오픈소스.
- **학습–추론 같은 회사 스택** — B70(학습) → OpenVINO(추론) → 노트북 iGPU/NPU(배포)까지 한 흐름.

### 그럼 CUDA보다 좋은 건가요?

솔직하게 답합니다.

- **생태계 성숙도는 CUDA가 앞서 있습니다.** 일부 라이브러리(특정 커스텀 커널, 양자화 라이브러리 등)는 CUDA를 먼저 지원하고, 인텔 지원은 뒤따라오는 경우가 있습니다.
- 대신 **추론·엣지 배포** 에서는 OpenVINO가 강점이 분명하고, 학습 쪽도 PyTorch 공식 지원 이후 빠르게 따라오고 있습니다.
- 실무에서는 "학습은 어디서든, 배포는 현장 하드웨어에 맞게"가 일반적입니다. 이 과정에서 배운 **OpenVINO 변환·배포 역량은 학습을 어디서 했든 그대로 씁니다.**

### NPU가 있는데 왜 노트북에서 학습 안 하나요?

- NPU는 **추론 전용** 가속기라 학습(역전파)을 하지 않습니다.
- 노트북 iGPU로도 학습은 가능하지만, 반 전체 데이터로 ACT를 학습하면 수업 시간 안에 끝나기 어렵습니다.
- 그래서 **학습은 B70 GPU 서버, 추론은 노트북** 으로 역할을 나눴습니다. 이게 실제 산업 현장 구조입니다.

### B70은 어떤 GPU인가요?

인텔 **Arc Pro** 계열 워크스테이션용 외장 GPU입니다. 노트북 iGPU와 같은 Xe 아키텍처 계열이라 ** 같은 XPU 소프트웨어 스택**(드라이버, PyTorch XPU, OpenVINO)으로 동작합니다. 정확한 사양은 현장 PC에서 확인합니다.

```bash
ls /dev/dri/                # card*, renderD* 장치
clinfo -l                   # OpenCL로 보이는 장치 목록 (clinfo 설치 시)
```

### Physical AI Studio랑 LeRobot은 무슨 관계인가요?

- **LeRobot** (Hugging Face): 로봇 학습 라이브러리이자 ** 데이터셋 형식의 사실상 표준**.
- **Physical AI Studio** (Intel): LeRobot 형식을 그대로 쓰면서, 설정·녹화·학습·내보내기·추론을 **GUI로 묶고** OpenVINO 배포를 붙인 플랫폼. LeRobot 정책도 쓸 수 있습니다.
- 그래서 Studio에서 녹화한 데이터를 LeRobot으로 학습하거나, 반대로 가져오는 것이 가능합니다.

### 왜 업데이트가 이렇게 자주 되나요? 불안정하지 않나요?

인텔이 Physical AI 사업을 강화하며 **새로 공개한 플랫폼** 이라 개발이 매우 활발합니다 (거의 매일 커밋). 그래서

- 수업 기간에는 **버전을 고정**(이미지를 다시 받지 않음)하고,
- 버그를 만나면 **기록해 두는 것** 자체가 의미 있는 경험입니다. 실제로 GitHub 이슈로 제보할 수도 있습니다.

### 에피소드는 몇 개가 적당한가요?

Studio 문서는 학습 전 **최소 50개** 를 권장합니다. 우리는 1인 10개를 **반 전체로 합쳐서** 이 기준을 넘깁니다. 개인 프로젝트라면 작업 하나당 50개 이상을 목표로 하세요.

### SmolVLA나 Pi0.5는 왜 안 쓰나요?

- 모델이 커서 학습 시간이 길고, 노트북에서의 실시간 추론 부담이 큽니다.
- Hugging Face 토큰이 필요합니다.
- 경험자라면 최종 프로젝트에서 도전해 볼 수 있습니다 (B70 학습 시간을 강사와 상의).
