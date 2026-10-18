# 02. Docker 기초와 Physical AI Studio 설치 (90–91차시)

## 학습 목표

- Docker의 이미지·컨테이너·볼륨 개념을 설명할 수 있다.
- 노트북에 Docker를 설치하고 Physical AI Studio를 XPU 모드로 실행한다.
- 컨테이너가 USB 장치(로봇·카메라)를 인식하는 원리와 주의점을 안다.

## 1. 왜 Docker인가 — 3분 개념

Physical AI Studio는 Python 백엔드, 웹 UI, PyTorch(XPU), OpenVINO, FFmpeg, 인텔 GPU 런타임까지 **수십 개 구성요소** 가 정확한 버전으로 맞물려야 돌아갑니다. 이걸 노트북마다 따로 설치하면 반나절이 걸리고, 노트북마다 조금씩 다르게 깨집니다.

Docker는 **"이미 다 설치된 실행 환경"을 통째로 받아서 실행** 하는 도구입니다.

| 용어 | 비유 | 이번 수업에서 |
|---|---|---|
| **이미지 (image)** | 설치가 끝난 프로그램의 스냅샷 (붕어빵 틀) | `physical-ai-studio-xpu:main` |
| **컨테이너 (container)** | 이미지를 실행한 것 (구워진 붕어빵) | `physical-ai-studio-xpu` |
| **볼륨 (volume)** | 컨테이너를 지워도 남는 저장 공간 | 데이터셋·모델·DB가 저장되는 `physical-ai-studio-storage` |
| **docker compose** | 여러 설정을 파일 하나(`docker-compose.yaml`)로 실행 | `docker compose up -d` 한 줄로 Studio 실행 |

> 한 줄 요약: **이미지를 받아(pull) 컨테이너로 실행(up)하고, 데이터는 볼륨에 남는다.**

## 2. 사전 확인

터미널(`Ctrl`+`Alt`+`T`)을 열고 한 줄씩 실행합니다.

```bash
docker --version
```

- 버전이 나오면 → [4. Studio 설치](#4-physical-ai-studio-설치)로 이동
- `command not found`가 나오면 → 3번 Docker 설치부터

```bash
ls /dev/dri/renderD*
```

`/dev/dri/renderD128` 같은 출력이 나와야 인텔 GPU를 쓸 수 있습니다. 없으면 강사에게 알립니다 (GPU 드라이버 문제).

## 3. Docker 설치 (Ubuntu 24.04)

Docker 공식 저장소로 설치합니다. **여러 줄을 한꺼번에 붙여넣지 말고 블록 단위로** 실행하세요. (sudo 비밀번호 입력 중에 다음 줄이 씹힐 수 있음)

### 3-1. 저장소 등록

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
```

```bash
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

### 3-2. 설치

```bash
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### 3-3. sudo 없이 쓰기 + 장치 권한

```bash
sudo usermod -aG docker,dialout,video,render,plugdev $USER
```

| 그룹 | 의미 |
|---|---|
| `docker` | sudo 없이 docker 명령 |
| `dialout` | 로봇 시리얼 포트(`/dev/ttyACM*`) |
| `video` | 카메라(`/dev/video*`) |
| `render` | 인텔 GPU(`/dev/dri/renderD*`) |
| `plugdev` | USB 장치 |

**로그아웃 후 다시 로그인** 해야 적용됩니다. (급하면 `newgrp docker`로 현재 터미널에만 적용)

### 3-4. 확인

```bash
docker run --rm hello-world
docker compose version
```

`Hello from Docker!`가 보이고 Compose 버전이 **v2.24.0 이상** 이면 성공입니다.

## 4. Physical AI Studio 설치

### 4-1. 코드 받기

```bash
cd ~
git clone https://github.com/open-edge-platform/physical-ai-studio.git
cd physical-ai-studio/application/docker
```

### 4-2. 설정 파일 만들기와 XPU 지정

```bash
cp .env.example .env
./setup-devices.sh --xpu
```

`setup-devices.sh --xpu`는 `.env`의 `COMPOSE_PROFILES`를 `xpu`로 바꾸고, 장치 그룹 권한을 점검합니다. 경고가 나오면 3-3의 그룹 추가가 빠진 것입니다.

```bash
grep COMPOSE_PROFILES .env
```

`COMPOSE_PROFILES=xpu`가 보여야 합니다. (`cpu`로 남아 있으면 GPU를 안 씀)

### 4-3. 이미지 받기 (오래 걸림)

```bash
docker compose pull
```

수 GB를 받습니다. 반 전체가 동시에 받으면 오래 걸리니, **받는 동안 다음 교안(03 SO-101 세팅)의 하드웨어 조립·배선을 먼저 진행** 합니다.

> **강사 지시가 있을 때만 pull 하세요.** 이미지 태그가 `main`이라 받는 시각에 따라 버전이 달라집니다. 반 전체·학습 PC가 같은 버전이어야 나중에 데이터와 모델을 주고받을 수 있습니다.

### 4-4. 로봇·카메라를 먼저 꽂고 실행

```bash
docker compose up -d
```

브라우저에서 <http://localhost:7860> 접속 → 프로젝트 목록 화면이 보이면 성공.

## 5. 반드시 기억할 규칙 — USB는 먼저 꽂고 켜기

컨테이너는 **시작될 때 있던 USB 장치만** 봅니다. Studio를 켠 뒤에 로봇이나 카메라를 꽂으면 화면에 안 나옵니다.

```bash
docker compose down
# 로봇·카메라 USB 연결
docker compose up -d
```

`docker compose restart`로는 해결되지 않습니다. 반드시 `down` → `up`.

2인 1로봇이므로 **로봇을 짝꿍 노트북으로 옮길 때마다** 이 순서를 지킵니다.

## 6. 자주 쓰는 Docker 명령

| 하고 싶은 일 | 명령 (`application/docker/`에서) |
|---|---|
| Studio 켜기 | `docker compose up -d` |
| Studio 끄기 | `docker compose down` |
| 실행 중인지 확인 | `docker ps` |
| 로그 보기 (문제 생겼을 때) | `docker compose logs -f --tail 100` |
| 컨테이너 안으로 들어가기 | `docker exec -it -u appuser physical-ai-studio-xpu bash` |
| 파일 넣기 | `docker cp ~/파일 physical-ai-studio-xpu:/app/storage/` |

> ⚠️ `docker compose down -v`는 **데이터셋·모델·DB를 전부 삭제** 합니다. `-v`는 절대 붙이지 마세요.

## 더 해보기 (경험자)

- `docker compose config`로 실제 적용되는 설정을 확인하고, `privileged: true`가 무엇을 의미하는지 찾아보기
- `docker volume ls`, `docker run --rm -v docker_physical-ai-studio-storage:/s alpine ls -la /s`로 볼륨 안을 들여다보기
- `docker-compose.yaml`의 비특권(non-privileged) 모드 주석을 읽고, 필요한 장치만 넘기는 설정 써 보기

## 체크포인트

- [ ] `docker run --rm hello-world` 성공
- [ ] `.env`에 `COMPOSE_PROFILES=xpu`
- [ ] <http://localhost:7860> 접속 성공
- [ ] "USB 먼저 꽂고 `up`" 규칙을 말로 설명할 수 있다
