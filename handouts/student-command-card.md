# 학생용 명령어 카드

> 한 줄씩 복사해서 실행하세요. 여러 줄을 한꺼번에 붙여넣지 마세요.

## 우리 반 통일 기준 (강사가 채움)

| 항목 | 값 |
|---|---|
| 프로젝트 이름 | |
| 카메라 이름 | |
| 해상도 / fps | 640×480 / 30 |
| 작업 문구 | |
| 내 데이터셋 이름 | `s___ _pick` |
| 공유폴더 | |

## Studio 켜고 끄기

```bash
cd ~/physical-ai-studio/application/docker
```

```bash
docker compose up -d
```

```bash
docker compose down
```

브라우저: <http://localhost:7860>

## 🔌 로봇·카메라 USB를 꽂거나 옮길 때

```text
1. docker compose down
2. USB 연결
3. docker compose up -d
```

`restart`로는 안 됩니다. **`down -v`는 절대 금지** (데이터 전부 삭제).

## 장치 확인

```bash
ls /dev/ttyACM* /dev/video*
```

```bash
docker ps
```

## 문제가 생기면 — 로그

```bash
docker compose logs --tail 50
```

## 추론 실행 (13일차)

```bash
docker cp ~/inference physical-ai-studio-xpu:/app/storage/inference
```

```bash
docker exec -it -u appuser physical-ai-studio-xpu bash
```

컨테이너 안에서:

```bash
cd /app/storage/inference
```

```bash
physicalai run --config runtime.yaml --run.duration_s=10
```

⚠️ 처음엔 10초. **한 손은 전원 스위치에.** 이상하면 바로 전원 차단.

## 녹화 단축키

| 키 | 동작 |
|---|---|
| → | Start episode / Accept |
| ← | Discard |
