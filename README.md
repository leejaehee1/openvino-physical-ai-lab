# OpenVINO Physical AI 실습 교안 (M5–M6)

OpenVINO™ 활용 온디바이스 AI 실무자 양성과정(120차시) 중 **M5 Physical AI 실습** 과 **M6 최종 프로젝트** 교안입니다.
Intel Physical AI Studio와 SO-ARM101 로봇 팔로 "노트북에서 모으고, B70 PC에서 학습하고, 다시 노트북에서 움직이는" 모방학습 전 주기를 다룹니다.

## 이번 주 진도

| 일차 | 모듈 | 차시 | 주제 | 교안 |
|---|---|---|---|---|
| 12일차 | M5 | 89–92 | 로봇 환경 구축 — Physical AI 개념, SO-ARM101 구조, 서보 ID, 캘리브레이션 | [01](docs/01-physical-ai-intro.md) · [02](docs/02-docker-and-studio-install.md) · [03](docs/03-so101-setup.md) |
| 12일차 | M5 | 93–96 | Studio 설정 — 환경 구성, 카메라 연동, 리더-팔로워 텔레오퍼레이션 | [04](docs/04-studio-environment-teleop.md) |
| 13일차 | M5 | 97–100 | 데이터 수집 — 에피소드 녹화, 품질 관리, LeRobot 데이터셋 | [05](docs/05-data-collection.md) · [05a 이론](docs/05a-data-collection-theory.md) |
| 13일차 | M5 | 101–104 | 이미테이션 러닝 — ACT 학습(B70), 실물 로봇 추론·평가 | [06](docs/06-act-train-and-inference.md) |
| 14–15일차 | M6 | 105–120 | 최종 프로젝트 — 기획, 인식 계층, 판단·행동 계층, 발표 | [07a 도입](docs/07a-agent-view-for-m6.md) · [07](docs/07-final-project.md) |
| 보충 | — | 하루 추가 시 | 이론 보강 — LLM은 어떻게 만들어지나, 데이터 수집 이론 | [09](docs/09-how-llms-work.md) · [05a](docs/05a-data-collection-theory.md) |

## 강의 슬라이드

- [12일차. 로봇 환경 구축과 Physical AI Studio 설정 (89–96차시)](https://claude.ai/artifact/NE1y1bGyrRK1a4UmWUDfct)
- [13일차. 데이터 수집 → ACT 학습 → 실물 로봇 추론 (97–104차시)](https://claude.ai/artifact/BdUTWUWwfG1cFAQHBKY1dv)

## 함께 보는 자료

- [강사 소개](docs/instructor.md) · [수강생 수준 진단](docs/diagnostic.md)
- 강사용: [00 운영 메모](docs/00-instructor-runbook.md) — 당일 시간표, 사전 점검, 수준 차이 대응
- 공통: [08 문제 해결 · FAQ](docs/08-troubleshooting-faq.md) — CUDA와 XPU 차이 등 자주 받는 질문 포함
- 배포용: [학생용 명령어 카드](handouts/student-command-card.md)

## 실습 환경

| 구분 | 사양 | 역할 |
|---|---|---|
| 학생 노트북 (1인 1대) | Ubuntu 24.04 · Intel Core Ultra 5 125H(Meteor Lake) · Arc iGPU · NPU · 16GB | 데이터 수집, 정책 추론 |
| 학습 PC (교실 1대) | Intel Arc Pro B70 GPU | 데이터 병합, ACT 학습, OpenVINO 내보내기 |
| 로봇 (2인 1대) | SO-ARM101 리더·팔로워 1세트 + USB 카메라 | 텔레오퍼레이션, 녹화, 추론 |
| 플랫폼 | [Intel Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio) (Docker, XPU 프로필) · LeRobot | GUI 기반 수집–학습–배포 |

## 전체 흐름

```text
[학생 노트북]                  [B70 PC · 강사]                         [학생 노트북]
 ① 에피소드 10개 녹화  ──zip──▶  ③ 데이터셋 병합 (CLI)  ──▶  ④ ACT 학습   ──zip──▶  ⑤ physicalai run
 (Studio GUI)          ② 전달     (lerobot-edit-dataset)       OpenVINO 내보내기       (OpenVINO 추론)
```

## 참고 자료

- Physical AI Studio — <https://github.com/open-edge-platform/physical-ai-studio>
- OpenVINO PhysicalAI 런타임 — <https://github.com/openvinotoolkit/physicalai>
- LeRobot SO-101 문서 — <https://huggingface.co/docs/lerobot/so101>
- LeRobot 데이터셋 v3 — <https://huggingface.co/docs/lerobot/lerobot-dataset-v3>
- ACT 논문 (Zhao et al., 2023) — <https://arxiv.org/abs/2304.13705>

> Physical AI Studio는 개발이 활발히 진행 중이라 메뉴 이름이나 동작이 자주 바뀝니다.
> 교안은 2026-10-18 `main`(커밋 `0874e5d`) 문서와 서큘러스 수업 준비 가이드(커밋 `b9975df` 기준)를 바탕으로 작성했습니다. 화면이 다르면 현장 버전을 우선합니다.
