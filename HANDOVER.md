# Handover — 2026-10-18 (Windows → Linux), 2026-10-31 갱신

다음 컴퓨터(Linux)에서 이어서 작업할 사람/에이전트를 위한 인수인계 메모입니다.
저장소 규칙은 `AGENTS.md`(Claude 는 `CLAUDE.md` → `@AGENTS.md`)가 우선합니다.

## 1. 현재 상태

- 최신 릴리스: **v0.2.5** (2026-10-02, 태그 `v0.2.5`). 그 뒤 작업은 아직 릴리스 전.
- `main` = `98f5008` (2026-10-30, `grotyx/reference-compare-surface-hu` 를 `--no-ff` merge, 푸시됨).
  - 들어간 기능: 기준 계획 비교, surface HU, undo/redo, Virtual X-ray, Screw MPR 가이드 핸들,
    post-op CT 스크류 추출(Tools > Extract Screws from Post-op CT). 내용은 `CHANGELOG*.md` `[Unreleased]`.
  - merge 전 테스트 2014 passed / 1 skipped, ruff clean.
- 2026-10-31 `grotyx/package-size-notes` merge: 데스크톱 빌드에서 Triton 제외
  (`packaging/PedicleScrewSimulator.spec`, `tests/test_packaging.py`), changelog 쌍, BUILDING_DESKTOP 쌍
  "Package size" 절, 이 문서. 테스트 2015 passed / 1 skipped.
- 남은 브랜치: `grotyx/review-fixes`(일부러 남김, 지우지 말 것), `grotyx/reference-compare-surface-hu`
  (merge 됨, 로컬·origin 에 남아 있음 — 삭제는 소유자 결정).

## 2. Linux 에서 시작하기

```bash
git clone https://github.com/grotyx/pedicle-screw-simulator.git
cd pedicle-screw-simulator
./scripts/run_app.sh            # venv 만들고 설치 후 실행
./scripts/run_app.sh --test     # 테스트
./scripts/run_app.sh --with-totalseg   # TotalSegmentator 포함 (GPU/가중치 다운로드)
```

직접 실행할 때 체크(AGENTS.md):

```bash
python -m pytest tests -p no:cacheprovider     # -q 를 더 붙이지 말 것(요약줄 사라짐)
python -m ruff check src tests scripts main.py
git diff --check
```

- 헤드리스 서버에서 테스트는 `QT_QPA_PLATFORM=offscreen`. 앱(VTK/OpenGL)은 디스플레이가 있어야 함.
- 표본 CT(`data/`)는 git 에 없음 — 직접 옮겨야 함. **에이전트는 `data/` 를 열거나 읽지 말 것**(`data/README.md` 제외).
- 데스크톱 패키지는 Windows/macOS 만 빌드함(Linux 패키지 없음, 소스 실행).

## 3. 핵심 설계 결정 (소유자 확정)

- 자동 계획은 실패 시 나사를 **만들어내지 않는다**.
- 천공 기본값 0 (Gertzbein **A only**). "Accepted breach" 옵션으로 Up to B / Up to C 허용. 내측(척추관) 천공은 어떤 경우도 불허.
- 좁은 척추경: 기본은 배치 안 함. "Place narrow screws even if not contained" 를 켤 때만.
- 재계획: "Replace planned screws?" 확인창, (level, side) 단위로 교체, 새 계획 성공 후에만.
- 골질 경고는 **Info** 등급 유지.
- 계획 JSON 에는 Series UID 원문 대신 `series_uid_sha256`. DICOM 오류 로그는 예외 타입만(경로 정규식 마스킹은 환자 폴더명이 새어서 폐기).
- 스크류 추출: head 는 항상 등쪽(LPS y 큰 쪽) 끝. 결과 grade 는 금속 인공물 때문에 참고용으로 표시.
  추출 스크류의 Ped. 열은 비워 둠(금속 있는 척추경 폭 분석은 신뢰 불가).
- **Rust 전면 재작성 안 함 (2026-10-30 검토).** 무거운 계산은 이미 numpy/VTK/SimpleITK(C/C++)가 하고,
  UI(PyQt6)·3D(VTK)를 Rust 로 대체할 성숙한 라이브러리가 없으며, 용량·속도는 TotalSegmentator(torch+CUDA)가
  좌우해서 얻는 게 거의 없음. 나중에 특정 함수가 느리면 그 함수만 PyO3 모듈로 바꾸는 방식만 고려
  (후보: 스크류 추출 반경별 EDT 0.4–7 s, Virtual X-ray 투영). 먼저 프로파일링으로 확인할 것.
- **분할은 GPU 전용.** CPU 전용 배포판 만들지 않음. **ONNX Runtime 전환 안 함**(nnU-Net 전·후처리 재구현 필요).
- UI: A+C 워크스테이션(좌측 단계 레일, 3D 메인 + 썸네일 3개, 도구 독은 이미지 밖 별도 행), 그라데이션 최소.

## 4. 교훈

- VTK 창(네이티브) 위 위젯은 반투명 불가, 이미지 위에 띄우면 방향 글자를 가림 → 도구는 이미지 밖에.
- 오프스크린 VTK `grab()` 은 검은 화면. 스크린 캡처는 다른 창(실제 CT)을 찍을 위험 → 캡처 시 맨 위 고정 + 즉시 확인/삭제.
- 전체 pytest 를 동시에 여러 개 돌리지 말 것(시그널 대기 플레이키).
- QMessageBox 질문은 `setText` 에(macOS 는 창 제목을 무시 → v0.2.4 macOS 빌드 실패 사례).
- 패키지 용량의 ~85% 는 torch+CUDA+가중치. `nccl`·`nvshmem`·`cusparselt` 는 `libtorch_cuda.so` 가 직접 링크해서
  뺄 수 없음(`readelf -d` 로 확인). Triton 만 빼도 됨(GPU 추론 실측 확인). 자세한 표는 `docs/BUILDING_DESKTOP.md` "Package size".
- TotalSegmentator 를 Python API 로 직접 부르는 스크립트는 `if __name__ == "__main__":` 필수(nnU-Net 작업자가 spawn).
- `Agent(isolation="worktree")` 는 main 에서 분기. 스택 작업은 `git worktree add -b <branch> <path> <base>`.

## 5. 릴리스 절차 (검증됨)

1. `[Unreleased]` → 새 버전 섹션으로 정리(한/영 changelog 쌍), 빈 `[Unreleased]` + compare 링크 유지.
2. `VERSION`, `CITATION.cff`(버전·날짜), README 쌍, `THIRD_PARTY_NOTICES.md`, BUILDING_DESKTOP 쌍, USER_GUIDE 쌍 갱신 (`tests/test_version.py` 가 검사).
3. 전체 체크 → `git merge -F <msg>` → `git push origin main && git tag -a vX -m "Pedicle Screw Simulator X" && git push origin vX`.
4. "Tests" 와 "Build desktop packages" 워크플로 확인 → `gh release create vX --verify-tag --notes-file <notes>`.

## 6. 작업 방식 (소유자 선호)

- 응답은 한국어.
- 멀티에이전트: 계획 Fable, 구현 Sonnet(medium) 또는 Muse(OpenCode), 리뷰·커밋 Opus, 어려운 부분 Codex(`gpt-5.6-sol`). 1~2 파일 수정·릴리스 잡무는 직접.
- 외부 에이전트(codex/opencode)는 헤드리스로 돌리지 말고 보이는 창(Orca 탭)으로. 임상 데이터 없는 깨끗한 worktree 에서만. 실데이터 검증은 코디네이터가 직접.
- Linux 에서 Orca CLI 는 `orca-ide`(bare `orca` 는 GNOME 화면낭독기일 수 있음).
- 스테이징은 경로 명시(`git add -A/-u/.` 금지), bare `git stash` 금지.

## 7. 나중에 할 수 있는 것 (요청된 것 아님)

- 실제 CT 에서 `scripts/compare_planning_defaults.py` 실행(이전 vs 현재 기본값 비교) — 소유자가 직접.
- USER_GUIDE 5.4 의 표본 수치는 0.2.4 이전 기본값 기준(본문에 날짜 표기됨) — 새로 측정해 갱신 가능.
- 남은 브랜치 `grotyx/review-fixes`, `grotyx/reference-compare-surface-hu` 정리 여부 결정.
- VTK 용량 축소: 소스 7곳의 `import vtk` 를 필요한 `vtkmodules.*` 로 바꾸면 565 MB → 약 100–200 MB 예상(미측정).
- 스크류 추출 남은 과제: 반경별 EDT 속도, op05 에서 8개 중 1개 놓침, ball-head entry 오차 약 2.85 mm.
- v0.2.6 릴리스(위 5번 절차).
