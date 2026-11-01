# RadWright Screw 배포하기

[English](RELEASING.md) | [한국어](RELEASING.ko.md)

RadWright Screw(내부 이름 Pedicle Screw Simulator)의 한 버전이 `main`에서 사용자 PC까지
가는 절차와, 더 넓게 배포하기 전에 아직 남은 일을 정리합니다. 빌드 세부 사항은
[데스크톱 빌드 안내](BUILDING_DESKTOP.ko.md)에 있습니다.

## 사용자가 받는 것

| 플랫폼 | 파일 | 설치 방식 | 업데이트 |
|:--|:--|:--|:--|
| Windows x64 + NVIDIA GPU | `PedicleScrewSimulator-<버전>-Windows-x64-Setup.exe` + `…-Setup-1.bin` | 사용자별, `%LOCALAPPDATA%\Programs\RadWright Screw`, 관리자 승인 없음 | 자동, 바뀐 레이어만 |
| Windows x64(포터블) | `PedicleScrewSimulator-<버전>-Windows-x64.zip` | 아무 곳에 압축 해제 | 없음(다시 내려받기) |
| macOS Apple Silicon | `PedicleScrewSimulator-<버전>-macOS-arm64.zip` | 압축 해제 후 우클릭 > 열기 | 없음 |

사용자에게 알릴 요구 사항: Windows 10/11 64비트, 최신 드라이버의 NVIDIA GPU(분할을 GPU로
실행), 앱용 약 6 GB와 첫 분할 때 내려받는 모델 가중치용 최대 1.4 GB의 여유 공간, 첫 실행과
업데이트를 위한 인터넷 연결.

## 배포 절차

1. **`main` 준비.** 두 changelog의 `[Unreleased]`를 새 버전 절로 옮기고, `VERSION`,
   `CITATION.cff`, README·USER_GUIDE·BUILDING_DESKTOP 쌍, `THIRD_PARTY_NOTICES.md`를
   올립니다(`tests/test_version.py`가 검사). `AGENTS.md`의 검사를 돌립니다.
2. **태그와 빌드.** `git tag -a vX.Y.Z -m "RadWright Screw X.Y.Z"` 후 태그를 push합니다.
   **Build desktop packages** workflow가 Windows(ZIP, 설치 프로그램, 업데이트 레이어)와
   macOS를 빌드하고 `--self-check`를 돌린 뒤 결과를 올립니다. Windows에서 직접
   `scripts\build_desktop.py`를 돌려도 같은 파일이 나옵니다.
3. **일반 사용자 권한으로 점검.** 관리자 권한이 *없는* Windows 계정에서 설치 → 시작 메뉴로
   실행 → 표본 CT 열기 → 분할 한 번 → 제거까지 해 봅니다. (RadWright Spine은 개발 PC가 늘
   관리자로 돌아서, 모든 사용자가 실행하지 못하는 빌드를 배포한 적이 있습니다.)
4. **R2에 게시.** 사용자가 받는 파일은 모두 R2에 둡니다. GitHub는 파일 하나를 2 GB로
   제한하고 설치 프로그램 `.bin`은 약 2.8 GB이기 때문입니다. macOS ZIP을 Windows 파일과 함께
   `release\`에 두고, Windows 빌드 PC에서 `python scripts\publish_update.py release\update
   --downloads <.exe, .bin, ZIP> --dry-run`을 실행합니다(정확한 명령은 빌드 안내에 있음).
   새로 올라갈 레이어가 예상과 같은지 보고 `--dry-run` 없이 실행합니다. 설치 프로그램과 ZIP은
   `radwright-screw/downloads/<버전>/`에 올라가고, 설치된 앱은 다음 실행 때 새 버전을 봅니다.
5. **GitHub Release 만들기.** `gh release create vX.Y.Z --verify-tag --notes-file <노트>`로
   릴리스 노트와 R2 다운로드 링크만 올립니다(파일은 붙이지 않음).
6. **공지.** 쉬운 말로 쓴 릴리스 노트: 사용자에게 무엇이 바뀌었는지, 연구·교육용이라는 점,
   사용자가 할 일이 있으면 그 내용.

## 되돌리기

R2에서 `radwright-screw/versions/<이전>/components.json`을
`radwright-screw/latest/components.json`에 덮어씁니다. 아직 업데이트하지 않은 앱은 이전
버전에 머뭅니다. 이미 업데이트한 앱은 새 버전을 유지하므로, 고친 버전을 더 높은 번호로
배포합니다.

## 더 넓게 배포하기 전에 (아직 안 함)

- **다운로드 페이지.** RadWright Spine에는 최신 버전, 다운로드 링크, 요구 사항, SmartScreen
  안내가 있는 사이트가 있습니다. RadWright Screw는 아직 없습니다. 만들면 R2의
  `radwright-screw/downloads/<버전>/`을 가리킵니다.
- **사용자 매뉴얼.** `docs/USER_GUIDE*.md`는 개발자용 참고서입니다. 설치, 첫 계획, grade 읽는
  법, 한계를 담은 사용자 매뉴얼은 배포 전에 작성하고 소유자가 검토합니다.
- **사용 조건.** RadWright Spine은 설치 프로그램과 첫 실행에서 EULA 동의를 받습니다.
  RadWright Screw는 연구용 표시만 있고 EULA가 없습니다. 연구실 밖 사람에게 주기 전에
  문구(연구·교육용, 의료기기 아님, 모든 결과는 사용자가 확인)를 정해야 합니다.
- **코드 서명.** 설치 프로그램과 앱에 서명이 없어서 처음 실행할 때 Windows SmartScreen이
  경고합니다(**추가 정보** > **실행**). 코드 서명 인증서가 있으면 경고가 사라집니다.
- **첫 R2 업로드.** `radwright-screw/` 아래에 아직 아무것도 올리지 않았습니다. 그 전까지
  업데이트 확인은 아무것도 찾지 못하고 조용히 넘어갑니다.
- **로고.** 완료: `packaging/icons/`(RadWright 계열, 축상 척추와 긴 척추경 스크류 두 개). 실행
  파일, 설치 프로그램, macOS 앱, 창에 들어가 있고, 워드마크는 다운로드 페이지와 매뉴얼에 쓸 수
  있습니다.
