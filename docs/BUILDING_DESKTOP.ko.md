# macOS·Windows 단독 실행 패키지 만들기

[English](BUILDING_DESKTOP.md) | [한국어](BUILDING_DESKTOP.ko.md)

## 지원하는 결과물

| 플랫폼 | 결과물 | 빌드 환경 |
|---|---|---|
| macOS Apple Silicon | ZIP 안의 `.app` | macOS arm64 |
| macOS Intel | ZIP 안의 `.app` | macOS Intel |
| Windows x64 | ZIP 안의 `.exe`와 지원 폴더 | Windows x64 |

PyInstaller는 교차 컴파일러가 아니므로 각 운영체제용 패키지는 해당 운영체제에서 별도로 만들어야 합니다.

## macOS에서 직접 빌드

```bash
./scripts/run_app.sh --setup-only
source venv/bin/activate
python -m pip install -r requirements-dev.txt
python scripts/build_desktop.py
```

결과 압축파일은 `release/`에 생성됩니다. 현재 앱에는 Apple Developer 서명과 공증이 적용되지 않으므로 최초 실행 시 Gatekeeper 경고가 나타날 수 있습니다. 앱을 우클릭하고 **Open**을 선택하면 실행할 수 있습니다.

## Windows에서 직접 빌드

Python 3.12가 설치된 PowerShell에서 실행합니다.

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\venv\Scripts\python.exe scripts\build_desktop.py
```

압축파일에는 `PedicleScrewSimulator.exe`와 실행에 필요한 `_internal` 폴더가 포함됩니다. 두 항목을 분리하거나 일부 파일만 이동하면 안 됩니다.

## GitHub Actions 자동 빌드

**Build desktop packages** workflow는 다음 환경에서 각각 빌드합니다.

- `windows-2025`: Windows x64
- `macos-15`: macOS Apple Silicon
- `macos-15-intel`: macOS Intel

GitHub Actions 탭에서 수동으로 실행하거나 `v0.1.0` 같은 버전 tag를 push하면 됩니다. 완료된 workflow의 Artifacts에서 압축파일을 내려받을 수 있습니다.

## Standalone 기능 범위

Standalone 패키지에는 DICOM 로딩, 기본 및 screw 방향 MPR, 3D 렌더링, 수동 도구, 스크류 계획, 측정과 threshold fallback segmentation이 포함됩니다.

TotalSegmentator와 PyTorch는 매우 큰 실행환경, 외부 모델 가중치 및 별도 task별 라이선스가 필요하므로 포함하지 않습니다. AI segmentation이 필요하면 소스 버전에서 `./scripts/run_app.sh --with-totalseg`를 사용하십시오.

## 로그 위치

- macOS: `~/Library/Logs/PedicleScrewSimulator/app.log`
- Windows: `%LOCALAPPDATA%\PedicleScrewSimulator\logs\app.log`

배포 압축파일에는 MIT License, 제작자 정보, 학술 인용 정보와 README가 함께 포함됩니다.
