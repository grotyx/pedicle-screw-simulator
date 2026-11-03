# macOS·Windows 단독 실행 패키지 만들기

[English](BUILDING_DESKTOP.md) | [한국어](BUILDING_DESKTOP.ko.md)

## 지원하는 결과물

| 플랫폼 | 결과물 | 빌드 환경 |
|---|---|---|
| macOS Apple Silicon | ZIP 안의 `.app` | macOS arm64 |
| Windows x64 | 설치 프로그램(`…-Setup.exe` + `.bin` 조각)과 포터블 ZIP | Windows x64 |

PyInstaller는 교차 컴파일러가 아니므로 각 운영체제용 패키지는 해당 운영체제에서 별도로 만들어야 합니다.

## macOS에서 직접 빌드

```bash
./scripts/run_app.sh --setup-only
source venv/bin/activate
python -m pip install -r requirements-desktop.txt
python scripts/build_desktop.py
```

결과 압축파일은 `release/`에 생성됩니다. 현재 앱에는 Apple Developer 서명과 공증이 적용되지 않으므로 최초 실행 시 Gatekeeper 경고가 나타날 수 있습니다. 앱을 우클릭하고 **Open**을 선택하면 실행할 수 있습니다.

## Windows에서 직접 빌드

Python 3.12와 [Inno Setup 6](https://jrsoftware.org/isinfo.php)이 설치된 PowerShell에서
실행합니다(`winget install JRSoftware.InnoSetup --scope user`).

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
.\venv\Scripts\python.exe -m pip install -r requirements-desktop.txt
.\venv\Scripts\python.exe scripts\build_desktop.py
```

`release\`에 두 가지 결과물이 생깁니다.

- **설치 프로그램**: `RadWright-Screw-<버전>-Windows-x64-Setup.exe`와 그 옆의
  `…-Setup-N.bin` 파일들입니다(CUDA 런타임 때문에 수 GB라 Inno Setup이 나눕니다).
  한 폴더에 두고 `.exe`를 실행합니다. 관리자 승인 없이 사용자별로
  `%LOCALAPPDATA%\Programs\RadWright Screw`에 설치되고, 시작 메뉴 항목(바탕화면
  바로가기는 선택)과 설정 > 앱의 제거 항목이 생깁니다. 새 버전 설치 프로그램을
  실행하면 그 자리에서 업데이트됩니다. 조용히 설치: `…-Setup.exe /VERYSILENT`.
- **포터블 ZIP**: `PedicleScrewSimulator.exe`와 실행에 필요한 `_internal` 폴더입니다.
  두 항목을 분리하거나 일부 파일만 이동하면 안 됩니다.

ZIP만 만들려면 `--no-installer`를 붙입니다. 설치 프로그램은 코드 서명이 없어서 처음
실행할 때 Windows SmartScreen 경고가 뜰 수 있습니다(**추가 정보** > **실행**).

## RadWright Screw 이미지 자산

브랜드 자산은 `packaging/icons/`에 있습니다. 직접 작성한
`radwright-screw.svg`가 1024 단위의 원본입니다. 외부 파일 없이 사용할 수
있는 SVG 워드마크 두 개는 시스템 sans 글꼴을 사용합니다.
`radwright-screw-wordmark.svg`는 어두운 배경용,
`radwright-screw-wordmark-light.svg`는 흰 배경용입니다. 해부학 도형을
수정하면 두 워드마크에 포함된 아이콘 도형도 원본과 맞춰 수정하십시오.

PNG 9개 크기(16, 24, 32, 48, 64, 128, 256, 512, 1024), 프레임 7개를
포함한 Windows ICO, macOS ICNS와 검토용 이미지를 다시 생성합니다.

```bash
python -m pip install Pillow  # 이미지 자산 제작 도구 전용
python scripts/render_icons.py
```

렌더러는 앱의 기존 PyQt6 SVG 기능을 사용하며 Linux에서도 화면 창 없이
동작합니다. 앱에 래스터 변환용 의존성을 추가하지 않습니다. 어느 작업
디렉터리에서든 실행할 수 있으며, `--output-dir <directory>`로 다른 위치에
내보낼 수 있습니다. 수정 후 `packaging/icons/preview.png`를 확인하십시오.
어두운 배경과 흰 배경에 1024, 256, 64, 32, 16 px 아이콘, 작은 아이콘의
확대본과 두 워드마크를 함께 표시합니다. PNG 모서리는 투명합니다.
시스템 글꼴 선택에 따라 워드마크 모양은 컴퓨터마다 달라질 수 있습니다.

## GitHub Actions 자동 빌드

**Build desktop packages** workflow는 다음 환경에서 각각 빌드합니다.

- `windows-2025`: Windows x64
- `macos-15`: macOS Apple Silicon

GitHub Actions 탭에서 수동으로 실행하거나 `v0.2.8` 같은 버전 tag를 push하면 됩니다. 완료된 workflow의 Artifacts에서 압축파일을 내려받을 수 있습니다.

## Standalone 기능 범위

Standalone 패키지에는 DICOM 로딩, 기본 및 screw 방향 MPR, 3D 렌더링, 수동 도구, 스크류 계획, 측정, TotalSegmentator 2.12.0, PyTorch, nnU-Net과 threshold fallback segmentation이 포함됩니다.

Windows 빌드에는 Spine Only가 쓰는 TotalSegmentator 모델 두 개(척추 part `Dataset292`, 6 mm 자르기 모델 `Dataset298`, 약 400 MB)가 `_internal/totalseg_weights`에 들어 있습니다. `scripts/build_desktop.py`가 빌드할 때 내려받고, Spine Only + Accurate 분할은 여기서 읽으므로 아무것도 내려받지 않습니다. Total task, Fast 품질, macOS 빌드는 처음 쓸 때 공개 가중치를 `~/.totalsegmentator/nnunet/results`에 내려받고 이후 재사용합니다. Windows 패키지에는 PyTorch CUDA 12.8 실행환경이 포함되며 지원되는 NVIDIA GPU가 없으면 CPU로 자동 전환합니다.

GUI를 열지 않고 번들 상태를 검사할 수 있습니다.

```bash
dist/PedicleScrewSimulator.app/Contents/MacOS/PedicleScrewSimulator --self-check
```

```powershell
.\dist\PedicleScrewSimulator\PedicleScrewSimulator.exe --self-check
```

## 척추경 세부영역(Subregion) 모델(선택 사항)

프로그램은 선택적으로 척추를 pedicle/corpus/lamina/spinous/transverse/articular 세부영역으로 분할하는 로컬 nnU-Net 모델([MICN-Lab/Spine_Subregions](https://github.com/MICN-Lab/Spine_Subregions); Da Mutten et al., *J Imaging Inform Med* 2026)로 척추경 탐지를 정밀화할 수 있습니다. 이 기능은 소스 설치 전용입니다. TotalSegmentator와 달리 이 단계는 항상 `nnUNetv2_predict`를 별도 프로세스로 실행하는데, frozen 빌드에는 그 실행을 대신할 Python 인터프리터가 없으므로 macOS·Windows standalone 패키지는 이 기능을 아예 거부합니다.

소스 설치에서 사용하는 방법:

1. [Spine_Subregions Releases 페이지](https://github.com/MICN-Lab/Spine_Subregions/releases)에서 학습된 가중치를 내려받습니다.
2. 표준 nnU-Net v2 results 폴더 구조로 배치합니다. 예:
   ```
   <root>/Dataset501_SpineSubregions/nnUNetTrainer__nnUNetPlans__3d_fullres/
       dataset.json
       fold_0/checkpoint_final.pth
   ```
3. 프로그램을 실행하는 것과 동일한 환경에 `nnunetv2`를 설치합니다(`nnUNetv2_predict`가 인터프리터의 `Scripts`/`bin` 폴더 옆에서 실행 가능해야 합니다).
4. Segment 페이지의 **Advanced options** 토글 안에 있는 **Model directory** 입력란(또는 `PSS_SUBREGION_MODEL_DIR` 환경변수)이 `.../nnUNetTrainer__nnUNetPlans__3d_fullres` 폴더를 가리키도록 설정한 뒤 **Use pedicle subregion model**을 켭니다.

**현재 알려진 제한사항:** 현재 upstream에 공개된 Spine_Subregions release 자산은 위에서 설명한 nnU-Net **v2** results 폴더 구조가 아니라 nnU-Net **v1** 방식의 폴더 이름(`nnUNetTrainerV2__nnUNetPlansv2.1`)을 사용합니다. 이 가중치가 nnU-Net v2용으로 재출력(재변환)되기 전까지는 해당 release 폴더를 **Model directory**에 지정해도 유효한 모델로 인식되지 않으며, segmentation 상태 표시줄에는 "pedicle model used" 대신 "pedicle model unavailable"이 표시됩니다.

라벨은 하드코딩하지 않고 모델 자체의 `dataset.json`에서 실행 시점에 읽어오므로, 세부영역 이름을 알아볼 수 있는 nnU-Net 체크포인트라면 어떤 것이든 올바르게 인식됩니다. 메모리 요구량은 TotalSegmentator의 `3d_fullres` 설정과 비슷하며 GPU를 권장합니다. 2단계 실패는 기존 TotalSegmentator 결과를 막지 않습니다. 라이선스는 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)를 참고하십시오.

## 업데이트 게시

Windows 빌드는 `release\update\`도 만듭니다. 레이어별 ZIP과 `components.json`입니다
(같은 manifest가 앱 폴더 안에도 들어갑니다).

| 레이어 | 내용 | 바뀌는 때 |
|:--|:--|:--|
| `cuda` | `_internal\torch\lib`(CUDA, cuDNN DLL) | torch를 올릴 때 |
| `runtime` | `_internal` 아래 나머지 패키지 | 의존성이 바뀔 때 |
| `app` | 실행 파일(우리 코드)과 바깥 파일 | 매 릴리스 |

레이어 이름에 파일 해시가 붙어 있어서, 게시할 때는 서버에 이미 있는 레이어를 건너뛰고
앱은 해시가 바뀐 레이어만 받습니다. 업로드:

```powershell
.\venv\Scripts\python.exe -m pip install boto3
.\venv\Scripts\python.exe scripts\publish_update.py release\update --downloads (Get-ChildItem release\*.exe, release\*.bin, release\*.zip) --dry-run
.\venv\Scripts\python.exe scripts\publish_update.py release\update --downloads (Get-ChildItem release\*.exe, release\*.bin, release\*.zip)
```

R2 버킷 `spinemetrics-dl`의 `radwright-screw/` 아래에 올리고,
`https://dl.spinemetrics.sangmin.me/radwright-screw/`로 제공합니다. `--downloads`는 설치
프로그램 `.exe` + `.bin`과 ZIP을 `downloads/<버전>/`에 올립니다(모두 R2에 둡니다. GitHub는
파일 하나를 2 GB로 제한합니다). 인증 정보는
`R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_ACCOUNT_ID` 환경 변수 또는
`%USERPROFILE%\.spinemetrics\r2.env`에서 읽습니다(커밋 금지). `latest/components.json`을
맨 마지막에 올리므로 레이어가 빠진 릴리스가 앱에 보이지 않습니다. 되돌리려면
`versions/<이전>/components.json`을 `latest/components.json`에 덮어씁니다. 다른 서버로
시험하려면 `PSS_UPDATE_URL`, 확인을 끄려면 `PSS_NO_UPDATE=1`을 씁니다.

## 패키지 크기

패키지 대부분은 프로그램이 아니라 GPU 분할 런타임입니다. Linux 개발 환경에서
측정한 값입니다(torch 2.14 + CUDA 13, 2026-10-30).

| 구성 | 크기 |
|:--|--:|
| CUDA 라이브러리(`nvidia/*`: cu13, cuDNN, NCCL, cuSPARSELt, NVSHMEM) | 3.2 GB |
| `torch` | 1.2 GB |
| `triton`(Linux 전용, **빌드에서 제외**) | 0.9 GB |
| VTK + SimpleITK + PyQt6 + SciPy + NumPy | 약 1.25 GB |
| 프로그램 코드 | 수 MB |
| TotalSegmentator Spine Only 가중치(Windows 패키지에 포함) | 0.4 GB |
| 그 밖의 TotalSegmentator 가중치(처음 사용할 때 다운로드) | 0.16–1.2 GB |

- 패키지는 GPU 추론만 대상으로 합니다. CPU 전용 빌드는 CUDA 라이브러리가 빠지지만
  분할이 몇 분 걸리므로 제공하지 않습니다.
- NCCL, cuSPARSELt, NVSHMEM은 다중 GPU용처럼 보이지만 `libtorch_cuda`가 직접
  링크합니다. 빼면 torch가 로드되지 않습니다.
- Triton은 `torch.compile` 전용입니다. Triton 없이 TotalSegmentator GPU 추론이
  동작하는 것을 확인했습니다. Windows와 macOS torch wheel에는 원래 없습니다.
- 그 밖의 가중치는 task별로 처음 실행할 때 받습니다(`~/.totalsegmentator`). 사용자가
  돌린 task만 공간을 차지합니다. 전체 `total` task는 다섯 part(약 1.2 GB)가 모두 필요하고,
  fast 모드는 159 MB 모델 하나면 됩니다. TotalSegmentator 2.12는 Spine Only에서도 다섯
  part를 모두 받으려 하므로, 포함된 모델이 있으면 앱이 계산하지 않는 네 part를 건너뜁니다.
- 소스가 `import vtk` 대신 필요한 `vtkmodules.*`만 불러오면(7개 파일) VTK를
  약 100–200 MB로 줄일 수 있습니다. 아직 하지 않았습니다.
- PyTorch를 ONNX Runtime으로 바꾸면 가장 많이 줄지만, nnU-Net 전처리·후처리를
  다시 구현해야 하므로 의도적으로 하지 않습니다.

## 로그 위치

- macOS: `~/Library/Logs/PedicleScrewSimulator/app.log`
- Windows: `%LOCALAPPDATA%\PedicleScrewSimulator\logs\app.log`

배포 압축파일에는 MIT License, 제작자 정보, 학술 인용 정보, README와 `THIRD_PARTY_NOTICES.md`가 함께 포함됩니다. TotalSegmentator 기본 `total` task는 Apache 2.0으로 공개되어 있습니다. 별도 라이선스가 필요한 TotalSegmentator task는 프로그램 기본 workflow에서 제공하지 않습니다.
