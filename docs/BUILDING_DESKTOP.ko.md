# macOS·Windows 단독 실행 패키지 만들기

[English](BUILDING_DESKTOP.md) | [한국어](BUILDING_DESKTOP.ko.md)

## 지원하는 결과물

| 플랫폼 | 결과물 | 빌드 환경 |
|---|---|---|
| macOS Apple Silicon | ZIP 안의 `.app` | macOS arm64 |
| Windows x64 | ZIP 안의 `.exe`와 지원 폴더 | Windows x64 |

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

Python 3.12가 설치된 PowerShell에서 실행합니다.

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
.\venv\Scripts\python.exe -m pip install -r requirements-desktop.txt
.\venv\Scripts\python.exe scripts\build_desktop.py
```

압축파일에는 `PedicleScrewSimulator.exe`와 실행에 필요한 `_internal` 폴더가 포함됩니다. 두 항목을 분리하거나 일부 파일만 이동하면 안 됩니다.

## GitHub Actions 자동 빌드

**Build desktop packages** workflow는 다음 환경에서 각각 빌드합니다.

- `windows-2025`: Windows x64
- `macos-15`: macOS Apple Silicon

GitHub Actions 탭에서 수동으로 실행하거나 `v0.1.0` 같은 버전 tag를 push하면 됩니다. 완료된 workflow의 Artifacts에서 압축파일을 내려받을 수 있습니다.

## Standalone 기능 범위

Standalone 패키지에는 DICOM 로딩, 기본 및 screw 방향 MPR, 3D 렌더링, 수동 도구, 스크류 계획, 측정, TotalSegmentator 2.12.0, PyTorch, nnU-Net과 threshold fallback segmentation이 포함됩니다.

모델 가중치는 ZIP 안에 넣지 않습니다. 첫 자동 분할 시 TotalSegmentator가 공개 `total` task 모델을 `~/.totalsegmentator/nnunet/results`에 내려받고 이후 실행에서 재사용합니다. 따라서 최초 실행에는 인터넷 연결, 추가 저장공간과 다운로드 시간이 필요합니다. Windows 패키지에는 PyTorch CUDA 12.8 실행환경이 포함되며 지원되는 NVIDIA GPU가 없으면 CPU로 자동 전환합니다.

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
4. Segmentation → Advanced의 **Model directory** 입력란(또는 `PSS_SUBREGION_MODEL_DIR` 환경변수)이 `.../nnUNetTrainer__nnUNetPlans__3d_fullres` 폴더를 가리키도록 설정한 뒤 **Use pedicle subregion model**을 켭니다.

**현재 알려진 제한사항:** 현재 upstream에 공개된 Spine_Subregions release 자산은 위에서 설명한 nnU-Net **v2** results 폴더 구조가 아니라 nnU-Net **v1** 방식의 폴더 이름(`nnUNetTrainerV2__nnUNetPlansv2.1`)을 사용합니다. 이 가중치가 nnU-Net v2용으로 재출력(재변환)되기 전까지는 해당 release 폴더를 **Model directory**에 지정해도 유효한 모델로 인식되지 않으며, segmentation 상태 표시줄에는 "pedicle model used" 대신 "pedicle model unavailable"이 표시됩니다.

라벨은 하드코딩하지 않고 모델 자체의 `dataset.json`에서 실행 시점에 읽어오므로, 세부영역 이름을 알아볼 수 있는 nnU-Net 체크포인트라면 어떤 것이든 올바르게 인식됩니다. 메모리 요구량은 TotalSegmentator의 `3d_fullres` 설정과 비슷하며 GPU를 권장합니다. 2단계 실패는 기존 TotalSegmentator 결과를 막지 않습니다. 라이선스는 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)를 참고하십시오.

## 로그 위치

- macOS: `~/Library/Logs/PedicleScrewSimulator/app.log`
- Windows: `%LOCALAPPDATA%\PedicleScrewSimulator\logs\app.log`

배포 압축파일에는 MIT License, 제작자 정보, 학술 인용 정보, README와 `THIRD_PARTY_NOTICES.md`가 함께 포함됩니다. TotalSegmentator 기본 `total` task는 Apache 2.0으로 공개되어 있습니다. 별도 라이선스가 필요한 TotalSegmentator task는 프로그램 기본 workflow에서 제공하지 않습니다.
