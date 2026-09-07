# Pedicle Screw Simulator

[English](README.md) | [한국어](README.ko.md)

DICOM CT 영상 확인, 척추 자동 분할, 척추경 나사못 계획 및 MPR·3D 검토를 위한 연구용 데스크톱 프로그램입니다.

**버전:** 0.1.0

**주요 검증 환경:** macOS, Python 3.12

**제작자:** [Sang-Min Park, MD, Ph.D.](https://sangmin.me)

**소속:** 분당서울대학교병원 척추센터·정형외과, 서울대학교 의과대학

**연락처:** [psmini@snu.ac.kr](mailto:psmini@snu.ac.kr)

> **연구 및 교육 목적으로만 사용하십시오.** 이 프로그램은 인증된 의료기기가 아니며 진단, 수술, 내비게이션 또는 환자 진료의 단독 근거로 사용할 수 없습니다. 자동 분할, 스크류 제안, 길이·직경, breach grade와 경고는 반드시 자격을 갖춘 의료진이 독립적으로 확인해야 합니다.

## 주요 기능

- 여러 시리즈를 포함한 DICOM CT 불러오기
- Axial, Sagittal, Coronal MPR 동기화
- VTK 기반 CPU 볼륨 렌더링 및 척추별 3D 메시 표시
- GPU 우선 TotalSegmentator 연동 및 CPU 재시도
- 로컬에 설치한 척추경 세부영역(subregion) nnU-Net 모델을 이용한 선택적 isthmus 라벨 정밀화(소스 설치 전용)
- 여러 척추 레벨 선택과 자동 스크류 제안
- 다목적 궤적 최적화기(기본값)와 legacy 플래너 대체, 그리고 피질골 궤적(cortical bone trajectory, CBT) 계획 모드
- 기본 MPR 및 스크류 방향에 정렬된 oblique MPR
- MPR과 3D에서 entry, tip, 전체 스크류 직접 수정
- 수동 스크류 추가, 거리 측정, 각도 측정
- JSON 계획 저장·불러오기 및 지원되는 CSV/STL 내보내기
- 세 가지 UI 테마, MPR 이동·확대, 3D 탐색 기능

## 빠른 시작

### macOS / Linux

```bash
git clone https://github.com/grotyx/pedicle-screw-simulator.git
cd pedicle-screw-simulator
./scripts/run_app.sh
```

실행 스크립트가 프로젝트 내부에 `venv`를 만들고 필요한 패키지를 설치한 후 프로그램을 시작합니다.

### Windows

```powershell
git clone https://github.com/grotyx/pedicle-screw-simulator.git
cd pedicle-screw-simulator
powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1
```

PowerShell 실행 스크립트도 동일하게 `venv`를 만들고 필요한 패키지를 설치한 후 프로그램을 시작합니다.

### TotalSegmentator 선택 설치

```bash
./scripts/run_app.sh --with-totalseg
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1 --with-totalseg
```

TotalSegmentator 최초 실행 시 모델 파일을 내려받을 수 있으며 많은 RAM 또는 GPU 메모리가 필요할 수 있습니다.

### 직접 실행

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

### 검증

```bash
./scripts/run_app.sh --check
./scripts/run_app.sh --test
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1 --check
powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1 --test
```

현재 자동 테스트 스위트를 실행하고 최신 테스트 개수를 확인하려면 `./scripts/run_app.sh --test`(macOS/Linux) 또는 `powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1 --test`(Windows)를 실행하십시오.

검증된 macOS/Python 3.12 환경을 동일하게 재현하려면 다음 파일을 사용합니다.

```bash
python -m pip install -r requirements-lock.txt
```

`requirements.txt`는 호환 가능한 최소 버전을 제공하고, `requirements-lock.txt`는 v0.1.0 검증에 사용한 핵심 실행·테스트 환경을 기록합니다. `requirements-desktop.txt`는 standalone 빌드에 추가되는 TotalSegmentator와 PyTorch 버전을 고정합니다.

## 단독 실행 패키지

GitHub의 **Build desktop packages** workflow에서 다음 파일을 각각 생성합니다.

- macOS Apple Silicon용 `.app`
- Windows x64용 `.exe`와 지원 폴더

[데스크톱 빌드 안내](docs/BUILDING_DESKTOP.ko.md)를 참고하십시오. 현재 macOS 앱은 Apple 공증을 받지 않았으므로 최초 실행 시 앱을 우클릭하고 **Open**을 선택해야 할 수 있습니다.

Standalone 패키지에는 TotalSegmentator 2.12.0, PyTorch와 nnU-Net이 포함되므로 Python을 별도로 설치할 필요가 없습니다. 첫 자동 분할 시 공개 `total` task 모델을 사용자 TotalSegmentator 캐시에 내려받기 때문에 최초 한 번은 인터넷 연결과 추가 저장공간이 필요합니다. 이후에는 캐시된 모델을 재사용합니다.

## 기본 사용 순서

1. DICOM CT 폴더를 엽니다.
2. 기본 MPR과 3D에서 올바른 영상인지 확인합니다.
3. 자동 분할을 실행합니다.
4. 표시하고 계획할 척추 레벨을 선택합니다.
5. **Plan Screws**를 눌러 수정 가능한 스크류 제안을 만듭니다.
6. 스크류를 선택하고 **Screw MPR**에서 궤적 방향 영상을 확인합니다.
7. 필요에 따라 entry, tip, 전체 위치, 직경과 길이를 수정합니다.
8. 측정을 추가하고 계획 파일을 저장합니다.

## 주요 조작법

### MPR

| 기능 | 조작 |
|---|---|
| 기본 MPR 단면 이동 | 마우스 휠 |
| 확대·축소 | Ctrl/Cmd + 휠 또는 `− / +` |
| 영상과 표시 함께 이동 | `Pan` 활성화 후 왼쪽 드래그 |
| 화면 맞춤 | `Fit` 또는 `Fit MPR` |
| Window/level | 오른쪽 드래그 |

### 3D

| 기능 | 조작 |
|---|---|
| 회전 | 왼쪽 드래그 |
| 확대·축소 | 마우스 휠 |
| 이동 | `Pan` 활성화 후 드래그 또는 Shift+드래그 |
| 특정 위치 확대 | 해부학 구조 더블클릭 |
| 초기 방향 복원 | `Reset View` |
| 내부 스크류 확인 | `Vertebra Transparency` 증가 |

### 스크류 수정

| 선택 부위 | 동작 |
|---|---|
| 스크류 head 더블클릭 | Entry만 이동 |
| Tip 또는 원위부 shaft 더블클릭 | Tip만 이동 |
| 중간 shaft 더블클릭 | Entry와 tip을 함께 이동 |
| 다시 더블클릭 | 현재 위치 확정 |
| `Esc` | 이동 모드 종료 |

## 문서

- [English User Guide](docs/USER_GUIDE.md)
- [한국어 사용설명서](docs/USER_GUIDE.ko.md)
- [기여 안내](CONTRIBUTING.md)
- [보안 및 의료데이터 보호](SECURITY.md)
- [로컬 DICOM 데이터 정책](data/README.md)
- [제작자와 크레딧](AUTHORS.md)
- [학술 인용 정보](CITATION.cff)
- [macOS·Windows 빌드 안내](docs/BUILDING_DESKTOP.ko.md)

## 의료데이터 보호

임상 DICOM에는 표준·비공개 태그, UID, 파일명, 오버레이, 메타데이터 또는 영상에 포함된 문자로 환자정보가 남을 수 있습니다. 이 저장소는 로컬 DICOM, NIfTI, 로그, 가상환경 및 생성된 결과물을 Git에서 제외합니다.

임상 영상, 식별 가능한 스크린샷, 계획 결과물 또는 비식별화되지 않은 로그를 GitHub 이슈, Pull Request, Release 또는 CI 결과에 올리지 마십시오.

## 프로젝트 구성

```text
main.py          프로그램 시작 파일
src/             프로그램 소스 코드
tests/           자동 테스트
scripts/         실행 및 TotalSegmentator 설치 스크립트
docs/            한글·영문 사용설명서
data/README.md   로컬 데이터 보호 안내; 임상 데이터 없음
```

## 현재 제한사항

- 자동 계획은 기하학 기반 연구 기능이며 임상적으로 검증된 내비게이션이 아닙니다.
- 결과는 DICOM geometry와 segmentation 정확도에 영향을 받습니다.
- 변형, 골절, 삽입물, artifact, 이행성 척추 및 부정확한 segmentation은 제안을 무효화할 수 있습니다.
- 척추경 협부(isthmus) 탐지는 기하학적 방식(TotalSegmentator mask의 coronal 단면 최소 면적 탐색)이며 변형 척추에서는 검증되지 않았습니다.
- Standalone 빌드는 Apple Silicon macOS와 Windows x64용으로 제공하며 Intel macOS와 Linux 패키지는 제공하지 않습니다.
- Standalone에 TotalSegmentator가 포함되지만 모델은 최초 사용 시 내려받으며 많은 메모리와 저장공간이 필요할 수 있습니다.
- 선택적 척추경 세부영역(subregion) 모델(nnU-Net 기반 pedicle/corpus/lamina 라벨 정밀화)은 `nnunetv2`가 설치된 소스 환경이 필요하며, standalone 패키지에서는 지원하지 않습니다.
- 현재 공개된 Spine_Subregions release는 이 프로그램이 기대하는 nnU-Net v2 폴더 구조가 아니라 nnU-Net v1 방식의 폴더 구조를 사용하므로, 가중치가 nnU-Net v2용으로 재출력되기 전까지는 model directory가 인식되지 않습니다.
- 임상 정확도, 관찰자 간 일치도 및 전향적 결과는 아직 확립되지 않았습니다.

## 학술 인용

연구에 이 프로그램을 사용했다면 GitHub의 **Cite this repository** 기능이나 [`CITATION.cff`](CITATION.cff)를 사용하십시오.

> Park S-M. Pedicle Screw Simulator (Version 0.1.0) [Computer software]. 2026. https://github.com/grotyx/pedicle-screw-simulator

소프트웨어 논문이 출판되면 버전별 소프트웨어 인용을 유지하면서 논문 DOI를 preferred citation으로 추가할 수 있습니다.

## 참고문헌

[사용설명서](docs/USER_GUIDE.ko.md)에서 설명하는 골질(HU) 임계값, breach·facet 분류, 계획 검증 지표는 아래 문헌을 근거로 하며, 여기 열거된 모든 문헌은 이 버전에 실제로 구현되어 있는 기능의 근거입니다. 프로그램의 상수와 계산식은 이 문헌들을 공학적으로 근사한 것이며 문헌 자체를 대체하지 않습니다. 정확한 구현은 `src/utils/constants.py`, `src/core/bone_quality.py`, `src/core/breach_classification.py`, `src/core/plan_metrics.py`를 참고하십시오.

- Götschi et al. (2026), *Journal of Spine Surgery* — 척추경 나사못 크기 결정 여유값(pedicle fill ratio, 피질골 여유거리, anterior margin).
- Yamamoto et al. (2025), *Asian Spine Journal* — 나사못 이완(loosening) 위험과 관련된 궤적 HU 기준값.
- Dhar et al. (2026), *Asian Spine Journal* — 나사못 이완과 관련된 궤적 HU 범위.
- Chen et al. (2024), *Orthopaedic Surgery* — 척추경 영역 HU 측정 방법.
- Yang et al. (2026), *Global Spine Journal* — 궤적/척추체 HU 비율과 이완 위험도.
- Sankar et al. (2026), *Neurosurgery* — L1–L5 해면골 HU 기준값(골다공증·저골밀도).
- Heary et al. (2004) — 척추경 나사못 피질골 천공 방향 분류.
- Babu et al. (2012) — 척추경 나사못에 의한 후관절(facet) 침범 등급 분류.
- Scherer et al. (2022), *The Spine Journal* — 척추경 나사못 궤적 계획의 평가자 간 일치도.
- Wang et al. (2024), *Bioengineering* — 척추경 나사못 크기 결정 규칙(협부 폭 대비 직경 비율 및 피질골 여유거리)과 나사못 부피 Dice 중첩도·계획 비교 방법.
- Da Mutten et al. (2026), *Journal of Imaging Informatics in Medicine* — 흉요추 세부영역(pedicle/corpus/lamina/spinous/transverse/articular)을 자동 분할하는 nnU-Net 모델로, 선택적 척추경 세부영역 모델에서 사용합니다.
- Massalimova et al. (2025), *Scientific Reports* — 자동 나사못 궤적 평가를 위한 척추경 중심 오프셋(pedicle-centre offset) 지표.
- Herkner et al. (2026), *Journal of Clinical Medicine* — 자동 산출 임플란트 치수와 술자 선택 치수의 Bland-Altman 비교.
- Zhang et al. (2024), *Asian Spine Journal* — cortical bone trajectory(CBT) 나사못 적응증에 대한 Delphi 합의 — cortical bone trajectory(CBT) 계획 모드의 금기사항 안내문의 근거입니다.
- Zeng et al. (2024), *Orthopaedic Surgery* — CT 기반 cortical bone trajectory(CBT) 나사못 궤적 파라미터 — cortical bone trajectory(CBT) 계획 모드의 기본 entry 각도의 근거입니다.

## 라이선스

프로젝트 소스 코드는 [MIT License](LICENSE)로 공개합니다. 저작권 및 허가문을 유지하면 사용, 수정 및 재배포할 수 있습니다.

외부 구성요소에는 각각의 라이선스가 적용됩니다. [외부 구성요소 고지](THIRD_PARTY_NOTICES.md)를 확인하십시오. 재배포자는 PyQt6의 GPL/상용 라이선스와 VTK, SimpleITK, TotalSegmentator, PyTorch, nnU-Net 등 각 의존성의 조건을 별도로 준수해야 합니다.
