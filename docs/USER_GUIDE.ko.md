# 사용설명서

[English](USER_GUIDE.md) | [한국어](USER_GUIDE.ko.md)

**버전:** 0.1.0

## 1. 목적과 안전 범위

Pedicle Screw Simulator는 CT 확인, 척추 자동 분할, 척추경 나사못 계획 및 스크류 방향 MPR 검토를 위한 연구·교육용 데스크톱 프로그램입니다.

> 이 프로그램은 인증된 의료기기가 아닙니다. 진단, 수술, 내비게이션 또는 환자 진료의 단독 근거로 사용하지 마십시오. 모든 결과는 자격을 갖춘 의료진이 독립적으로 확인해야 합니다.

임상 DICOM이나 환자정보가 포함된 스크린샷을 공개하지 마십시오. [로컬 DICOM 데이터 안내](../data/README.md)를 확인하십시오.

## 2. 시스템 요구사항

- Python 3.12
- 제공된 실행 스크립트 사용 시 macOS 또는 Linux
- 최소 1280×760 이상의 화면 영역
- CT 용량에 맞는 충분한 RAM
- TotalSegmentator 가속을 위한 CUDA 지원 NVIDIA GPU(선택)

Standalone 빌드는 Apple Silicon macOS와 Windows x64용으로 제공합니다. Intel macOS는 지원하지 않습니다.

## 3. 설치와 실행

```bash
git clone https://github.com/grotyx/pedicle-screw-simulator.git
cd pedicle-screw-simulator
./scripts/run_app.sh
```

TotalSegmentator 선택 설치:

```bash
./scripts/run_app.sh --with-totalseg
```

점검 명령:

```bash
./scripts/run_app.sh --check
./scripts/run_app.sh --test
./scripts/run_app.sh --setup-only
```

### macOS·Windows 단독 실행 패키지

Standalone 패키지는 Python을 별도로 설치하지 않아도 됩니다. GitHub의 **Build desktop packages** workflow에서 생성된 압축파일을 내려받아 전체 폴더를 해제한 후 `.app` 또는 `.exe`를 실행합니다. Windows에서는 실행 파일 옆의 `_internal` 폴더를 함께 보관해야 합니다.

Standalone 패키지에는 TotalSegmentator, PyTorch와 nnU-Net이 포함됩니다. 첫 자동 분할 시 공개 `total` task 모델을 내려받고 이후에는 로컬 캐시를 재사용합니다. 자세한 내용은 [데스크톱 빌드 안내](BUILDING_DESKTOP.ko.md)를 참고하십시오.

## 4. 화면 구성

Planning 작업화면은 다음 영역으로 구성됩니다.

- **Axial, Sagittal, Coronal MPR:** crosshair, segmentation, 측정 및 스크류가 표시되는 동기화 CT 단면
- **3D 화면:** CT volume, 척추 메시, 스크류 및 선택적인 MPR plane
- **작업 패널:** Study, Segmentation, Planning, Selected Screw, Validation 설정
- **도구 모음:** Select, Add Screw, Distance, Angle

**Planning**은 큰 3D 화면과 작은 MPR을 사용합니다. **MPR Focus**는 MPR 검토를 위한 큰 2×2 배치를 사용합니다.

## 5. 기본 계획 과정

### 5.1 CT 열기

1. **Open DICOM Folder**를 누르거나 `Ctrl+O`를 사용합니다.
2. DICOM 시리즈가 들어 있는 폴더를 선택합니다.
3. 여러 시리즈가 있으면 사용할 CT 시리즈를 선택합니다.
4. 세 MPR과 3D에서 해부학 구조와 방향이 올바른지 확인합니다.

### 5.2 자동 분할

1. **Run Auto Segmentation**을 누릅니다.
2. GPU를 우선 사용하고 필요한 경우 CPU로 다시 시도합니다.
3. 척추 라벨과 3D 메시가 나타날 때까지 기다립니다.
4. 스크류 계획 전에 segmentation 경계를 확인합니다.

Standalone에는 TotalSegmentator가 이미 포함되어 있습니다. 최초 모델 다운로드나 추론에 실패하면 프로그램이 이유를 표시하고 threshold fallback을 생성합니다. 대체 결과를 임상적으로 정확하다고 가정하면 안 됩니다. 소스 설치에서는 `--with-totalseg`로 AI 실행환경을 추가할 수 있습니다.

### 5.3 척추 레벨 선택

**Visible / Plan Levels** 체크박스를 사용합니다.

- 체크한 척추가 3D에 표시됩니다.
- **Plan Screws**는 동일하게 체크한 레벨만 사용합니다.
- **Isolate Vertebrae**는 MPR에서 척추 외 구조를 가립니다.
- **Restore Full Volume**은 원래 CT로 돌아갑니다.

### 5.4 자동 스크류 생성

**Plan Screws**를 누르면 수정 가능한 스크류가 목록에 바로 추가됩니다.

현재 기본값:

- 길이는 일반적으로 50 mm 이하에서 5 mm 간격으로 제안합니다.
- 안전한 골내 길이가 60 mm 이상일 때만 55 mm를 허용합니다.
- 권장 직경은 S1과 L3–L5에서 6.5 mm, L1–L2에서 6.0 mm, T1–T12에서 5.5 mm입니다.
- 자동 직경은 최대 7.0 mm, 수동 조절은 최대 7.5 mm입니다.

이 값은 작업을 위한 기본 설정이며 모든 환자에게 적용되는 임상 권고가 아닙니다.

### 5.5 Screw MPR 검토

1. 목록, MPR 또는 3D에서 스크류를 선택합니다.
2. **Screw MPR**을 선택합니다.
3. Oblique Axial, Oblique Sagittal, Cross-section 영상을 확인합니다.
4. **Position**을 entry에서 tip 방향으로 이동하며 궤적을 확인합니다.
5. **Std MPR**을 선택하면 기본 단면으로 돌아갑니다.

Screw MPR에는 선택한 스크류가 표시됩니다. Standard MPR에는 현재 단면과 만나는 스크류가 표시됩니다.

### 5.6 스크류 측정값과 등급

**Selected Screw** 패널에는 다음 값이 표시됩니다.

- **Convergence(수렴각):** 정중선 방향으로의 axial 각도이며 부호가 있습니다. 양수는 medial(팁이 정중선을 향함), 음수는 lateral을 의미합니다.
- **Craniocaudal(두미측각):** axial 평면 위로의 궤적 상승각이며 부호가 있습니다. 양수는 cranial입니다. 스크류가 수렴하는 경우 이 값은 sagittal 투영각과 약간 다를 수 있습니다.
- **Safety(등급):** TotalSegmentator mask 위에서 원통 표면과 척추 경계 사이 거리로 계산한 Gertzbein-Robbins 등급입니다. 등급 계산에는 HU가 전혀 사용되지 않으며, 함께 표시되는 궤적의 평균·최소 HU는 참고용 정보일 뿐입니다. 해당 스크류의 척추에 TotalSegmentator mask가 없으면 등급은 `N/A`로 표시됩니다.

자동 크기 결정은 직경을 측정된 척추경 협부(isthmus) 폭의 80% 이하이면서 각 방향으로 최소 1 mm의 피질골 여유를 확보하도록 설정하고, 팁을 anterior cortex보다 최소 4 mm 뒤쪽에 위치시키며, 길이는 25–55 mm 카탈로그에서 5 mm 간격으로 선택합니다. 이 값은 작업을 위한 기본 설정이며 모든 환자에게 적용되는 임상 권고가 아닙니다.

불러온 volume은 표시 전에 LPS(identity 방향)로 재정렬됩니다. Oblique 방식으로 촬영된 volume은 identity 방향 격자로 resampling되며, 이 경우 정보 패널에 "(oblique volume resampled)"가 표시됩니다.

## 6. 스크류 수정

### 6.1 직접 수정

1. 큰 head를 더블클릭하면 entry point만 이동합니다.
2. 뾰족한 tip 또는 원위부 shaft를 더블클릭하면 tip만 이동합니다.
3. 중간 shaft를 더블클릭하면 entry와 tip이 함께 이동합니다.
4. 마우스 버튼을 누르지 않은 상태로 포인터를 움직입니다.
5. 다시 더블클릭하면 이동을 종료합니다.
6. `Esc`를 누르면 마지막 유효 위치를 유지하고 이동 모드를 종료합니다.

MPR에서 수정할 때 CT는 고정되고 스크류가 움직입니다. 수정 후 Screw MPR은 변경된 궤적에 다시 정렬됩니다.

### 6.2 직경 입력

직경 입력칸은 축약 입력을 지원합니다.

| 입력 | 결과 |
|---:|---:|
| `65` | 6.5 mm |
| `55` | 5.5 mm |
| `60` | 6.0 mm |
| `7` | 7.0 mm |

4.0–7.5 mm 범위에서 0.5 mm 간격으로 정규화됩니다.

### 6.3 삭제

스크류를 선택하고 **Delete Screw** 또는 `Delete` 키를 사용합니다.

## 7. 수동 도구

### Add Screw

1. **Add Screw**를 선택합니다.
2. MPR에서 entry point를 클릭합니다.
3. 같은 MPR에서 target point를 클릭합니다.
4. 새 스크류를 확인하고 수정합니다.

### 거리와 각도

- **Distance:** 한 MPR에서 두 점을 클릭합니다.
- **Angle:** 세 점을 클릭하며 두 번째 점이 꼭짓점입니다.

측정값은 생성한 단면에 속합니다. 다른 단면으로 이동하면 숨겨지고 원래 단면으로 돌아오면 다시 표시됩니다.

- **Show Cut**으로 측정한 단면으로 돌아갑니다.
- 측정선을 클릭하면 선택되고 노란 조절점이 표시됩니다.
- 조절점을 드래그해 한 점을 수정합니다.
- **Edit**으로 전체 측정을 다시 합니다.
- **Delete** 또는 `Delete` 키로 제거합니다.

## 8. 화면 조작

### MPR

| 조작 | 기능 |
|---|---|
| 마우스 휠 | 기본 MPR 단면 이동 |
| Ctrl/Cmd + 휠 | 확대·축소 |
| `Pan` 후 왼쪽 드래그 | 영상과 표시 이동 |
| `− / + / Fit` | 축소, 확대, 화면 맞춤 |
| 오른쪽 드래그 | Window/level 조절 |

### 3D

| 조작 | 기능 |
|---|---|
| 왼쪽 드래그 | 회전 |
| 마우스 휠 | 확대·축소 |
| `Pan` 후 드래그 또는 Shift+드래그 | 모델 이동 |
| 해부학 구조 더블클릭 | 해당 위치로 초점 이동 |
| `Reset View` | 초기 sagittal 방향 복원 |
| `Vertebra Transparency` | 내부 스크류 표시 정도 조절 |
| `Planes On / Off` | MPR plane 표시·숨김 |

## 9. 저장과 내보내기

- 계획을 JSON 형식으로 저장하고 불러옵니다.
- 지원되는 계획 표를 CSV로 내보냅니다.
- 지원되는 골 표면을 STL로 내보냅니다.

계획 파일은 schema version 2를 사용하며 각 스크류에 `mean_hu`, `min_hu`, `warnings`, `source`가 추가되었습니다. 이전 버전으로 저장한 계획 파일도 계속 불러올 수 있습니다. CSV 내보내기에는 이름이 변경된 `convergence_angle_deg`, `craniocaudal_angle_deg` 열과 함께 `mean_hu`, `min_hu`, `source`, `warnings` 열이 포함됩니다.

계획 파일, 스크린샷과 3D 메시는 DICOM 헤더가 없어도 환자와 연결될 수 있으므로 공유 전에 확인하십시오.

## 10. 문제 해결

### CT가 너무 작게 보임

`+`를 사용하거나 `Pan`을 활성화해 이동한 후 `Fit`을 누릅니다. 전체 MPR은 **Fit MPR**로 초기화합니다.

### TotalSegmentator가 느리거나 메모리가 부족함

- 가능한 경우 CUDA GPU를 사용합니다.
- 메모리를 많이 사용하는 다른 프로그램을 종료합니다.
- 최초 모델 다운로드가 끝날 때까지 기다립니다. 이후에는 캐시된 모델을 재사용합니다.
- 저해상도 segmentation은 경계 정확도를 낮출 수 있습니다.

### 특정 스크류가 생성되지 않음

허용되는 골내 궤적을 찾지 못하면 해당 방향을 건너뜁니다. Segmentation을 확인하고 스크류를 수동으로 추가하거나 수정하십시오.

### 실행 문제

```bash
./scripts/run_app.sh --check
tail -100 logs/app.log
```

문제를 보고할 때 운영체제, Python 버전, 재현 과정 및 비식별화한 로그를 포함하십시오. 임상 데이터를 첨부하지 마십시오.

## 11. 버전, 제작자 및 학술 인용

**Help → About Pedicle Screw Simulator**에서 설치된 버전, 제작자, 소속, 이메일, 홈페이지, 소스 저장소, MIT License와 연구용 안내를 확인할 수 있습니다.

- 제작자: Sang-Min Park, MD, Ph.D.
- 조직: 분당서울대학교병원
- 학술 소속: 서울대학교 의과대학
- 홈페이지: [https://sangmin.me](https://sangmin.me)
- 학술 인용: [`CITATION.cff`](../CITATION.cff) 참고
