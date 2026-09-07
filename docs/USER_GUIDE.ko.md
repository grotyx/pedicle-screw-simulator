# 사용설명서

[English](USER_GUIDE.md) | [한국어](USER_GUIDE.ko.md)

**버전:** 0.1.0

## 1. 목적과 안전 범위

Pedicle Screw Simulator는 CT 확인, 척추 자동 분할, 척추경 나사못 계획 및 스크류 방향 MPR 검토를 위한 연구·교육용 데스크톱 프로그램입니다.

> 이 프로그램은 인증된 의료기기가 아닙니다. 진단, 수술, 내비게이션 또는 환자 진료의 단독 근거로 사용하지 마십시오. 모든 결과는 자격을 갖춘 의료진이 독립적으로 확인해야 합니다.

임상 DICOM이나 환자정보가 포함된 스크린샷을 공개하지 마십시오. [로컬 DICOM 데이터 안내](../data/README.md)를 확인하십시오.

## 2. 시스템 요구사항

- Python 3.12
- 제공된 실행 스크립트 사용 시 macOS, Linux 또는 Windows
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

Windows에서는 PowerShell 실행 스크립트를 대신 사용합니다.

```powershell
git clone https://github.com/grotyx/pedicle-screw-simulator.git
cd pedicle-screw-simulator
powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1
```

TotalSegmentator 선택 설치:

```bash
./scripts/run_app.sh --with-totalseg
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1 --with-totalseg
```

점검 명령:

```bash
./scripts/run_app.sh --check
./scripts/run_app.sh --test
./scripts/run_app.sh --setup-only
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1 --check
powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1 --test
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

### 5.5 계획 매개변수(Planning Parameters)

**Planning** 아래의 접이식 **Planning parameters** 그룹에서는 자동 플래너가 스크류 크기와 위치를 정할 때 사용하는 설정을 조정할 수 있습니다. 각 값은 입력 즉시 검증되고 저장되므로(Qt `QSettings`), 프로그램을 다시 시작해도 유지됩니다.

| 항목 | 기본값 | 의미 |
|---|---:|---|
| Pedicle fill | 0.80 | 측정된 척추경 협부(isthmus) 폭 대비 스크류 직경의 비율 |
| Wall clearance | 1.0 mm | 스크류와 피질골 사이에 유지하는 최소 거리 |
| Anterior margin | 4.0 mm | anterior cortex 뒤쪽에 유지하는 안전 여유 |
| Max convergence | 35° | 플래너가 사용할 수 있는 최대 medial 수렴각 |
| HU threshold | 123 HU | 이완(loosening) 위험 경고가 표시되는 궤적 HU 기준값 |

**Reset Defaults**를 선택하면 이 다섯 개 값이 기본값으로 즉시 복원되고 저장됩니다. Lateral divergence 한계값(−5°, 플래너가 허용하는 가장 lateral한 각도)은 이번 버전에서 고정되어 있으며 패널에 노출되지 않습니다.

### 5.6 Screw MPR 검토

1. 목록, MPR 또는 3D에서 스크류를 선택합니다.
2. **Screw MPR**을 선택합니다.
3. Oblique Axial, Oblique Sagittal, Cross-section 영상을 확인합니다.
4. **Position**을 entry에서 tip 방향으로 이동하며 궤적을 확인합니다.
5. **Std MPR**을 선택하면 기본 단면으로 돌아갑니다.

Screw MPR에는 선택한 스크류가 표시됩니다. Standard MPR에는 현재 단면과 만나는 스크류가 표시됩니다.

### 5.7 스크류 측정값과 등급

**Selected Screw** 패널에는 다음 값이 표시됩니다.

- **Convergence(수렴각):** 정중선 방향으로의 axial 각도이며 부호가 있습니다. 양수는 medial(팁이 정중선을 향함), 음수는 lateral을 의미합니다.
- **Craniocaudal(두미측각):** axial 평면 위로의 궤적 상승각이며 부호가 있습니다. 양수는 cranial입니다. 스크류가 수렴하는 경우 이 값은 sagittal 투영각과 약간 다를 수 있습니다.
- **Safety(등급):** TotalSegmentator mask 위에서 원통 표면과 척추 경계 사이 거리로 계산한 Gertzbein-Robbins 등급입니다. 등급 계산에는 HU가 전혀 사용되지 않으며, 함께 표시되는 궤적의 평균·최소 HU는 참고용 정보일 뿐입니다. 해당 스크류의 척추에 TotalSegmentator mask가 없으면 등급은 `N/A`로 표시됩니다.

자동 크기 결정은 직경을 측정된 척추경 협부(isthmus) 폭의 80% 이하이면서 각 방향으로 최소 1 mm의 피질골 여유를 확보하도록 설정하고, 팁을 anterior cortex보다 최소 4 mm 뒤쪽에 위치시키며, 길이는 25–55 mm 카탈로그에서 5 mm 간격으로 선택합니다. 이 값은 작업을 위한 기본 설정이며 모든 환자에게 적용되는 임상 권고가 아닙니다.

불러온 volume은 표시 전에 LPS(identity 방향)로 재정렬됩니다. Oblique 방식으로 촬영된 volume은 identity 방향 격자로 resampling되며, 이 경우 정보 패널에 "(oblique volume resampled)"가 표시됩니다.

### 5.8 스크류 골질(骨質) 지표(Screw Quality Metrics)

스크류가 segmentation을 기준으로 등급이 매겨지면, **Selected Screw** 패널(Body HU, Wall margin, Facet, Heary 행)과 CSV/JSON 내보내기에 문헌에 근거한 골질·안전성 지표 모음이 표시됩니다.

- **Trajectory HU(평균/최소):** 스크류의 원통형 궤적을 따라 측정한 Hounsfield Unit의 평균값과 최소값입니다.
- **Pedicle HU:** 척추경 협부(isthmus) 중심에서 10 mm 이내에 있는 궤적 샘플만으로 계산한 평균 HU입니다. 자동 계획된 스크류에서만 제공됩니다 — 수동 스크류는 기준이 될 isthmus 중심이 없기 때문입니다.
- **Vertebral body HU(척추체 HU):** 척추체 중심에 위치한 8×8×6 mm 타원체 관심영역을 해당 척추의 segmentation label과 교차시켜 계산한 평균 HU입니다. 같은 이유로 자동 계획된 스크류에서만 제공됩니다.
- **Trajectory/body HU 비율:** 궤적 평균 HU를 척추체 HU로 나눈 값입니다.
- **최소 피질골 여유거리("Wall margin"):** 스크류와 피질골 사이의 가장 가까운 거리(mm)입니다.
- **Heary breach 방향:** 가장 심한 피질골 천공의 해부학적 방향 — medial, lateral, anterior, posterior, superior, inferior 중 하나입니다(Heary 2004). Side 정보가 없는 수동 스크류에서 medial/lateral 방향의 breach가 발생하면 "mediolateral"로, breach가 없으면 "none"으로 표시됩니다.
- **후관절(facet) 침범 등급(0–3):** Babu(2012) 등급을 근사한 값으로, 스크류 근위부(entry에서 가까운 1/3) 구간과 상위(cephalad) 척추의 segmentation label 사이 관계로 판정합니다 — 0은 접촉 없음, 1은 1 mm 이내로 후관절에 접함, 2는 1 mm 미만으로 침범, 3은 1 mm 이상 침범을 의미합니다.

측정값이 문헌 기준값을 넘으면 패널과 내보내기에 경고가 함께 표시됩니다.

| 지표 | 기준값 | 경고 | 참고문헌 |
|---|---|---|---|
| Trajectory HU | 123 HU 미만 | 이완(loosening) 위험 | Yamamoto 2025; Dhar 2026 |
| Vertebral body HU | 132 HU 미만 | 골다공증 | Sankar 2026 |
| Vertebral body HU | 141 HU 미만(골다공증에 해당하지 않는 경우) | 저골밀도 | Sankar 2026 |
| Trajectory/body HU 비율 | 1.0 미만 | 이완 위험 | Yang 2026 |
| 후관절 침범 등급 | 2 이상 | 후관절 침범 | Babu 2012 |

이러한 골질 관련 경고는 자동 계획된 스크류에서만 생성됩니다. 수동으로 배치한 스크류도 전체 지표 모음은 제공받지만 골질 경고는 받지 않습니다. Segmentation을 다시 실행하거나 계획을 불러올 때(이미 segmentation이 있는 경우) 계획을 다시 등급 매기면 모든 스크류의 breach distance·피질골 여유 경고가 다시 생성됩니다.

### 5.9 척추경 세부영역(Subregion) 모델(선택 사항)

Segmentation → Advanced에는 기본적으로 꺼져 있는 **Use pedicle subregion model** 옵션이 있습니다. 이를 켜면 척추를 pedicle/corpus/lamina/spinous/transverse/articular 세부영역으로 분할하는 로컬 nnU-Net 모델([MICN-Lab/Spine_Subregions](https://github.com/MICN-Lab/Spine_Subregions); Da Mutten et al., *J Imaging Inform Med* 2026)을 **Model directory** 입력란(또는 `PSS_SUBREGION_MODEL_DIR` 환경변수)이 가리키는 경로에서 찾습니다. 이 경로는 `dataset.json`과 `fold_*/checkpoint_final.pth`가 있는 nnU-Net results 폴더여야 합니다.

이 기능은 소스 설치 전용입니다. `nnunetv2`가 설치된 non-frozen Python 환경이 필요하며, 메모리 요구량은 TotalSegmentator의 `3d_fullres` 설정과 비슷합니다(GPU 권장). macOS·Windows standalone 패키지는 이 기능을 지원하지 않습니다. [데스크톱 빌드 안내](BUILDING_DESKTOP.ko.md)를 참고하십시오.

모델이 정상적으로 실행되면 각 방향의 척추경 협부(isthmus)를 먼저 해당 label에서 측정하고, label이 그 방향을 찾지 못한 경우에만 coronal 단면 탐색으로 대체합니다. Segmentation 상태 표시줄에는 "· pedicle model used"가 표시되거나, 실행되지 못했을 때는 이유와 함께 "· pedicle model unavailable: <reason>"이 표시됩니다 — 2단계 실패가 그 아래 TotalSegmentator 결과 자체를 막지는 않습니다.

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

계획 파일은 schema version 3을 사용하며, 각 스크류에 5.8절 "스크류 골질 지표"에서 설명한 골질·안전성 지표(trajectory/pedicle/body HU, HU 비율, 최소 wall 거리, Heary breach 방향, facet 침범 등급)를 담는 `metrics` 필드가 추가되었습니다. schema version 2에서는 각 스크류에 `mean_hu`, `min_hu`, `warnings`, `source`가 추가되었습니다. 이전 버전으로 저장한 계획 파일도 계속 불러올 수 있으며, 이미 segmentation이 있는 상태에서 계획을 불러오면 즉시 다시 등급이 매겨져 schema v3 지표가 채워집니다. CSV 내보내기에는 이름이 변경된 `convergence_angle_deg`, `craniocaudal_angle_deg` 열, `mean_hu`, `min_hu`, `source`, `warnings` 열과 함께 schema v3 지표 열인 `trajectory_mean_hu`, `pedicle_mean_hu`, `body_mean_hu`, `hu_ratio`, `min_wall_mm`, `heary_direction`, `facet_grade`가 포함됩니다.

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

## 12. 계획 검증

`scripts/validate_plans.py`는 예측 계획(자동 플래너 결과 또는 수련의 계획 등)을 기준 계획(전문가 계획, ground truth)과 비교하여, 척추경 나사못 계획 문헌에서 사용하는 편차 지표를 산출합니다: 진입점·팁점 평균절대편차(MAD), 두 나사못 축 사이의 3차원 각도, 수렴각(convergence)과 두미측각(craniocaudal) 차이, 직경·길이 일치도(Bland-Altman 편향 및 95% 일치한계), 척추경 중심 오프셋, 나사못 부피 Dice 중첩도입니다.

저장소 루트에서, 데스크톱 앱 밖에서 저장된 두 계획 JSON 파일로 실행합니다.

```bash
python scripts/validate_plans.py --pred pred_plan.json --ref ref_plan.json --out report
```

콘솔에 전체 코호트 요약(`key: value` 형식)을 출력하고 다음 파일을 생성합니다.

- `report.csv` — 매칭된 나사못마다 한 행씩, `level, side, head_mad_mm, tip_mad_mm, axis_angle_deg, convergence_delta_deg, craniocaudal_delta_deg, diameter_delta_mm, length_delta_mm, pedicle_center_offset_mm, dice` 열을 포함합니다.
- `report.json` — 코호트 요약(매칭/비매칭 개수, MAD 및 축 각도 평균·표준편차, 직경·길이 편향과 95% 일치한계, 평균 Dice)입니다.

두 계획의 나사못은 척추 레벨과 방향(side)으로 매칭됩니다. Dice 계산에 사용하는 래스터화 격자 크기는 `--voxel-mm`으로 조절할 수 있으며(기본값 `0.5` mm), 격자를 성기게 하면 계산은 빨라지지만 길거나 가는 나사못에서는 중첩도가 약간 과소평가될 수 있습니다.

**수치 해석.** 절대적인 합격/불합격 기준은 없으며, 사람 평가자들 사이의 편차 범위와 비교해 판단해야 합니다. 평가자 간 일치도 문헌(Scherer 2022)에 따르면, 동일 증례를 독립적으로 계획한 전문가들 사이에서도 진입점은 평균 약 4.9 mm, 팁점은 평균 약 4.4 mm 차이가 나며, 축 각도 차이는 평균 약 5.3°입니다. 예측 계획이 기준 계획과 대략 이 범위 내에서 차이 난다면 평가자 간 변동성과 부합하는 수준이며, 이를 크게 벗어나는 편차는 플래너 결과나 기준 계획 자체를 다시 검토할 필요가 있음을 시사합니다.
