# 사용설명서

[English](USER_GUIDE.md) | [한국어](USER_GUIDE.ko.md)

**버전:** 0.2.0

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

- **작업 단계 표시줄(workflow bar):** MPR/3D 화면 위에서 Open DICOM, Segment, Plan Screws를 순서대로 안내하는 표시줄
- **Axial, Sagittal, Coronal MPR:** crosshair, segmentation, 측정 및 스크류가 표시되는 동기화 CT 단면
- **3D 화면:** CT volume, 척추 메시, 스크류 및 선택적인 MPR plane — **Screw MPR**이 활성화된 동안에는 표준 axial/sagittal/coronal plane 대신 스크류 정렬 plane이 표시됩니다(5.8절 참고)
- **Selected Screw 검사창:** 작업 중인 스크류의 요약 — 크기, 각도, 종판각, construct 정렬, 등급, HU 통계, facet·Heary 분류, 척추경 판정
- **제어 탭:** **Study**(시리즈·표시·분할), **Planning**(레벨 선택, Planning parameters, 제안 생성, Screw Review 표), **Tools**(수동 배치와 측정)
- **도구 모음:** Select, Add Screw, Distance, Angle

### 작업 단계 표시줄

MPR/3D 화면 위의 표시줄에는 "① Open DICOM → ② Segment → ③ Plan Screws"가 표시됩니다. 각 단계는 이름이 가리키는 기존 동작을 그대로 실행합니다 — 별도의 작업 절차가 아니라 그 동작으로 가는 바로가기이며, 각 단계가 가리키는 탭 버튼은 그대로 남아 있고 이전과 똑같이 동작합니다.

| 단계 | 실행하는 동작 |
|---|---|
| ① Open DICOM | **Open DICOM Folder**(`Ctrl+O`) — 5.1절과 같은 폴더 선택 대화상자 |
| ② Segment | **Study** 탭의 **Run Auto Segmentation** — 5.2절과 동일 |
| ③ Plan Screws | **Planning** 탭의 **Plan Screws** — 5.4절과 동일 |

완료된 단계에는 체크 표시("✓ Open DICOM")가 붙습니다. 아직 해야 할 다음 단계는 비활성 상태라도 주요 동작으로 강조되어, 표시줄이 항상 다음에 할 일을 가리킵니다. 해당 동작을 아직 쓸 수 없으면(아직 study가 없거나, segmentation이 실행 중이거나, 선택한 척추 레벨이 없으면) 단계는 흐리게 표시됩니다. 각 단계에 마우스를 올리면 tooltip이 나타나며, 대기 중인 단계는 무엇이 있어야 풀리는지 알려줍니다.

| 단계 | Tooltip |
|---|---|
| ① Open DICOM | "Open a DICOM series folder" |
| ② Segment | "Run TotalSegmentator on the loaded study", 또는 study를 열기 전에는 "Open a DICOM series first" |
| ③ Plan Screws | "Plan screws for the selected vertebral levels", "Select vertebral levels in the Planning tab", 또는 "Run segmentation first" |

② 단계는 TotalSegmentator mask가 만들어진 경우에만 완료로 표시됩니다 — threshold fallback으로는 계획을 진행할 수 없으므로, 실제 segmentation이 성공할 때까지 이 단계는 열려 있습니다. ③ 단계는 자동으로 계획된 스크류가 존재할 때만 완료로 표시됩니다. 수동으로 배치한 스크류는 포함되지 않으며, 자동 스크류를 지우면 이 단계는 다시 열립니다. 앞선 단계가 끝나지 않은 상태에서 나중 단계가 완료로 표시되는 일은 없습니다. 새 study를 불러오면 표시줄 전체가 초기화되고, 현재 segmentation을 지우면 ② 단계가 다시 열립니다.

### 배치

**Planning**은 큰 3D 화면과 작은 MPR을 사용합니다. **MPR Focus**는 MPR 검토를 위한 큰 2×2 배치를 사용합니다.

### 방향 표시

각 MPR 창의 네 변에는 그 방향으로 바라볼 때의 해부학적 방향이 표시됩니다: **A**nterior(전방), **P**osterior(후방), 환자 **L**eft(좌), 환자 **R**ight(우), **S**uperior(상), **I**nferior(하). Axial 화면은 방사선과 관례대로 열립니다 — 전방이 위쪽, 환자 좌측이 화면 오른쪽 — 따라서 좌측 척추경은 영상의 오른쪽에 나타납니다. 글자 뒤에 작은따옴표가 붙으면 해당 축이 화면에 비스듬하다는 뜻이며, 표시는 가장 가까운 방향일 뿐 정확한 방향이 아닙니다.

계획 전에 표시가 환자의 알려진 좌우와 맞는지 반드시 확인하십시오. 이 표시는 DICOM direction cosine에서 유도하므로, 잘못 기록되었거나 사람이 수정한 시리즈에서는 틀릴 수 있습니다.

### 한 창 최대화

창 머리글을 더블클릭하거나 `Ctrl+M`을 누르면 작업 중인 창이 전체 영역으로 확대됩니다. 같은 동작으로 이전 배치로 돌아갑니다. `Ctrl+M`은 어떤 도구를 쓰고 있든 포인터가 마지막으로 작업한 창을 따라갑니다.

### 테마

**View** 메뉴의 **Theme** 선택 상자에서 밝은 테마 하나(Soft Light)와 어두운 테마 둘(기본값인 Graphite Blue, Graphite Mint) 중 고를 수 있습니다. 선택은 다음 실행에도 유지됩니다.

## 5. 기본 계획 과정

작업 단계 표시줄(4절 참고)은 이 절과 대응합니다. ① 단계는 5.1절, ② 단계는 5.2절, ③ 단계는 5.3절에서 레벨을 체크한 뒤의 5.4절입니다. 생성된 제안의 검토와 수정은 5.8절 이후와 6절에서 이어집니다.

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

#### CT 기준 경계 정제(Refine boundaries against CT)

TotalSegmentator는 약 1.5 mm에서 추론하고 그 결과를 CT 격자로 업샘플링하므로, 1 mm 미만 해상도 연구에서는 모든 표면에 계단 모양이 남습니다. 이 계단은 해부학적 구조가 아니며, 3D 렌더링과 마스크에서 측정하는 척추경 협부 폭을 모두 왜곡합니다.

**Refine boundaries against CT**를 켜 둔 채로(기본값) 사용하면 경계 부분을 CT 자체로 다시 판정합니다. 라벨마다 안티에일리어싱을 적용하고, 표면에서 1.5 mm 이내의 복셀을 골밀도로 재배정하며, 내부 구멍을 3D로 메우고, 각 척추에서 가장 큰 연결성분만 남깁니다. 512 × 512 × 292 연구에서 약 1–2초가 추가되며 취소할 수 있습니다.

상태 표시줄에 현재 사용 중인 마스크가 `Refined mask` 또는 `Raw mask`로 표시되고, 정제가 건드리지 않기로 한 라벨이 함께 나옵니다. 어떤 라벨의 부피를 30 % 넘게 바꾸는 정제는 교정이 아니라 실패로 보고 거부하며, 원본 라벨을 그대로 씁니다.

모델 출력을 있는 그대로 보고 싶을 때, 예를 들어 다른 도구의 마스크와 비교할 때는 이 옵션을 끄십시오.

**자동 분할을 다시 실행하면 현재 계획의 근거가 된 척추경 분석이 폐기됩니다.** 스크류 등급은 새 마스크로 즉시 다시 계산되지만, 척추경 폭·좁음 판정·종판 각도는 제안을 다시 생성할 때까지 표시되지 않습니다. 이 값들은 교체된 마스크를 분석해 얻은 것이므로, 다른 마스크에서 나온 등급 옆에 나란히 보여주면 오해를 부르기 때문입니다.

### 5.3 척추 레벨 선택

**Visible / Plan Levels** 체크박스를 사용합니다.

- 체크한 척추가 3D에 표시됩니다.
- **Plan Screws**는 동일하게 체크한 레벨만 사용합니다.
- **Isolate Vertebrae**는 MPR에서 척추 외 구조를 가립니다.
- **Restore Full Volume**은 원래 CT로 돌아갑니다.

### 5.4 자동 스크류 생성

**Plan Screws**를 누르면 수정 가능한 스크류가 목록에 바로 추가됩니다.

현재 기본값:

- 길이는 카탈로그(25–55 mm, 5 mm 간격) 중 팁이 설정된 Anterior margin만큼 앞쪽에 뼈를 남기는 가장 긴 길이이며, 스크류 자체 축을 따라 측정합니다 — 아래를 참고하십시오.
- 권장 직경은 S1과 L3–L5에서 6.5 mm, L1–L2에서 6.0 mm, T1–T12에서 5.5 mm입니다.
- 자동 직경은 최대 7.0 mm, 수동 조절은 최대 7.5 mm입니다.

이 값은 작업을 위한 기본 설정이며 모든 환자에게 적용되는 임상 권고가 아닙니다.

#### 진입점과 스크류 길이

모든 후보의 헤드는 스크류 자체 축을 따라 뒤로 옮겨져, 그 중심선이 마지막으로 만나는 뼈 — 실제로 드릴이 시작될 후방 피질골(dorsal cortex) — 위에 놓입니다. traditional 궤적에서는 이것이 후면(posterior surface)의 lateral facet 영역, 즉 횡돌기(transverse process)와 상관절돌기(superior articular process)가 만나는 전형적인 지점에 놓입니다. 플래너는 이 지점을 segmentation mask 위에서 기하학적으로 찾는 것이며, **facet이나 transverse process를 해부학적 랜드마크로 인식하는 것이 아닙니다 — 이 지점을 신뢰하기 전에 반드시 MPR에서 entry를 확인하십시오.**

헤드 뒤쪽으로 스크류 축을 따라 15 mm 이내에 같은 척추의 뼈가 있으면(예: lamina 밑에 있고 그 사이에 공기층이 있는 경우) 도달 불가능한 것으로 보고 후보에서 제외합니다. 실제 드릴은 그 뼈를 먼저 통과해야 하기 때문입니다. 이것이 후방 접근로(dorsal approach) 규칙이며, **Optimizer**가 후보 헤드 위치에 적용하는 유일한 도달 가능성 검사입니다.

헤드가 정해지면, 길이는 팁이 설정된 Anterior margin(기본 4.0 mm)만큼의 뼈를 여전히 앞쪽에 남기는 가장 긴 카탈로그 길이입니다. 이 margin은 팁에서 anterior 척추체 피질골까지 스크류 자체 축을 따라 잰 거리이며, 원위 원통 전체를 둘러싼 여유거리가 아니라 술자가 말하는 그대로의 margin입니다. 각 궤적에서는 실현 가능한 가장 긴 길이만 남기는데, 그렇지 않으면 같은 궤적 위의 더 짧은 스크류가 밀도가 높은 척추경 뼈에 머무른다는 이유로 density 목적함수에서 더 좋은 점수를 받기 때문입니다. 플래너의 점수(Length 가중치 포함, 5.6절 참고)는 궤적 사이의 선택에만 쓰이며, 한 궤적 안에서 길이를 고르는 데는 쓰이지 않습니다.

**Legacy** 플래너와 자동 per-side legacy fallback(5.6절)은 자체 탐색으로 궤적(entry, target, 직경)을 고르고 검증한 다음에야 헤드와 팁을 옮깁니다. 이때 세 가지 후보를 차례로 검토해, 검증된 스크류보다 천공이 늘지 않는 첫 번째 후보를 사용합니다. 첫째는 헤드를 스크류 자체 축을 따라 후방 피질골(dorsal cortex)에 재배치하고 팁을 Anterior margin을 남기는 가장 긴 카탈로그 길이까지 늘린 스크류로, 재배치한 헤드가 Optimizer와 같은 15 mm 후방 접근로 검사를 통과할 때만 후보가 됩니다. 둘째는 원래 헤드에서 팁만 같은 방식으로 늘린 스크류이고, 마지막은 검증된 스크류 그대로입니다. 천공이 늘지 않는다는 것은 medial 천공이 검증된 스크류보다 작거나, 같으면서 전체 천공도 늘지 않는다는 뜻입니다. 검증된 스크류는 항상 이 조건을 만족하므로, 길이를 늘리거나 헤드를 옮기려고 legacy 스크류의 안전성을 낮추는 일은 없습니다. **Cortical bone trajectory(CBT)** 모드는 영향받지 않습니다. CBT는 자신의 진입 랜드마크(5.7절)를 그대로 유지하며, 자체 full-length 실현 가능성 검사로 후보를 선택합니다.

정제된 mask와 기본 설정을 사용한 프로젝트 샘플 연구에서, 이 변경으로 스크류 수는 12개에서 13개로(S1-left가 새로 계획됨), legacy fallback은 1건에서 0건으로, 등급은 A 10 / B 2에서 A 13으로 바뀌었습니다. L5-right의 1.41 mm medial breach가 사라져 모든 스크류에서 0이 되었고, 헤드 매몰은 5–21 mm에서 0–0.2 mm로 줄었습니다. 길이(좌/우)는 L1 35/35, L2 50/45, L3 35/40, L4 40/35, L5 45/35, T12 35/25 mm가 되었으며, 이전에는 대부분의 side가 25 mm로 제한되어 있었습니다(좌측 L1, L2, L4 포함).

### 5.5 계획 매개변수(Planning Parameters)

**Planning** 아래의 접이식 **Planning parameters** 그룹에서는 자동 플래너가 스크류 크기와 위치를 정할 때 사용하는 설정을 조정할 수 있습니다. 각 값은 입력 즉시 검증되고 저장되므로(Qt `QSettings`), 프로그램을 다시 시작해도 유지됩니다.

| 항목 | 기본값 | 의미 |
|---|---:|---|
| Pedicle fill | 0.80 | 측정된 척추경 협부(isthmus) 폭 대비 스크류 직경의 비율 |
| Wall clearance | 0.0 mm | 스크류와 피질골 사이에 유지하는 최소 거리 |
| Anterior margin | 4.0 mm | 팁에서 anterior 척추체 피질골까지 남겨 두는 거리(스크류 축을 따라 측정) |
| Max convergence | 35° | 플래너가 사용할 수 있는 최대 medial 수렴각 |
| HU threshold | 123 HU | 이완(loosening) 위험 경고가 표시되는 궤적 HU 기준값 |
| Narrow pedicle (mm) | 5.0 mm | 이보다 좁으면 카탈로그의 가장 작은 스크류를 계획하고 해당 레벨을 narrow로 표시하는 척추경 폭 기준값 |
| Lateral breach cap (mm) | 2.0 mm | narrow 척추경에서 허용하는 lateral(in-out-in) 천공 한도이며, medial 벽은 절대 뚫지 않음 |
| Parallel to upper endplate | 켜짐 | 궤적을 수평 대신 상위 종판(upper endplate)을 따라 정렬함 |
| Endplate band | 10° | 위 옵션이 켜져 있을 때 최적화기가 종판 방향에서 얼마나 벗어난 각도까지 허용하는지 |
| Construct alignment | 0.30 | 스크류 헤드를 로드에 맞춰 정렬하고 레벨 간 수렴각을 맞추는 데 주어지는 가중치(5.6절 참고) |

**Reset Defaults**를 선택하면 이 열 개 값과 아래에서 설명하는 플래너 모드, 궤적 방식, Safety·Density 목적함수 가중치까지 모두 기본값으로 즉시 복원되고 저장됩니다. Lateral divergence 한계값(−5°, 플래너가 허용하는 가장 lateral한 각도)은 이번 버전에서 고정되어 있으며 패널에 노출되지 않습니다.

### 5.6 궤적 최적화기(Trajectory Optimizer)

Planning Parameters의 **Planner** 콤보박스는 **Plan Screws**가 사용할 두 가지 백엔드 중 하나를 선택합니다.

- **Optimizer(기본값)** — 척추경 하나마다 entry 오프셋 × 수렴각(convergence) × 두미측각(craniocaudal) × 카탈로그 길이로 이루어진 촘촘한 직선 궤적 후보 그리드를 만들고, 모든 후보를 한 번에 채점한 뒤 실현 불가능한 후보를 제외하고, 다섯 개의 정규화된(0–1) 목적함수의 가중합으로 나머지를 순위 매깁니다.
  - **Safety(안전성)** — 최소 피질골 여유거리이며 3 mm에서 포화됩니다.
  - **Density(밀도)** — 궤적 평균 HU를 100–600 HU 범위로 정규화한 값입니다.
  - **Length(길이)** — 카탈로그의 가장 긴 길이 대비 스크류 길이의 비율입니다.
  - **Endplate(종판)** — 상위 종판(upper endplate)과 궤적이 얼마나 평행한지를 15° 허용오차 안에서 나타낸 값입니다.
  - **Centering(중심성)** — 궤적이 척추경 협부(isthmus) 중심에 얼마나 가깝게 지나가는지를 협부 반폭 대비로 나타낸 값입니다.
- **Legacy** — 최적화기가 추가되기 전부터 사용하던 원래의 탐욕적(greedy) entry/target 탐색 방식입니다.

기본 가중치(패널에는 명목 가중치 대비 백분율, 0–300%로 표시): Safety 100%(1.0), Density 50%(0.5), Length 20%(0.2), Endplate 30%(0.3), Centering 30%(0.3)입니다. 여섯 번째 가중치인 **Construct alignment**(기본값 30% / 0.3, 내부적으로는 `rod` 가중치)는 스크류 한 개의 채점에는 영향을 주지 않으며, 아래에서 설명하듯 여러 스크류를 계획할 때 스크류 헤드를 같은 선상에 맞추고 수렴하는 레벨들의 각도를 서로 맞추기 위해 개별 스크류 점수를 얼마나 양보할 수 있는지만 제어합니다. 패널에는 Safety, Density, Construct alignment 가중치만 슬라이더로 노출되며, Length, Endplate, Centering 가중치는 이번 버전에서 기본값으로 고정되어 있습니다.

어떤 후보가 실현 가능하려면 설정된 최소 벽 여유거리를 유지해야 하고, 수렴각이 설정 범위 안에 있어야 하며, 팁 앞쪽에 설정된 Anterior margin만큼의 뼈를 스크류 자체 축을 따라 anterior 척추체 피질골까지 확보해야 합니다 — 원위부 shaft도 나머지 shaft와 마찬가지로 골내 포함(containment) 조건과 설정된 Wall clearance만 만족하면 됩니다(5.4절 "진입점과 스크류 길이" 참고). 척추경 폭이 정상 범위이면 여기에 더해 피질골 천공이 전혀 없어야 하지만, "Narrow pedicle (mm)" 임계값(기본 5.0 mm)보다 좁은 narrow 척추경에서는 대신 최적화기가 카탈로그에서 가장 작은 직경(4.0 mm, **Selected Screw** 패널에 빨간색으로 표시)의 스크류를 배치하며, medial(척추관 쪽) 벽은 그대로 유지하면서 "Lateral breach cap (mm)" 한도(기본 2.0 mm)까지 lateral(in-out-in) 천공을 허용합니다 — 두 스핀박스는 모두 Planning parameters의 Wall clearance 옆에 있습니다. 헤드 자체는 스크류 축을 따라 후방 피질골(dorsal cortex)에 배치되며(5.4절 참고), 그 축을 따라 뒤쪽 15 mm 이내에 같은 척추의 뼈가 여전히 있으면 도달 불가능한 것으로 제외합니다. 실제 드릴이 그 뼈를 먼저 통과해야 하기 때문입니다. 각 궤적에서는 실현 가능한 가장 긴 길이만 순위 산정에 포함되므로, 위의 Length 가중치는 같은 궤적 위의 길이가 아니라 궤적들 사이를 비교합니다. 척추경의 권장 직경보다 카탈로그 기준 두 단계 이내의 어떤 직경으로도 실현 가능한 궤적을 찾지 못하면 해당 척추경은 legacy 방식으로 대신 계획되며, 스크류의 경고 목록에 "Optimizer found no feasible trajectory; legacy planner used"가 추가됩니다. narrow 척추경의 경우 이 legacy 대체 경로도 해당 side를 포기하지 않습니다 — lateral shift 탐색이 찾아낸 것 중 medial 쪽으로 가장 덜 치우친 entry를 그대로 배치하며, 그 탐색 후에도 척추관 쪽 천공이 남아 있으면 조용히 받아들이거나 버리는 대신 스크류에 "Medial breach _x_.x mm — canal side" 경고를 표시합니다. 테스트 결과 Optimizer 모드는 같은 증례를 Legacy 모드로 계획했을 때보다 Gertzbein 등급이 나빠지거나 벽 여유거리가 의미 있게 줄어드는 경우가 없었습니다.

같은 방향(side)에 스크류가 두 개 이상 계획되는 경우, 최적화기는 각 척추경의 상위 후보들을 다시 순위 매겨 스크류 헤드가 공통의 한 직선에 가깝게 놓이고 이웃 레벨의 수렴각이 서로 맞도록 조정합니다 — 이는 로드를 얼마나 구부리고 비틀어야 하는지를 대신 나타내는 값입니다. 이때 Construct alignment 슬라이더 가중치에 따라 스크류 자신의 최고 점수 중 최대 10%까지만 양보할 수 있습니다. 계획이 끝나면 상태 표시줄과 auto-screw 상태 줄에 "Construct: rod fit *x* mm (L), *y* mm (R) · convergence spread *a*° (L), *b*° (R)" 형식으로 결과가 표시되고, 각 스크류의 metrics에는 `score`, `score_components`, `rod_misalignment_mm`, `convergence_deviation_deg`가 함께 기록됩니다. Legacy 방식은 스크류를 각각 따로 배치하므로 조화시킬 대상이 없습니다 — 이 경우 상태 줄에는 대신 "Construct alignment needs Optimizer mode"가 표시됩니다.

실행 시간은 일반적인 CT 기준 척추경 한 개당 약 2–3초이며, 다중 레벨 증례는 척추경을 하나씩 순서대로 계획하기 때문에 그만큼 더 걸립니다. 현재 버전에는 계획 중 레벨별 진행 표시가 없으며, "Analyzing vertebral pedicles...", "Analyzed N vertebrae, M with pedicle data. Planning screws...", "Planned N screw trajectories." 세 개의 큰 단계 메시지만 표시됩니다.

### 5.7 피질골 궤적(Cortical Bone Trajectory, CBT) 모드

Planning Parameters의 **Trajectory** 콤보박스는 **Plan Screws**가 목표로 하는 궤적 방식을 선택합니다.

- **Traditional(기본값)** — 척추경 축을 따라가는 수렴형(convergent) 척추경 나사못이며, 위의 두 백엔드 중 어느 쪽으로도 계획할 수 있습니다.
- **Cortical bone trajectory** — pars/lamina 접합부, 즉 척추경 협부(isthmus)보다 약간 inferior·medial 지점에서 시작해 cranial·lateral 방향으로 척추체 안쪽으로 들어가는 짧고 가는 나사못으로, traditional 궤적과 정반대 방향입니다. 척추경을 가득 채우는 대신 지나가는 경로 상의 피질골 접촉에서 뽑힘 강도(pull-out strength)를 얻으며, 이 점이 골다공증 뼈에서 이 술식이 갖는 장점입니다. CBT를 선택하면 Planner 모드 설정과 무관하게 궤적 탐색 자체가 완전히 바뀌며, 최적화기의 후보 그리드도 legacy 플래너의 medial 탐색도 사용되지 않습니다.

시작 각도는 CBT 문헌을 따릅니다: cranial 각도 ≈25°, lateral 각도 ≈12°이며, 각각 ±5° 범위를 2.5° 간격으로 훑어 가장 점수가 높은 방향을 찾습니다. 임플란트 카탈로그는 직경 5.0/5.5/6.0 mm, 길이 30/35/40 mm로 traditional 카탈로그보다 가늘고 짧습니다. 후보는 safety와 density를 동일한 비중으로 채점해 순위를 매기며(피질골과의 접촉이 이 술식의 핵심이기 때문입니다), 실현 가능하려면 천공이 전혀 없고 설정된 벽 여유거리를 만족해야 합니다. 동점일 경우 더 긴 스크류, 그다음 더 굵은 스크류를 우선합니다.

계획된 모든 CBT 스크류에는 다음과 같은 고정 경고가 함께 표시되어, CT에서 이 술식의 금기사항을 직접 확인하도록 안내합니다. 플래너 스스로는 이를 자동으로 판별할 수 없기 때문입니다: "CBT consensus contraindications: spondylolisthesis grade >= 3, pars defect, absent lamina/isthmus, rotational deformity > 2° (Zhang 2024)". 실현 가능한 CBT 궤적이 없는 방향(side)은 traditional 궤적으로 대체되지 않고 그대로 제외됩니다. 두 술식은 헤드 위치가 서로 다르므로 섞어 쓰면 구조물(construct) 전체가 어긋나기 때문입니다.

CBT 스크류도 traditional 스크류와 완전히 동일한 마무리 경로를 거쳐 등급이 매겨집니다 — 5.9절과 5.10절에서 설명한 것과 같은 mask 기반 Gertzbein-Robbins 천공·벽 여유거리 판정, HU 통계, 골질 경고, facet/Heary 분류를 그대로 사용하므로, CBT 스크류의 등급과 지표는 **Selected Screw** 패널과 내보내기에서 traditional 스크류와 직접 비교할 수 있습니다. 시작 각도의 근거는 Zeng 등(2024, *Orthopaedic Surgery*, CT 기반 CBT 궤적 형태계측 연구)이며, 금기사항 안내문의 근거는 Zhang 등(2024, *Asian Spine Journal*, CBT 적응증에 대한 Delphi 합의)입니다.

### 5.8 Screw MPR 검토

1. 목록, MPR 또는 3D에서 스크류를 선택합니다.
2. **Screw MPR**을 선택합니다.
3. Oblique Axial, Oblique Sagittal, Cross-section 영상을 확인합니다.
4. **Position**을 entry에서 tip 방향으로 이동하며 궤적을 확인합니다.
5. **Std MPR**을 선택하면 기본 단면으로 돌아갑니다.

Screw MPR에는 선택한 스크류가 표시됩니다. Standard MPR에는 현재 단면과 만나는 스크류가 표시됩니다.

#### 스크류 정렬 단면 움직이기

Oblique 단면은 고정되어 있지 않습니다. 스크류 정렬 검토를 벗어나지 않고 스크류 주변을 살펴볼 수 있습니다.

| 조작 | 동작 |
|---|---|
| 마우스 휠 | 스크류 축을 따라 단면 이동(**Position**과 같은 이동) |
| `Shift` + 휠 | 스크류 축을 중심으로 단면 회전 |
| 가운데 버튼 드래그 | 스크류 방향을 유지한 채 단면을 옆으로 평행이동 |
| `Ctrl/Cmd` + 휠 | 확대·축소 |
| 오른쪽 버튼 드래그 | Window/Level 조정 |

각 창의 판독값에 현재 회전각과 오프셋이 표시되므로, 움직여 놓은 화면을 기준 화면으로 착각할 일이 없습니다. 회전은 스크류가 정의하는 평면에서만이 아니라 궤적 전체에 걸쳐 medial 벽을 확인할 때 유용하고, 옆으로의 오프셋은 궤적을 몇 밀리미터 옆의 척추경 벽과 비교할 때 쓸 수 있습니다.

다른 스크류를 선택하거나 **Std MPR**로 갔다가 돌아오면 회전과 오프셋은 0으로 돌아갑니다.

#### 3D에서 보는 Screw MPR

Screw MPR이 활성화되어 있는 동안 3D 화면은 이 모드에서는 어느 창에도 나오지 않는 표준 axial/sagittal/coronal plane 표시를 세 개의 스크류 정렬 plane으로 대체합니다. 각 plane은 스크류를 중심으로 한 80 mm 정사각형으로 그려지며, 그것을 보여주는 창의 머리글 색과 같은 색으로 표시됩니다: Oblique Axial은 axial 창의 색, Oblique Sagittal은 sagittal 창의 색, Cross-section은 coronal 창의 색입니다.

CT volume과 척추 메시는 cross-section에서 잘려 tip 쪽만 남습니다. 뒤에서 바라볼 때 절단면(cut face)은 Cross-section 창에 표시되는 단면과 같습니다. 절단면은 **Position**을 entry에서 tip 방향으로 옮길 때(Cross-section 창에서 마우스 휠을 한 칸 돌릴 때마다 1 mm씩) 함께 이동하며, 3D의 plane들은 회전, 옆으로의 오프셋, 스크류 수정 등 창을 움직이는 모든 조작을 따라갑니다. 스크류 자체는 절대 잘리지 않습니다. Screw MPR이 열려 있는 동안 다시 만들어진 척추 메시(예: segmentation을 다시 실행한 경우)도 같은 평면에서 잘립니다.

**Planes On / Off**(8절)는 현재 표시 중인 plane 세트가 표준이든 스크류 정렬이든 그대로 표시하거나 숨깁니다. **Std MPR**은 절단을 없애고 표준 plane으로 되돌립니다.

### 5.9 스크류 측정값과 등급

**Selected Screw** 패널에는 다음 값이 표시됩니다.

- **Convergence(수렴각):** 정중선 방향으로의 axial 각도이며 부호가 있습니다. 양수는 medial(팁이 정중선을 향함), 음수는 lateral을 의미합니다.
- **Craniocaudal(두미측각):** axial 평면 위로의 궤적 상승각이며 부호가 있습니다. 양수는 cranial입니다. 스크류가 수렴하는 경우 이 값은 sagittal 투영각과 약간 다를 수 있습니다.
- **Endplate(종판각):** 상위 종판(upper endplate) 대비 궤적의 부호 있는 각도이며, 양수는 팁이 cranial 쪽, 0은 평행을 뜻합니다. "Parallel to upper endplate" 계획 옵션을 켜면 설정된 endplate band 안에서 0°를 직접 목표로 삼습니다.
- **Alignment(정렬):** 이 스크류가 다중 스크류 construct 안에서 어떻게 놓이는지 — 로드 선 대비 오프셋(`rod _x_ mm`)과 이웃 레벨과의 수렴각 차이(`conv ±_y_°`)를 나타냅니다. 두 값 모두 개별 스크류가 아니라 해당 쪽 전체를 설명하므로, 스크류를 하나라도 끌면 같은 쪽 모든 스크류에 대해 다시 측정되고, 하나를 지우면 남은 것들로 로드 선을 다시 맞춥니다. Optimizer가 construct를 조정한 경우에만 채워지며(5.6절 참고), Legacy 모드 스크류와 사용자가 직접 배치한 스크류는 이 행이 비어 있습니다.
- **Safety(등급):** TotalSegmentator mask 위에서 원통 표면과 척추 경계 사이 거리로 계산한 Gertzbein-Robbins 등급입니다. 등급 계산에는 HU가 전혀 사용되지 않으며, 함께 표시되는 궤적의 평균·최소 HU는 참고용 정보일 뿐입니다. 해당 스크류의 척추에 TotalSegmentator mask가 없으면 등급은 `N/A`로 표시됩니다.
- **Pedicle(척추경):** 측정된 척추경 협부(isthmus) 폭입니다. 해당 레벨이 "Narrow pedicle" 임계값보다 좁으면 빨간색으로 표시되며, 이는 플래너가 카탈로그에서 가장 작은 직경을 사용하고 medial 벽을 보호하기 위해 lateral(in-out-in) 천공 허용치를 제한했다는 뜻입니다. 폭 뒤에 `?`가 붙고 행에 "width not trusted"라고 표시되면서 같은 빨간색이 나타나는 경우는, 분석기가 그 측정값 자체를 신뢰할 수 없다고 판단한 것입니다(해당 레벨의 타당 범위를 위아래 어느 쪽으로든 벗어난 경우). 두 경우 모두 narrow 정책이 적용되지만, 신뢰할 수 있는 폭만 임상 수치로 인용됩니다.

**진입부 피질골.** 등급 산정은 헤드가 아니라, 스크류의 축이 척추에 처음 들어가는 지점에서 3 mm 지난 곳부터 시작합니다. 후방 피질골(dorsal cortex)에 배치된 헤드(5.4절)는 뼈에 들어가는 표면에 걸쳐 있으므로, 헤드를 기준으로 등급을 매기면 척추경에 도달하기도 전에 모든 스크류가 최대 자기 반경만큼의 breach로 읽히게 됩니다 — 프로젝트 샘플 연구에서 재배치된 12개 스크류 중 11개가 진입부 피질골만으로 B 또는 C 등급을 받았습니다. Gertzbein-Robbins는 척추경 벽을 평가하는 등급이며 진입부 피질골을 평가하는 것이 아니므로, 이 3 mm 진입 구간은 검사에서 제외됩니다. 이 구간은 헤드가 아니라 축이 실제로 뼈에 들어가는 지점을 기준으로 측정됩니다 — 뼈 밖으로 나와 있는 헤드는 축이 척추에 들어가기 전의 공중 구간까지 건너뛰게 되고, 뼈에 묻힌 헤드는 medial breach가 실제로 문제가 되는 협부에 한참 못 미치는 처음 3 mm만 관대하게 처리됩니다. 축이 척추 안으로 전혀 들어가지 않는 스크류는 헤드를 기준으로 등급이 매겨지므로, 그 자체로 breach로 읽힙니다. 같은 3 mm는 등급 산정에서 나오는 모든 수치 — 등급 자체, breach distance와 방향 구분, Wall margin, Trajectory HU 행의 평균/최소 HU(CSV `mean_hu` / `min_hu`) — 에서 동일하게 제외됩니다. 이는 등급이 매겨지는 모든 스크류에 적용됩니다: Optimizer와 Legacy 플래너의 자동 제안, CBT 스크류에 표시되는 등급, 수동으로 배치한 스크류, 드래그나 다른 방식으로 수정한 스크류, segmentation 이후나 계획 불러오기 시 다시 등급이 매겨지는 스크류 — 그러므로 계획된 스크류를 드래그한다고 해서 처음 계획했을 때보다 더 엄격한 규칙으로 다시 등급이 매겨지지 않습니다. 한 가지 결과로, 예전 빌드가 저장한 계획을 지금 다시 등급 매기면(이미 segmentation이 있는 상태에서 불러올 때, 또는 segmentation을 다시 실행한 뒤) 진입부 피질골이 더 이상 채점되지 않으므로 더 좋은 등급이 나올 수 있습니다(9절도 참고).

자동 크기 결정은 직경을 측정된 척추경 협부(isthmus) 폭의 80% 이하로 유지하고, 팁을 스크류 자체 축을 따라 anterior cortex보다 최소 4 mm 뒤쪽에 위치시키며, 길이는 25–55 mm 카탈로그에서 5 mm 간격으로 선택합니다. Wall clearance의 기본값은 이제 0 mm이며, 위에서 설명한 무천공(zero-breach) 요건보다 더 넉넉한 여유를 두고 싶다면 Planning parameters에서 값을 올리면 됩니다. 예전 빌드가 저장한 1.0 mm는 사용자가 고른 값이 아니라 당시 기본값이므로 최초 1회에 한해 0 mm로 이관되며 상태 표시줄에 안내가 나옵니다. 그 밖의 저장값(0.5 mm, 1.5 mm 등)과 이관 이후 사용자가 직접 지정한 1.0 mm는 그대로 존중되어 해당 스핀박스에 계속 표시됩니다. "Parallel to upper endplate" 체크박스와 그 허용오차 스핀박스(endplate band, 기본 10°)는 스크류 크기와 별개로 궤적을 수평 대신 상위 종판(upper endplate)에 맞춰 정렬하며, 두 항목 모두 Planning parameters의 크기 관련 필드 옆에 있습니다. 이 값은 작업을 위한 기본 설정이며 모든 환자에게 적용되는 임상 권고가 아닙니다.

불러온 volume은 표시 전에 LPS(identity 방향)로 재정렬됩니다. Oblique 방식으로 촬영된 volume은 identity 방향 격자로 resampling되며, 이 경우 정보 패널에 "(oblique volume resampled)"가 표시됩니다.

### 5.10 스크류 골질(骨質) 지표(Screw Quality Metrics)

스크류가 segmentation을 기준으로 등급이 매겨지면, **Selected Screw** 패널(Body HU, Wall margin, Facet, Heary 행)과 CSV/JSON 내보내기에 문헌에 근거한 골질·안전성 지표 모음이 표시됩니다.

- **Trajectory HU(평균/최소; 내보내기 필드 `trajectory_mean_hu`(CSV·JSON), `trajectory_min_hu`(JSON `metrics`에만)):** 스크류의 원통형 궤적 전체를 따라(진입 구간 포함) 측정한 HU(Hounsfield unit)의 평균값과 최소값입니다. 이는 3 mm 진입 구간을 제외하는 Selected Screw 패널의 **Trajectory HU** 행(CSV `mean_hu` / `min_hu`, 5.9절 참고)과는 별개의 수치입니다.
- **Pedicle HU:** 척추경 협부(isthmus) 중심에서 10 mm 이내에 있는 궤적 샘플만으로 계산한 평균 HU입니다. 자동 계획된 스크류에서만 제공됩니다 — 수동 스크류는 기준이 될 isthmus 중심이 없기 때문입니다.
- **Vertebral body HU(척추체 HU):** 척추체 중심에 위치한 8×8×6 mm 타원체 관심영역을 해당 척추의 segmentation label과 교차시켜 계산한 평균 HU입니다. 같은 이유로 자동 계획된 스크류에서만 제공됩니다.
- **Trajectory/body HU 비율:** 궤적 평균 HU를 척추체 HU로 나눈 값입니다.
- **최소 피질골 여유거리("Wall margin"):** 스크류와 피질골 사이의 가장 가까운 거리(mm)입니다(3 mm 진입 구간 이후, 5.9절 참고).
- **Heary breach 방향:** 가장 심한 피질골 천공의 해부학적 방향 — medial, lateral, anterior, posterior, superior, inferior 중 하나입니다(Heary 2004). Side 정보가 없는 수동 스크류에서 medial/lateral 방향의 breach가 발생하면 "mediolateral"로, breach가 없으면 "none"으로 표시됩니다.
- **후관절(facet) 침범 등급(0–3):** Babu(2012) 등급을 근사한 값으로, 스크류 근위부(entry에서 가까운 1/3) 구간과 상위(cephalad) 척추의 segmentation label 사이 관계로 판정합니다 — 0은 접촉 없음, 1은 1 mm 이내로 후관절에 접함, 2는 1 mm 미만으로 침범, 3은 1 mm 이상 침범을 의미합니다.

측정값이 문헌 기준값을 넘으면 패널과 내보내기에 경고가 함께 표시됩니다.

| 지표 | 기준값 | 경고 | 참고문헌 |
|---|---|---|---|
| Trajectory HU (`trajectory_mean_hu`) | 123 HU 미만 | 이완(loosening) 위험 | Yamamoto 2025; Dhar 2026 |
| Vertebral body HU | 132 HU 미만 | 골다공증 | Sankar 2026 |
| Vertebral body HU | 141 HU 미만(골다공증에 해당하지 않는 경우) | 저골밀도 | Sankar 2026 |
| Trajectory/body HU 비율 | 1.0 미만 | 이완 위험 | Yang 2026 |
| 후관절 침범 등급 | 2 이상 | 후관절 침범 | Babu 2012 |

이러한 골질 관련 경고는 자동 계획된 스크류에서만 생성됩니다. 수동으로 배치한 스크류도 전체 지표 모음은 제공받지만 골질 경고는 받지 않습니다. Segmentation을 다시 실행하거나 계획을 불러올 때(이미 segmentation이 있는 경우) 계획을 다시 등급 매기면 모든 스크류의 breach distance·피질골 여유 경고가 다시 생성됩니다.

### 5.11 척추경 세부영역(Subregion) 모델(선택 사항)

Segmentation → Advanced에는 기본적으로 꺼져 있는 **Use pedicle subregion model** 옵션이 있습니다. 이를 켜면 척추를 pedicle/corpus/lamina/spinous/transverse/articular 세부영역으로 분할하는 로컬 nnU-Net 모델([MICN-Lab/Spine_Subregions](https://github.com/MICN-Lab/Spine_Subregions); Da Mutten et al., *J Imaging Inform Med* 2026)을 **Model directory** 입력란(또는 `PSS_SUBREGION_MODEL_DIR` 환경변수)이 가리키는 경로에서 찾습니다. 이 경로는 `dataset.json`과 `fold_*/checkpoint_final.pth`가 있는 nnU-Net results 폴더여야 합니다.

이 기능은 소스 설치 전용입니다. `nnunetv2`가 설치된 non-frozen Python 환경이 필요하며, 메모리 요구량은 TotalSegmentator의 `3d_fullres` 설정과 비슷합니다(GPU 권장). macOS·Windows standalone 패키지는 이 기능을 지원하지 않습니다. [데스크톱 빌드 안내](BUILDING_DESKTOP.ko.md)를 참고하십시오. 참고: 현재 upstream에 공개된 Spine_Subregions release는 이 프로그램이 기대하는 nnU-Net v2 폴더 구조가 아니라 nnU-Net v1 방식의 폴더 구조를 사용하므로, 재출력(재변환)되기 전까지는 인식되지 않습니다 — 자세한 내용은 데스크톱 빌드 안내를 참고하십시오.

모델이 정상적으로 실행되면 각 방향의 척추경 협부(isthmus)를 먼저 해당 label에서 측정하고, label이 그 방향을 찾지 못한 경우에만 coronal 단면 탐색으로 대체합니다. Segmentation 상태 표시줄에는 "· pedicle model used"가 표시되거나, 모델은 실행되었지만 결과에 pedicle label이 전혀 없을 때는 "· pedicle model ran but found no pedicle voxels"가, 실행되지 못했을 때는 이유와 함께 "· pedicle model unavailable: <reason>"이 표시됩니다 — 2단계 실패가 그 아래 TotalSegmentator 결과 자체를 막지는 않습니다.

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
| 머리글 더블클릭 또는 `Ctrl+M` | 이 창을 최대화, 다시 누르면 복원 |

Screw MPR에서는 휠과 가운데 버튼 드래그가 스크류 정렬 단면을 움직입니다. 5.8절을 참고하십시오.

### 3D

| 조작 | 기능 |
|---|---|
| 왼쪽 드래그 | 회전 |
| 마우스 휠 | 확대·축소 |
| `Pan` 후 드래그 또는 Shift+드래그 | 모델 이동 |
| 해부학 구조 더블클릭 | 해당 위치로 초점 이동 |
| `Reset View` | 초기 sagittal 방향 복원 |
| `Vertebra Transparency` | 내부 스크류 표시 정도 조절 |
| `Planes On / Off` | MPR plane 표시·숨김 — Screw MPR이 활성화되어 있으면 스크류 정렬 plane |
| 머리글 더블클릭 또는 `Ctrl+M` | 3D 창을 최대화, 다시 누르면 복원 |

Screw MPR이 이 plane들의 내용과 절단 방식을 어떻게 바꾸는지는 5.8절을 참고하십시오.

볼륨 렌더링은 Windows와 Linux에서 GPU를, macOS에서는 CPU 레이캐스터를 사용합니다. macOS는 OpenGL→Metal 변환 계층이 3D 텍스처 업로드에서 멈추기 때문입니다. 선택은 자동이며 로그에 기록됩니다.

## 9. 저장과 내보내기

- 계획을 JSON 형식으로 저장하고 불러옵니다.
- 지원되는 계획 표를 CSV로 내보냅니다.
- 지원되는 골 표면을 STL로 내보냅니다.

계획 파일은 schema version 3을 사용하며, 각 스크류에 5.10절 "스크류 골질 지표"에서 설명한 골질·안전성 지표(trajectory/pedicle/body HU, HU 비율, 최소 wall 거리, Heary breach 방향, facet 침범 등급)를 담는 `metrics` 필드가 추가되었습니다. schema version 2에서는 각 스크류에 `mean_hu`, `min_hu`, `warnings`, `source`가 추가되었습니다. 이전 버전으로 저장한 계획 파일도 계속 불러올 수 있으며, 이미 segmentation이 있는 상태에서 계획을 불러오면 즉시 다시 등급이 매겨져 schema v3 지표가 채워집니다. 이렇게 다시 등급이 매겨지면 진입부 피질골이 더 이상 채점되지 않으므로(5.9절 참고) 예전 빌드가 저장한 계획이 더 좋은 등급으로 나올 수 있습니다 — 계획 파일 자체는 바뀌지 않습니다. CSV 내보내기에는 이름이 변경된 `convergence_angle_deg`, `craniocaudal_angle_deg` 열, `mean_hu`, `min_hu`, `source`, `warnings` 열과 함께 schema v3 지표 열인 `trajectory_mean_hu`, `pedicle_mean_hu`, `body_mean_hu`, `hu_ratio`, `min_wall_mm`, `heary_direction`, `facet_grade`가 포함됩니다.

계획 파일, 스크린샷과 3D 메시는 DICOM 헤더가 없어도 환자와 연결될 수 있으므로 공유 전에 확인하십시오.

## 10. 문제 해결

### CT가 너무 작게 보임

`+`를 사용하거나 `Pan`을 활성화해 이동한 후 `Fit`을 누릅니다. 전체 MPR은 **Fit MPR**로 초기화합니다.

### TotalSegmentator가 느리거나 메모리가 부족함

- 가능한 경우 CUDA GPU를 사용합니다.
- 메모리를 많이 사용하는 다른 프로그램을 종료합니다.
- 최초 모델 다운로드가 끝날 때까지 기다립니다. 이후에는 캐시된 모델을 재사용합니다.
- 저해상도 segmentation은 경계 정확도를 낮출 수 있습니다.

### 3D 화면이 느리거나 끊김

사용 가능한 GPU 컨텍스트가 없으면 — 원격 데스크톱, 가상 머신, 소프트웨어 OpenGL 환경 — 볼륨 렌더링이 CPU 레이캐스팅으로 전환됩니다. 프로그램은 이를 감지해 볼륨을 자동으로 다시 다운샘플링하므로 화면은 세부 묘사를 줄인 채 반응성을 유지합니다. 이때 로그에 `render mode=cpu-raycast`가 기록됩니다. 완전한 화질로 보려면 GPU 드라이버가 정상인 컴퓨터에서 직접 실행하십시오. `Vertebra Transparency`를 낮추거나 `Planes Off`로 두어도 부하가 줄어듭니다.

### 특정 스크류가 생성되지 않음

허용되는 골내 궤적을 찾지 못하면 해당 방향을 건너뜁니다. Segmentation을 확인하고 스크류를 수동으로 추가하거나 수정하십시오.

### 실행 문제

```bash
./scripts/run_app.sh --check
```

로그 위치는 실행 방식에 따라 다릅니다.

- **소스 빌드(`python main.py` 또는 `run_app.sh`로 실행)**: 프로젝트 디렉터리 안의 `logs/app.log`.
- **macOS 패키지 빌드**: `~/Library/Logs/PedicleScrewSimulator/app.log`.
- **Windows 패키지 빌드**: `%LOCALAPPDATA%\PedicleScrewSimulator\logs\app.log`.

해당 파일의 마지막 부분을 확인하십시오. 예시:

```bash
tail -100 logs/app.log
```

```powershell
Get-Content "$env:LOCALAPPDATA\PedicleScrewSimulator\logs\app.log" -Tail 100
```

문제를 보고할 때 운영체제, Python 버전, 재현 과정 및 비식별화한 로그를 포함하십시오. 임상 데이터를 첨부하지 마십시오.

## 11. 버전, 제작자 및 학술 인용

**Help → About Pedicle Screw Simulator**에서 설치된 버전, 제작자, 소속, 이메일, 홈페이지, 소스 저장소, MIT License와 연구용 안내를 확인할 수 있습니다.

- 제작자: Sang-Min Park, MD, Ph.D.
- 조직: 분당서울대학교병원
- 학술 소속: 서울대학교 의과대학
- 홈페이지: [https://sangmin.me](https://sangmin.me)
- 학술 인용: [`CITATION.cff`](../CITATION.cff) 참고
- 변경 이력: [`CHANGELOG.ko.md`](../CHANGELOG.ko.md) 참고

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
