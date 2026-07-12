# Pedicle Screw Simulator

[English](README.md) | [한국어](README.ko.md)

DICOM CT 영상 확인, 척추 자동 분할, 척추경 나사못 계획 및 MPR·3D 검토를 위한 연구용 데스크톱 프로그램입니다.

**버전:** 0.1.0

**주요 검증 환경:** macOS, Python 3.12

**개발자:** Dr. Sang-Min Park

> **연구 및 교육 목적으로만 사용하십시오.** 이 프로그램은 인증된 의료기기가 아니며 진단, 수술, 내비게이션 또는 환자 진료의 단독 근거로 사용할 수 없습니다. 자동 분할, 스크류 제안, 길이·직경, breach grade와 경고는 반드시 자격을 갖춘 의료진이 독립적으로 확인해야 합니다.

## 주요 기능

- 여러 시리즈를 포함한 DICOM CT 불러오기
- Axial, Sagittal, Coronal MPR 동기화
- VTK 기반 CPU 볼륨 렌더링 및 척추별 3D 메시 표시
- GPU 우선 TotalSegmentator 연동 및 CPU 재시도
- 여러 척추 레벨 선택과 자동 스크류 제안
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

### TotalSegmentator 선택 설치

```bash
./scripts/run_app.sh --with-totalseg
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

현재 자동 테스트는 **454개**입니다.

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
- 현재 네이티브 실행은 주로 macOS에서 검증했으며 Windows/Linux 설치 패키지는 아직 제공하지 않습니다.
- TotalSegmentator는 선택 기능이며 모델 다운로드와 많은 메모리가 필요할 수 있습니다.
- 임상 정확도, 관찰자 간 일치도 및 전향적 결과는 아직 확립되지 않았습니다.

## 라이선스 상태

오픈소스 라이선스는 아직 선택하지 않았습니다. 라이선스가 추가되기 전에는 기본 저작권 제한이 적용됩니다. 외부 재배포나 기여를 허용하기 전에 PyQt6 라이선스와 배포 방식을 검토해야 합니다.
