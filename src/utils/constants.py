"""
Constants and configuration for the Screw Fixation Simulator
"""

import numpy as np

# Coordinate System Transform: DICOM LPS to VTK RAS
# DICOM uses LPS (Left, Posterior, Superior)
# VTK uses RAS (Right, Anterior, Superior)
LPS_TO_RAS_MATRIX = np.array([
    [-1,  0,  0,  0],
    [ 0, -1,  0,  0],
    [ 0,  0,  1,  0],
    [ 0,  0,  0,  1]
], dtype=np.float64)

# CT Hounsfield Unit thresholds
HU_BONE_MIN = 300      # Minimum HU for bone
HU_BONE_OPTIMAL = 400  # Optimal threshold for cortical bone
HU_BONE_MAX = 3000     # Maximum HU for bone

# Default Window/Level for spine CT
DEFAULT_WINDOW_CENTER = 400
DEFAULT_WINDOW_WIDTH = 1500

# Screw parameters (mm)
DEFAULT_SCREW_LENGTH = 45.0
DEFAULT_SCREW_DIAMETER = 6.5
MIN_SCREW_LENGTH = 20.0
MAX_SCREW_LENGTH = 70.0
MIN_SCREW_DIAMETER = 4.0
MAX_SCREW_DIAMETER = 7.5

# Gertzbein-Robbins Grading
GRADE_A_DESCRIPTION = "Completely intrapedicular (optimal)"
GRADE_B_DESCRIPTION = "Breach <2mm (acceptable)"
GRADE_C_DESCRIPTION = "Breach 2-4mm (unsafe)"
GRADE_D_DESCRIPTION = "Breach 4-6mm (unsafe)"
GRADE_E_DESCRIPTION = "Breach >6mm (dangerous)"

# MPR Plane Direction Cosines
# Axial: radiological convention (looking from the patient's feet).
# Row i of this flat list becomes column i of the reslice axes
# (screen-right / screen-up / normal): screen-right=+X (patient left on the
# viewer's right), screen-up=-Y (anterior at the top), normal=+Z (scroll
# unchanged).
AXIAL_DIRECTION_COSINES = [
    1,  0, 0,   # Row 0
    0, -1, 0,   # Row 1
    0,  0, 1    # Row 2
]

# Coronal: XZ plane (looking from front)
# Row i of this flat list becomes column i of the reslice axes
# (screen-right / screen-up / normal): screen-right=X(L-R),
# screen-up=Z(superior-up), normal=-Y(A-P).
CORONAL_DIRECTION_COSINES = [
    1, 0,  0,   # Row 0
    0, 0, -1,   # Row 1
    0, 1,  0    # Row 2
]

# Sagittal: YZ plane (looking from side)
# Row i of this flat list becomes column i of the reslice axes
# (screen-right / screen-up / normal): screen-right=Y(A-P),
# screen-up=Z(superior-up), normal=X(L-R).
SAGITTAL_DIRECTION_COSINES = [
    0, 0, 1,   # Row 0
    1, 0, 0,   # Row 1
    0, 1, 0    # Row 2
]

# UI Colors (RGB, 0-255)
COLOR_AXIAL = (255, 255, 0)      # Yellow
COLOR_SAGITTAL = (0, 255, 255)   # Cyan
COLOR_CORONAL = (255, 0, 255)    # Magenta
COLOR_SCREW = (30, 190, 235)     # Surgical cyan contrasts warm anatomy
COLOR_SCREW_BREACH = (255, 0, 0) # Red

# Viewport IDs
VIEWPORT_AXIAL = "axial"
VIEWPORT_SAGITTAL = "sagittal"
VIEWPORT_CORONAL = "coronal"
VIEWPORT_3D = "3d"


# ============================================================
# Transfer Function Presets for CPU Volume Rendering
# ============================================================
# Each preset defines:
#   color_points: list of (HU, R, G, B) — RGB in 0.0–1.0
#   opacity_points: list of (HU, opacity) — opacity in 0.0–1.0
#   gradient_opacity_points: list of (gradient_mag, opacity)
#   shade: bool — enable/disable Phong shading
#   ambient / diffuse / specular / specular_power: Phong params
#   blend_mode: "composite" or "maximum_intensity"

TRANSFER_FUNCTION_PRESETS = {
    "Bone": {
        "color_points": [
            (-1000, 0.0, 0.0, 0.0),       # Air: black
            (-100, 0.55, 0.25, 0.15),      # Soft tissue: brown
            (200, 0.88, 0.70, 0.55),       # Muscle/cartilage: tan
            (400, 0.95, 0.92, 0.82),       # Trabecular bone: ivory
            (1500, 1.0, 1.0, 0.95),        # Cortical bone: white
            (3000, 1.0, 1.0, 1.0),         # Dense bone/metal: bright white
        ],
        "opacity_points": [
            (-1000, 0.0),
            (100, 0.0),
            (200, 0.02),
            (350, 0.08),
            (500, 0.45),
            (1000, 0.70),
            (3000, 0.85),
        ],
        "gradient_opacity_points": [
            (0, 0.0),
            (30, 0.1),
            (80, 0.6),
            (200, 1.0),
        ],
        "shade": True,
        "ambient": 0.2,
        "diffuse": 0.7,
        "specular": 0.3,
        "specular_power": 16.0,
        "blend_mode": "composite",
    },
    "Soft Tissue": {
        "color_points": [
            (-1000, 0.0, 0.0, 0.0),
            (-500, 0.15, 0.05, 0.05),
            (-100, 0.55, 0.25, 0.20),
            (0, 0.75, 0.45, 0.35),
            (100, 0.85, 0.60, 0.50),
            (300, 0.92, 0.80, 0.70),
            (1000, 1.0, 1.0, 0.95),
        ],
        "opacity_points": [
            (-1000, 0.0),
            (-200, 0.0),
            (-100, 0.05),
            (0, 0.20),
            (100, 0.35),
            (300, 0.50),
            (1000, 0.60),
        ],
        "gradient_opacity_points": [
            (0, 0.0),
            (20, 0.2),
            (60, 0.7),
            (150, 1.0),
        ],
        "shade": True,
        "ambient": 0.3,
        "diffuse": 0.6,
        "specular": 0.2,
        "specular_power": 10.0,
        "blend_mode": "composite",
    },
    "CT Angiography": {
        "color_points": [
            (-1000, 0.0, 0.0, 0.0),
            (100, 0.55, 0.25, 0.20),
            (200, 0.85, 0.15, 0.10),
            (300, 1.0, 0.2, 0.15),
            (500, 1.0, 0.85, 0.70),
            (1500, 1.0, 1.0, 0.95),
        ],
        "opacity_points": [
            (-1000, 0.0),
            (100, 0.0),
            (200, 0.30),
            (300, 0.55),
            (500, 0.45),
            (1500, 0.65),
        ],
        "gradient_opacity_points": [
            (0, 0.0),
            (40, 0.3),
            (100, 0.8),
            (200, 1.0),
        ],
        "shade": True,
        "ambient": 0.2,
        "diffuse": 0.7,
        "specular": 0.4,
        "specular_power": 20.0,
        "blend_mode": "composite",
    },
    "MIP (Maximum Intensity)": {
        "color_points": [
            (-1000, 0.0, 0.0, 0.0),
            (0, 0.3, 0.3, 0.3),
            (500, 0.7, 0.7, 0.7),
            (1500, 1.0, 1.0, 1.0),
            (3000, 1.0, 1.0, 1.0),
        ],
        "opacity_points": [
            (-1000, 0.0),
            (-200, 0.0),
            (0, 0.1),
            (500, 0.5),
            (1500, 0.8),
            (3000, 1.0),
        ],
        "gradient_opacity_points": [
            (0, 1.0),
            (255, 1.0),
        ],
        "shade": False,
        "ambient": 1.0,
        "diffuse": 0.0,
        "specular": 0.0,
        "specular_power": 1.0,
        "blend_mode": "maximum_intensity",
    },
}

# Implant catalogue used by the automatic planner (mm)
IMPLANT_LENGTHS_MM = (25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0)
IMPLANT_DIAMETERS_MM = (4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5)
# Sizing rules (Götschi 2026, Wang 2024): diameter <= 80 % of isthmus width and
# >= 1 mm cortical clearance each side; tip >= 4 mm behind the anterior cortex.
PEDICLE_FILL_RATIO = 0.80
CORTICAL_WALL_CLEARANCE_MM = 1.0
ANTERIOR_SAFETY_MARGIN_MM = 4.0

# Trajectory HU below which screw loosening risk rises (Yamamoto 2025, Dhar 2026)
TRAJECTORY_HU_LOOSENING_THRESHOLD = 123.0
# L1-L5 trabecular HU thresholds from TotalSegmentator-derived ROIs (Sankar 2026)
VERTEBRAL_HU_LOW_BMD_THRESHOLD = 141.0
VERTEBRAL_HU_OSTEOPOROSIS_THRESHOLD = 132.0
# Trajectory / body HU ratio below which loosening odds rise (Yang 2026)
TRAJECTORY_BODY_HU_RATIO_THRESHOLD = 1.0
# Facet violation: distance from screw head cylinder to the cephalad vertebra (Babu 2012 approximation)
FACET_CONTACT_DISTANCE_MM = 1.0

# ------------------------------------------------------------------
# Cortical bone trajectory (CBT / mCBT)
# ------------------------------------------------------------------
# Starting angles from Zeng 2024 (3D-CT morphometry of lumbar CBT corridors:
# cranial 22-25 deg, lateral 11-14 deg) with the screw catalogue CBT actually
# uses -- short, narrow screws that buy their purchase from cortex rather than
# from pedicle fill.
CBT_DEFAULTS = {
    "cranial_angle_deg": 25.0,        # 22-25°
    "lateral_angle_deg": 12.0,        # 11-14°
    "angle_search_deg": 5.0,          # ± search window
    "diameter_mm": (5.0, 5.5, 6.0),
    "lengths_mm": (30.0, 35.0, 40.0),
}
# Zhang 2024 Delphi consensus on when CBT must not be used; attached to every
# planned CBT screw so the reviewer is reminded to rule these out on the CT.
CBT_CONTRAINDICATION_NOTE = ("CBT consensus contraindications: spondylolisthesis grade >= 3, pars defect, "
                             "absent lamina/isthmus, rotational deformity > 2° (Zhang 2024)")
