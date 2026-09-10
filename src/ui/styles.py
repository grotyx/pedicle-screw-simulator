"""Selectable flat-color QSS themes for the planning workstation."""

DEFAULT_THEME = "graphite_blue"

THEME_LABELS = {
    "soft_light": "Soft Light",
    "graphite_blue": "Graphite Blue",
    "graphite_mint": "Graphite Mint",
}

THEMES = {
    "soft_light": {
        "bg_primary": "#F3F5F7",
        "bg_secondary": "#FFFFFF",
        "bg_tertiary": "#F7F8FA",
        "bg_toolbar": "#FFFFFF",
        "accent": "#3478F6",
        "accent_hover": "#2469E8",
        "accent_pressed": "#1F58C7",
        "accent_dim": "#E8F0FF",
        "implant": "#1EBEEB",
        "text_primary": "#1B2128",
        "text_secondary": "#69737E",
        "text_disabled": "#A4ABB3",
        "border": "#D8DDE3",
        "border_light": "#B8C1CB",
        "danger": "#D8424D",
        "success": "#16875D",
        "warning": "#BD7100",
        "button_text": "#FFFFFF",
        "secondary_action_text": "#245FB7",
        "warning_bg": "#FFF7E6",
        "warning_text": "#9B5C00",
        "danger_bg": "#FFF0F1",
        "danger_border": "#F0B8BD",
        "danger_text": "#B82F3A",
        "viewer_header": "#12161B",
        "viewer_readout": "#0B0E12",
        "viewer_bg": "#050A0F",
        "viewer_foreground": "#E8EDF2",
        "viewer_separator": "#B8C1CB",
        "grade_a": "#2E9E5B",
        "grade_b": "#8FBF3F",
        "grade_c": "#E0A326",
        "grade_d": "#D64545",
        "grade_na": "#8D99A5",
        "grade_text": "#FFFFFF",
        "table_alternate": "#F1F3F5",
        "table_header": "#ECEFF2",
        "scrollbar_bg": "#E7EAEE",
        "scrollbar_handle": "#B7C0C9",
        "scrollbar_hover": "#8D99A5",
    },
    "graphite_blue": {
        "bg_primary": "#0D0F12",
        "bg_secondary": "#15181C",
        "bg_tertiary": "#1C2025",
        "bg_toolbar": "#111419",
        "accent": "#4C8DFF",
        "accent_hover": "#65A0FF",
        "accent_pressed": "#3776DB",
        "accent_dim": "#202D42",
        "implant": "#35C7F0",
        "text_primary": "#EEF1F4",
        "text_secondary": "#9AA4AF",
        "text_disabled": "#5E6873",
        "border": "#2B3138",
        "border_light": "#46505B",
        "danger": "#FF5D6C",
        "success": "#4ED19A",
        "warning": "#FFB020",
        "button_text": "#FFFFFF",
        "secondary_action_text": "#BCD5FF",
        "warning_bg": "#2B2417",
        "warning_text": "#FFD17A",
        "danger_bg": "#352025",
        "danger_border": "#7D3B45",
        "danger_text": "#FFBBC2",
        "viewer_header": "#15181C",
        "viewer_readout": "#101317",
        "viewer_bg": "#050A0F",
        "viewer_foreground": "#E8EDF2",
        "viewer_separator": "#2B3138",
        "grade_a": "#35B36B",
        "grade_b": "#9BCB4C",
        "grade_c": "#E8AE35",
        "grade_d": "#E05561",
        "grade_na": "#5E6873",
        "grade_text": "#0B0E12",
        "table_alternate": "#181C21",
        "table_header": "#111419",
        "scrollbar_bg": "#15181C",
        "scrollbar_handle": "#343B44",
        "scrollbar_hover": "#4B5662",
    },
    "graphite_mint": {
        "bg_primary": "#10110F",
        "bg_secondary": "#181A17",
        "bg_tertiary": "#20231F",
        "bg_toolbar": "#141613",
        "accent": "#5CC8AD",
        "accent_hover": "#70D6BC",
        "accent_pressed": "#45A98F",
        "accent_dim": "#1F3932",
        "implant": "#35C7F0",
        "text_primary": "#F0F2ED",
        "text_secondary": "#A2AAA0",
        "text_disabled": "#626A60",
        "border": "#30352F",
        "border_light": "#4B544A",
        "danger": "#F56A72",
        "success": "#65D49D",
        "warning": "#F0AD4E",
        "button_text": "#07110E",
        "secondary_action_text": "#A7E6D6",
        "warning_bg": "#2C2518",
        "warning_text": "#F4CB86",
        "danger_bg": "#372224",
        "danger_border": "#804247",
        "danger_text": "#FFC0C4",
        "viewer_header": "#181A17",
        "viewer_readout": "#111310",
        "viewer_bg": "#050A0F",
        "viewer_foreground": "#E8EDF2",
        "viewer_separator": "#30352F",
        "grade_a": "#3FBE7B",
        "grade_b": "#A3D255",
        "grade_c": "#EDB645",
        "grade_d": "#E76A72",
        "grade_na": "#626A60",
        "grade_text": "#07110E",
        "table_alternate": "#1B1E1A",
        "table_header": "#141613",
        "scrollbar_bg": "#181A17",
        "scrollbar_handle": "#383E37",
        "scrollbar_hover": "#50594F",
    },
}


def get_theme(theme_name: str = DEFAULT_THEME) -> dict[str, str]:
    """Return a complete palette, falling back to the default theme."""
    return THEMES.get(str(theme_name), THEMES[DEFAULT_THEME])


def theme_rgb_float(theme_name: str, key: str) -> tuple[float, float, float]:
    """Return one palette colour as VTK-style floats in 0.0-1.0.

    Args:
        theme_name: Palette name; unknown names fall back to DEFAULT_THEME.
        key: Palette key, e.g. ``"viewer_foreground"``.

    Returns:
        (red, green, blue) each in 0.0-1.0; white if the key is missing.
    """
    value = get_theme(theme_name).get(key, "#FFFFFF").lstrip("#")
    if len(value) != 6:
        return (1.0, 1.0, 1.0)
    return (
        int(value[0:2], 16) / 255.0,
        int(value[2:4], 16) / 255.0,
        int(value[4:6], 16) / 255.0,
    )


def load_stylesheet(theme_name: str = DEFAULT_THEME) -> str:
    """Return the application stylesheet for one named palette."""
    t = get_theme(theme_name)
    return f"""
/* ===== Global ===== */
QMainWindow, QWidget {{
    background-color: {t["bg_primary"]};
    color: {t["text_primary"]};
    font-family: "Segoe UI", "SF Pro Text", "Avenir Next", "Helvetica Neue", Arial;
    font-size: 13px;
}}

QWidget#collapsibleSection {{
    background-color: {t["bg_secondary"]};
    border: 1px solid {t["border"]};
    border-radius: 9px;
    margin: 3px 2px;
}}
QWidget#collapsibleSection[role="review"] {{
    background-color: {t["bg_secondary"]};
    border: 2px solid {t["accent"]};
    border-radius: 10px;
}}
QWidget#collapsibleSection[role="review"] QToolButton#sectionHeader {{
    background-color: {t["accent_dim"]};
    color: {t["secondary_action_text"]};
    font-size: 13px;
    padding: 10px 11px;
}}
QToolButton#sectionHeader {{
    background: transparent;
    border: none;
    color: {t["text_primary"]};
    font-size: 12px;
    font-weight: 700;
    padding: 9px 10px;
    text-align: left;
}}
QToolButton#sectionHeader:hover {{
    background-color: {t["bg_tertiary"]};
    color: {t["text_primary"]};
}}
QToolButton#vertebraSelector {{
    background-color: {t["bg_tertiary"]};
    color: {t["text_primary"]};
    border: 1px solid {t["border"]};
    border-radius: 4px;
    padding: 5px 10px;
    min-height: 22px;
    text-align: left;
}}
QToolButton#vertebraSelector:hover,
QToolButton#vertebraSelector:pressed {{
    border-color: {t["accent"]};
    background-color: {t["accent_dim"]};
}}
QToolButton#vertebraSelector::menu-indicator {{
    subcontrol-position: right center;
    subcontrol-origin: padding;
    right: 6px;
}}
QWidget#sectionContent {{
    background-color: {t["bg_secondary"]};
}}

/* ===== Group Boxes ===== */
QGroupBox {{
    background-color: {t["bg_secondary"]};
    border: 1px solid {t["border"]};
    border-radius: 6px;
    margin-top: 14px;
    padding: 12px 8px 8px 8px;
    font-weight: 600;
    font-size: 12px;
    color: {t["text_secondary"]};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
    background-color: {t["bg_secondary"]};
    color: {t["text_secondary"]};
}}

/* ===== Labels ===== */
QLabel {{
    color: {t["text_primary"]};
    background: transparent;
    padding: 1px;
}}

/* ===== Push Buttons ===== */
QPushButton {{
    background-color: {t["bg_tertiary"]};
    color: {t["text_primary"]};
    border: 1px solid {t["border"]};
    border-radius: 4px;
    padding: 5px 14px;
    min-height: 22px;
    font-weight: 500;
}}
QPushButton:hover {{
    background-color: {t["accent"]};
    border-color: {t["accent"]};
    color: {t["button_text"]};
}}
QPushButton:pressed {{
    background-color: {t["accent_pressed"]};
}}
QPushButton:checked {{
    background-color: {t["accent"]};
    border-color: {t["accent_hover"]};
    color: {t["button_text"]};
}}
QPushButton:disabled {{
    background-color: {t["bg_secondary"]};
    color: {t["text_disabled"]};
    border-color: {t["bg_secondary"]};
}}
QPushButton[role="primary"] {{
    background-color: {t["accent"]};
    border-color: {t["accent"]};
    color: {t["button_text"]};
    font-weight: 700;
}}
QPushButton[role="primary"]:hover {{
    background-color: {t["accent_hover"]};
    border-color: {t["accent_hover"]};
}}
QPushButton[role="secondary"] {{
    background-color: {t["accent_dim"]};
    border-color: {t["border_light"]};
    color: {t["secondary_action_text"]};
}}
QPushButton[role="warning"] {{
    background-color: {t["warning_bg"]};
    border-color: {t["warning"]};
    color: {t["warning_text"]};
}}
QPushButton[role="danger"] {{
    background-color: {t["danger_bg"]};
    border-color: {t["danger_border"]};
    color: {t["danger_text"]};
}}
QPushButton[role="danger"]:hover {{
    background-color: {t["danger"]};
    border-color: {t["danger"]};
    color: {t["button_text"]};
}}

/* ===== Tool Bar ===== */
QToolBar {{
    background-color: {t["bg_toolbar"]};
    border: none;
    border-bottom: 1px solid {t["border"]};
    spacing: 6px;
    padding: 6px 8px;
}}
QComboBox#workspaceMode, QComboBox#themeSelector {{
    background-color: {t["accent_dim"]};
    border-color: {t["accent"]};
    color: {t["secondary_action_text"]};
    font-weight: 700;
}}
QComboBox#workspaceMode {{
    min-width: 88px;
}}
QComboBox#themeSelector {{
    min-width: 100px;
}}
QToolBar QPushButton {{
    min-width: 72px;
    border-radius: 3px;
}}

/* ===== Menu Bar ===== */
QMenuBar {{
    background-color: {t["bg_toolbar"]};
    color: {t["text_primary"]};
    border-bottom: 1px solid {t["border"]};
    padding: 2px;
}}
QMenuBar::item:selected {{
    background-color: {t["accent"]};
    color: {t["button_text"]};
    border-radius: 3px;
}}
QMenu {{
    background-color: {t["bg_secondary"]};
    color: {t["text_primary"]};
    border: 1px solid {t["border"]};
    padding: 4px;
}}
QMenu::item:selected {{
    background-color: {t["accent"]};
    color: {t["button_text"]};
    border-radius: 2px;
}}
QMenu::separator {{
    height: 1px;
    background: {t["border"]};
    margin: 4px 8px;
}}

/* ===== Sliders ===== */
QSlider::groove:horizontal {{
    background: {t["bg_tertiary"]};
    height: 6px;
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {t["accent"]};
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{
    background: {t["accent_hover"]};
}}
QSlider::sub-page:horizontal {{
    background: {t["accent"]};
    border-radius: 3px;
}}

/* ===== Spin Boxes ===== */
QSpinBox, QDoubleSpinBox {{
    background-color: {t["bg_tertiary"]};
    color: {t["text_primary"]};
    border: 1px solid {t["border"]};
    border-radius: 3px;
    padding: 3px 6px;
    min-height: 22px;
}}
QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {t["accent"]};
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background-color: {t["bg_secondary"]};
    border: none;
    width: 16px;
}}
QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: {t["accent"]};
}}

/* ===== Combo Boxes ===== */
QComboBox {{
    background-color: {t["bg_tertiary"]};
    color: {t["text_primary"]};
    border: 1px solid {t["border"]};
    border-radius: 3px;
    padding: 3px 8px;
    min-height: 22px;
}}
QComboBox:hover {{
    border-color: {t["border_light"]};
}}
QComboBox:focus {{
    border-color: {t["accent"]};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background-color: {t["bg_secondary"]};
    color: {t["text_primary"]};
    border: 1px solid {t["border"]};
    selection-background-color: {t["accent"]};
    selection-color: {t["button_text"]};
}}

/* ===== Check Boxes ===== */
QCheckBox {{
    spacing: 6px;
    color: {t["text_primary"]};
    background: transparent;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {t["border"]};
    border-radius: 3px;
    background: {t["bg_tertiary"]};
}}
QCheckBox::indicator:checked {{
    background: {t["accent"]};
    border-color: {t["accent"]};
}}
QCheckBox::indicator:hover {{
    border-color: {t["accent_hover"]};
}}

/* ===== List Widgets ===== */
QListWidget {{
    background-color: {t["bg_tertiary"]};
    color: {t["text_primary"]};
    border: 1px solid {t["border"]};
    border-radius: 3px;
    padding: 2px;
    outline: none;
}}
QListWidget::item {{
    padding: 4px 6px;
    border-radius: 2px;
}}
QListWidget::item:selected {{
    background-color: {t["accent"]};
    color: {t["button_text"]};
}}
QListWidget::item:hover {{
    background-color: {t["bg_secondary"]};
}}

/* ===== Control Tabs ===== */
QTabWidget#controlTabs::pane {{
    background-color: {t["bg_primary"]};
    border: 1px solid {t["border"]};
    border-radius: 8px;
    top: -1px;
}}
QTabWidget#controlTabs QTabBar::tab {{
    background-color: {t["bg_primary"]};
    color: {t["text_secondary"]};
    border: 1px solid {t["border"]};
    border-bottom: none;
    border-top-left-radius: 7px;
    border-top-right-radius: 7px;
    font-weight: 700;
    margin-right: 2px;
    padding: 7px 14px;
}}
QTabWidget#controlTabs QTabBar::tab:selected {{
    background-color: {t["bg_secondary"]};
    border-color: {t["accent"]};
    color: {t["text_primary"]};
}}
QTabWidget#controlTabs QTabBar::tab:hover:!selected {{
    color: {t["text_primary"]};
}}
QWidget#planningCockpit {{
    background-color: {t["bg_secondary"]};
    border: 2px solid {t["accent"]};
    border-radius: 10px;
}}

/* ===== Planning Cockpit ===== */
QLabel#selectedScrewTitle {{
    color: {t["text_primary"]};
    font-size: 17px;
    font-weight: 800;
    padding: 8px 4px 5px 4px;
}}
QLabel#screwReviewCounter {{
    background-color: {t["accent_dim"]};
    border: 1px solid {t["accent"]};
    border-radius: 5px;
    color: {t["secondary_action_text"]};
    font-size: 12px;
    font-weight: 800;
    padding: 6px 4px;
}}
QPushButton#screwReviewNav {{
    min-width: 78px;
    padding: 5px 8px;
    font-weight: 700;
}}
QWidget#screwMetrics {{
    background-color: {t["bg_tertiary"]};
    border: 1px solid {t["border"]};
    border-radius: 7px;
}}
QLabel#selectedScrewTitle[reviewActive="true"] {{
    color: {t["implant"]};
}}
QLabel#selectedScrewWarning {{
    background-color: {t["warning_bg"]};
    border: 1px solid {t["warning"]};
    border-radius: 6px;
    color: {t["warning_text"]};
    padding: 7px;
}}
QLabel#screwDragHint {{
    color: {t["text_secondary"]};
    background-color: {t["bg_tertiary"]};
    border-radius: 5px;
    padding: 6px 8px;
}}
MPRViewer, Viewer3D {{
    background-color: {t["viewer_bg"]};
    border: 1px solid {t["viewer_separator"]};
    border-radius: 7px;
}}
MPRViewer[reviewActive="true"] {{
    border: 2px solid {t["accent"]};
}}
QLabel#viewerHeader {{
    background-color: {t["viewer_header"]};
    color: {t["viewer_foreground"]};
    padding: 5px 8px;
    font-size: 12px;
}}
QLabel#viewerReadout {{
    background-color: {t["viewer_readout"]};
    color: #98A3AE;
    padding: 4px 8px;
    font-size: 11px;
}}

/* ===== Tables ===== */
QTableWidget {{
    background-color: {t["bg_tertiary"]};
    alternate-background-color: {t["table_alternate"]};
    color: {t["text_primary"]};
    gridline-color: {t["border"]};
    border: 1px solid {t["border"]};
    border-radius: 6px;
    selection-background-color: {t["accent_dim"]};
}}
QHeaderView::section {{
    background-color: {t["table_header"]};
    color: {t["text_secondary"]};
    border: none;
    border-right: 1px solid {t["border"]};
    border-bottom: 1px solid {t["border"]};
    padding: 6px 4px;
    font-weight: 700;
}}
QTableWidget#screwPlanTable {{
    background-color: {t["bg_tertiary"]};
    alternate-background-color: {t["table_alternate"]};
    border: 1px solid {t["border"]};
    border-radius: 7px;
    selection-background-color: {t["accent_dim"]};
    selection-color: {t["text_primary"]};
}}
QTableWidget#screwPlanTable::item {{
    padding: 5px 6px;
}}
QTableWidget#screwPlanTable::item:selected {{
    background-color: {t["accent_dim"]};
    color: {t["text_primary"]};
}}

/* ===== Status Bar ===== */
QStatusBar {{
    background-color: {t["bg_toolbar"]};
    color: {t["text_secondary"]};
    border-top: 1px solid {t["border"]};
    font-size: 12px;
    padding: 2px 8px;
}}

/* ===== Scroll Bars ===== */
QScrollBar:vertical {{
    background: {t["scrollbar_bg"]};
    width: 14px;
    margin: 0;
    border-radius: 5px;
}}
QScrollBar::handle:vertical {{
    background: {t["scrollbar_handle"]};
    min-height: 30px;
    border: 2px solid {t["scrollbar_bg"]};
    border-radius: 7px;
}}
QScrollBar::handle:vertical:hover {{
    background: {t["scrollbar_hover"]};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: {t["scrollbar_bg"]};
    height: 10px;
    margin: 0;
    border-radius: 5px;
}}
QScrollBar::handle:horizontal {{
    background: {t["scrollbar_handle"]};
    min-width: 30px;
    border-radius: 5px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {t["scrollbar_hover"]};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ===== Splitter Handle ===== */
QSplitter::handle:horizontal {{
    background: {t["bg_secondary"]};
    width: 3px;
}}
QSplitter::handle:horizontal:hover {{
    background: {t["accent"]};
}}

/* ===== Progress Dialog ===== */
QProgressDialog {{
    background-color: {t["bg_secondary"]};
    color: {t["text_primary"]};
}}
QProgressBar {{
    background-color: {t["bg_tertiary"]};
    border: 1px solid {t["border"]};
    border-radius: 4px;
    text-align: center;
    color: {t["text_primary"]};
    height: 18px;
}}
QProgressBar::chunk {{
    background-color: {t["accent"]};
    border-radius: 3px;
}}

/* ===== Input Dialog / Message Box ===== */
QInputDialog, QMessageBox {{
    background-color: {t["bg_secondary"]};
    color: {t["text_primary"]};
}}

/* ===== Tooltips ===== */
QToolTip {{
    background-color: {t["bg_secondary"]};
    color: {t["text_primary"]};
    border: 1px solid {t["border"]};
    border-radius: 3px;
    padding: 4px 8px;
    font-size: 12px;
}}

/* ===== File Dialog ===== */
QFileDialog {{
    background-color: {t["bg_secondary"]};
    color: {t["text_primary"]};
}}
"""
