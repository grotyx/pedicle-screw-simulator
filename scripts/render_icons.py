#!/usr/bin/env python3
"""Rebuild RadWright Screw PNG, ICO, ICNS and preview assets from the SVGs.

Uses the application's existing PyQt6 SVG renderer and build-time Pillow.
No clinical images or application data are used.
"""

import argparse
import io
import os
import struct
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from PyQt6.QtCore import QBuffer, QByteArray, QIODevice, QRectF, Qt
from PyQt6.QtGui import QGuiApplication, QImage, QPainter
from PyQt6.QtSvg import QSvgRenderer

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "packaging" / "icons"
PNG_SIZES = (16, 24, 32, 48, 64, 128, 256, 512, 1024)
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)


def render_svg(source, width, height, element_id=None):
    """Supersample vector geometry, preserving transparent rounded corners."""
    renderer = QSvgRenderer(QByteArray(source))
    if not renderer.isValid():
        raise ValueError("Invalid SVG source")
    image = QImage(width * 4, height * 4, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    if element_id is None:
        renderer.render(painter)
    else:
        if not renderer.elementExists(element_id):
            painter.end()
            raise ValueError(f"Missing SVG variant: {element_id}")
        renderer.render(painter, element_id, QRectF(0, 0, image.width(), image.height()))
    painter.end()
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    with Image.open(io.BytesIO(bytes(buffer.data()))) as raster:
        result = raster.convert("RGBA").resize((width, height), Image.Resampling.LANCZOS)
    # Remove faint Lanczos ringing outside flat shapes, keeping edge antialiasing.
    result.putalpha(result.getchannel("A").point(lambda alpha: 0 if alpha < 8 else alpha))
    return result


def write_icns(exports, path):
    """Write standard PNG-backed ICNS chunks, including native 16 px artwork.

    Pillow's ICNS writer omits the native 16 px frame. Use its PNG encoder
    for each frame and the ICNS container's eight-byte big-endian headers.
    """
    frames = (
        (b"icp4", 16), (b"icp5", 32), (b"icp6", 64),
        (b"ic07", 128), (b"ic08", 256), (b"ic09", 512), (b"ic10", 1024),
        (b"ic11", 32), (b"ic12", 64), (b"ic13", 256), (b"ic14", 512),
    )
    chunks = []
    for kind, size in frames:
        buffer = io.BytesIO()
        exports[size].save(buffer, "PNG")
        data = buffer.getvalue()
        chunks.append(struct.pack(">4sI", kind, len(data) + 8) + data)
    payload = b"".join(chunks)
    path.write_bytes(struct.pack(">4sI", b"icns", len(payload) + 8) + payload)


def preview_font(size):
    """Resolve a sans font for contact-sheet labels."""
    # Pillow's bundled fallback works when DejaVu Sans is not installed.
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def render_preview(exports, output_dir):
    """Show actual pixel sizes on both backgrounds, plus enlarged small icons."""
    sheet = Image.new("RGB", (2400, 1912))
    draw = ImageDraw.Draw(sheet)
    for column, (background, foreground, muted, suffix) in enumerate((
        ("#151F32", "#F1F5F9", "#94A3B8", ""),
        ("#FFFFFF", "#0B1220", "#475569", "-light"),
    )):
        left = column * 1200
        draw.rectangle((left, 0, left + 1199, 1911), fill=background)
        draw.text((left + 80, 48), "RadWright Screw", fill=foreground, font=preview_font(38))
        draw.text((left + 80, 108), "1024 px · master", fill=muted, font=preview_font(22))
        sheet.paste(exports[1024], (left + 88, 160), exports[1024])

        for size, x in ((256, 88), (64, 408), (32, 584), (16, 744), (24, 904)):
            draw.text((left + x, 1224), f"{size} px", fill=muted, font=preview_font(22))
            sheet.paste(exports[size], (left + x, 1272), exports[size])
        draw.text((left + 408, 1400), "32 px / 4×", fill=muted, font=preview_font(20))
        draw.text((left + 584, 1400), "24 px / 4×", fill=muted, font=preview_font(20))
        draw.text((left + 744, 1400), "16 px / 8×", fill=muted, font=preview_font(20))
        for size, x in ((32, 408), (16, 744)):
            enlarged = exports[size].resize((128, 128), Image.Resampling.NEAREST)
            sheet.paste(enlarged, (left + x, 1440), enlarged)
        enlarged = exports[24].resize((96, 96), Image.Resampling.NEAREST)
        sheet.paste(enlarged, (left + 584, 1440), enlarged)

        wordmark = render_svg(
            (SOURCE_DIR / f"radwright-screw-wordmark{suffix}.svg").read_bytes(), 1040, 189,
        )
        draw.text((left + 80, 1660), "WORDMARK", fill=muted, font=preview_font(20))
        sheet.paste(wordmark, (left + 80, 1696), wordmark)
    sheet.save(output_dir / "preview.png", optimize=True)


def render_icons(output_dir):
    """Export each size from vector geometry, including every ICO frame."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "png").mkdir(exist_ok=True)
    source = (SOURCE_DIR / "radwright-screw.svg").read_bytes()
    exports = {
        size: render_svg(source, size, size, "small-icon" if size in (16, 24, 32) else None)
        for size in PNG_SIZES
    }
    for size, image in exports.items():
        image.save(output_dir / "png" / f"icon-{size}.png", optimize=True)
    exports[256].save(
        output_dir / "radwright-screw.ico",
        sizes=[(size, size) for size in ICO_SIZES],
        append_images=[exports[size] for size in ICO_SIZES if size != 256],
    )
    write_icns(exports, output_dir / "radwright-screw.icns")
    render_preview(exports, output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=SOURCE_DIR,
        help="Export directory (default: packaging/icons beside the source SVGs)",
    )
    args = parser.parse_args()
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    application = QGuiApplication.instance() or QGuiApplication([])
    render_icons(args.output_dir.resolve())
    print(f"Wrote {len(PNG_SIZES)} PNGs, ICO, ICNS and preview to {args.output_dir.resolve()}")
    # Keep the application alive until SVG text and image painting have finished.
    del application


if __name__ == "__main__":
    main()
