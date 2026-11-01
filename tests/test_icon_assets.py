"""Brand asset exports retain transparency and every platform icon size."""

import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ICONS = ROOT / "packaging" / "icons"
PNG_SIZES = (16, 24, 32, 48, 64, 128, 256, 512, 1024)
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)


def test_master_and_wordmarks_are_standalone_vectors():
    master = ElementTree.parse(ICONS / "radwright-screw.svg").getroot()
    assert master.attrib["viewBox"] == "0 0 1024 1024"
    assert not list(master.iter("{http://www.w3.org/2000/svg}text"))
    for suffix in ("", "-light"):
        wordmark = ElementTree.parse(ICONS / f"radwright-screw-wordmark{suffix}.svg")
        text = "".join(wordmark.getroot().itertext())
        assert "RadWright" in text and "Screw" in text
        for element in wordmark.getroot().iter():
            assert not element.tag.endswith("image")
            assert not element.tag.endswith("use")


def assert_exports(directory):
    for size in PNG_SIZES:
        with Image.open(directory / "png" / f"icon-{size}.png") as image:
            assert image.size == (size, size)
            assert image.mode == "RGBA"
            assert image.getpixel((0, 0))[3] == 0
            assert image.getpixel((size // 2, size // 2))[3] == 255
    with Image.open(directory / "radwright-screw.ico") as image:
        assert image.ico.sizes() == {(size, size) for size in ICO_SIZES}
        for size in ICO_SIZES:
            frame = image.ico.getimage((size, size))
            with Image.open(directory / "png" / f"icon-{size}.png") as png:
                assert frame.tobytes() == png.tobytes()
    with Image.open(directory / "radwright-screw.icns") as image:
        sizes = {width * scale for width, height, scale in image.info["sizes"]}
        assert {16, 32, 64, 128, 256, 512, 1024} <= sizes
        for frame_size in image.info["sizes"]:
            size = frame_size[0] * frame_size[2]
            frame = image.icns.getimage(frame_size)
            frame.load()
            assert frame.size == (size, size)
            with Image.open(directory / "png" / f"icon-{size}.png") as png:
                assert frame.tobytes() == png.tobytes()
    with Image.open(directory / "preview.png") as image:
        image.verify()


def test_checked_in_platform_exports():
    assert_exports(ICONS)


def test_render_cli_rebuilds_assets_outside_repository(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "render_icons.py"),
         "--output-dir", str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert_exports(tmp_path)


def test_small_exports_select_simplified_artwork(monkeypatch, tmp_path, qapp):
    import scripts.render_icons as renderer

    # Contrasting synthetic geometry exposes which SVG variant each export uses.
    source = tmp_path / "source"
    source.mkdir()
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024">
      <defs><g id="small-icon">
        <rect width="1024" height="1024" rx="225" fill="#0B1220"/>
        <circle cx="512" cy="512" r="256" fill="#22D3EE"/>
      </g></defs>
      <g id="standard-icon">
        <rect width="1024" height="1024" rx="225" fill="#0B1220"/>
        <circle cx="512" cy="512" r="256" fill="#60A5FA"/>
      </g>
    </svg>"""
    for name in ("radwright-screw.svg", "radwright-screw-wordmark.svg",
                 "radwright-screw-wordmark-light.svg"):
        (source / name).write_text(svg, encoding="utf-8")
    monkeypatch.setattr(renderer, "SOURCE_DIR", source)
    output = tmp_path / "exports"
    renderer.render_icons(output)

    for size in PNG_SIZES:
        expected = (34, 211, 238, 255) if size in (16, 24, 32) else (96, 165, 250, 255)
        with Image.open(output / "png" / f"icon-{size}.png") as png:
            assert png.getpixel((size // 2, size // 2)) == expected
    # The small artwork must survive both platform containers too.
    with Image.open(output / "radwright-screw.ico") as ico:
        for size in (16, 24, 32):
            frame = ico.ico.getimage((size, size))
            assert frame.getpixel((size // 2, size // 2)) == (34, 211, 238, 255)
    with Image.open(output / "radwright-screw.icns") as icns:
        for frame_size in icns.info["sizes"]:
            size = frame_size[0] * frame_size[2]
            if size in (16, 32):
                frame = icns.icns.getimage(frame_size)
                assert frame.getpixel((size // 2, size // 2)) == (34, 211, 238, 255)
