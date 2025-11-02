"""QR code generator with customizable module and finder pattern assets."""
from __future__ import annotations

import argparse
from pathlib import Path
from collections.abc import Sequence
from typing import Tuple

from qrcode.main import QRCode
from qrcode.constants import ERROR_CORRECT_H

from PIL import Image, UnidentifiedImageError

FinderPosition = Tuple[int, int]


FINDER_POSITIONS: Tuple[FinderPosition, ...] = ((0, 0), (1, 0), (0, 1))
"""Finder pattern positions expressed as (column_multiplier, row_multiplier).

The QR matrix origin is the top-left module. Each finder pattern occupies a
7x7 region. The positions above represent top-left, top-right, and
bottom-left patterns respectively.
"""


class AssetBundle:
    """Container for module and finder pattern assets."""

    def __init__(self, module: Image.Image, finder_inner: Image.Image, finder_outer: Image.Image) -> None:
        self.module = module.convert("RGBA")
        self.finder_inner = finder_inner.convert("RGBA")
        self.finder_outer = finder_outer.convert("RGBA")
        self.module_size = self.module.width
        if self.module.height != self.module_size:
            raise ValueError("Module asset must be square.")

    @classmethod
    def from_paths(cls, module_path: Path, finder_inner_path: Path, finder_outer_path: Path) -> "AssetBundle":
        for path in (module_path, finder_inner_path, finder_outer_path):
            if path.suffix.lower() != ".png":
                raise ValueError(
                    f"Asset '{path}' must be a PNG file. Replace the placeholder text file with your design."
                )
        try:
            module = Image.open(module_path)
            finder_inner = Image.open(finder_inner_path)
            finder_outer = Image.open(finder_outer_path)
        except UnidentifiedImageError as exc:
            raise ValueError(
                "One or more assets are not valid images. Replace the placeholder text files with PNG artwork."
            ) from exc
        return cls(module, finder_inner, finder_outer)


try:
    RESAMPLE_FILTER = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9.1
    RESAMPLE_FILTER = Image.LANCZOS  # type: ignore[attr-defined]


def create_qr_matrix(data: str) -> list[list[bool]]:
    qr = QRCode(error_correction=ERROR_CORRECT_H, box_size=1, border=4)
    qr.add_data(data)
    qr.make(fit=True)
    return qr.get_matrix()


def is_finder_module(row: int, col: int, size: int) -> bool:
    finder_bounds = ((0, 0), (size - 7, 0), (0, size - 7))
    for origin_col, origin_row in finder_bounds:
        if origin_row <= row < origin_row + 7 and origin_col <= col < origin_col + 7:
            return True
    return False


def paste_modules(canvas: Image.Image, matrix: Sequence[Sequence[bool]], assets: AssetBundle, margin: int) -> None:
    module_img = assets.module
    module_size = assets.module_size
    sized_module = module_img.resize((module_size, module_size), RESAMPLE_FILTER)
    for row_index, row in enumerate(matrix):
        for col_index, cell in enumerate(row):
            if not cell or is_finder_module(row_index, col_index, len(matrix)):
                continue
            canvas.alpha_composite(sized_module, (margin + col_index * module_size, margin + row_index * module_size))


def paste_finder_patterns(canvas: Image.Image, matrix_size: int, assets: AssetBundle, margin: int) -> None:
    module_size = assets.module_size
    finder_span = 7 * module_size

    outer = assets.finder_outer
    inner = assets.finder_inner

    outer_offset = (finder_span - outer.width) // 2
    inner_offset = (finder_span - inner.width) // 2

    for column_multiplier, row_multiplier in FINDER_POSITIONS:
        top_left_x = margin + column_multiplier * (matrix_size - 7) * module_size
        top_left_y = margin + row_multiplier * (matrix_size - 7) * module_size
        if outer_offset >= 0:
            outer_image = outer.resize((finder_span, finder_span), RESAMPLE_FILTER)
            outer_position = (top_left_x, top_left_y)
        else:
            outer_image = outer
            outer_position = (top_left_x + outer_offset, top_left_y + outer_offset)
        canvas.alpha_composite(outer_image, outer_position)

        if inner_offset >= 0:
            inner_image = inner.resize(
                (finder_span - 2 * inner_offset, finder_span - 2 * inner_offset),
                RESAMPLE_FILTER,
            )
            inner_position = (
                top_left_x + inner_offset,
                top_left_y + inner_offset,
            )
        else:
            inner_image = inner
            inner_position = (top_left_x + inner_offset, top_left_y + inner_offset)
        canvas.alpha_composite(inner_image, inner_position)


def build_canvas(matrix_size: int, assets: AssetBundle, background: Tuple[int, int, int, int]) -> Tuple[Image.Image, int]:
    module_size = assets.module_size
    finder_span = 7 * module_size
    outer = assets.finder_outer
    margin = max(0, (outer.width - finder_span) // 2)
    canvas_size = matrix_size * module_size + margin * 2
    canvas = Image.new("RGBA", (canvas_size, canvas_size), background)
    return canvas, margin


def generate_qr(
    data: str,
    output: Path,
    module_asset: Path,
    finder_inner_asset: Path,
    finder_outer_asset: Path,
    background_color: Tuple[int, int, int, int],
) -> Path:
    assets = AssetBundle.from_paths(module_asset, finder_inner_asset, finder_outer_asset)
    matrix = create_qr_matrix(data)
    canvas, margin = build_canvas(len(matrix), assets, background_color)
    paste_modules(canvas, matrix, assets, margin)
    paste_finder_patterns(canvas, len(matrix), assets, margin)

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def parse_color(value: str) -> Tuple[int, int, int, int]:
    if value.startswith("#"):
        value = value[1:]
    if len(value) not in {6, 8}:
        raise argparse.ArgumentTypeError("Color must be in #RRGGBB or #RRGGBBAA format")
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    a = 255
    if len(value) == 8:
        a = int(value[6:8], 16)
    return r, g, b, a


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a QR code with custom module assets")
    parser.add_argument("url", help="URL to encode into the QR code")
    parser.add_argument("--output", default="output/qr.png", type=Path, help="Where to write the generated PNG")
    parser.add_argument(
        "--module",
        default=Path("assets/modules/default.txt"),
        type=Path,
        help="Module asset (40x40 PNG). Replace the placeholder .txt with your PNG file.",
    )
    parser.add_argument(
        "--finder-inner",
        dest="finder_inner",
        default=Path("assets/finder_inner/default.txt"),
        type=Path,
        help="Finder inner asset (120x120 PNG). Replace the placeholder .txt with your PNG file.",
    )
    parser.add_argument(
        "--finder-outer",
        dest="finder_outer",
        default=Path("assets/finder_outer/default.txt"),
        type=Path,
        help="Finder outer asset (300x300 PNG). Replace the placeholder .txt with your PNG file.",
    )
    parser.add_argument(
        "--background",
        type=parse_color,
        default="#FFFFFFFF",
        help="Background color in hexadecimal (#RRGGBB or #RRGGBBAA)",
    )
    return parser


def main(args: list[str] | None = None) -> None:
    parser = build_argument_parser()
    parsed = parser.parse_args(args=args)
    output_path = generate_qr(
        parsed.url,
        parsed.output,
        parsed.module,
        parsed.finder_inner,
        parsed.finder_outer,
        parsed.background,
    )
    print(f"QR code saved to {output_path}")


if __name__ == "__main__":
    main()
