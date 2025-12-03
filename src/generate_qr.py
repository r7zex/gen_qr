"""QR code generator with customizable module and finder pattern assets."""
from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
from collections.abc import Sequence
from typing import Callable, Protocol, Tuple, cast

from PIL import Image, UnidentifiedImageError
from qrcode.exceptions import DataOverflowError

FinderPosition = Tuple[int, int]


def finder_pattern_origins(matrix_size: int, border: int) -> Tuple[FinderPosition, ...]:
    """Return the top-left indices for the three finder patterns."""

    offset = matrix_size - border - 7
    return (
        (border, border),
        (offset, border),
        (border, offset),
    )


DEFAULT_OUTPUT_DIR = Path("output")


class AssetBundle:
    """Container for module and finder pattern assets."""

    def __init__(self, module: Image.Image, finder_inner: Image.Image, finder_outer: Image.Image) -> None:
        self.module = module.convert("RGBA")
        self.finder_inner = finder_inner.convert("RGBA")
        self.finder_outer = finder_outer.convert("RGBA")
        self.module_size = self.module.width
        if self.module.height != self.module_size:
            raise ValueError("Module asset must be square.")

    def resized(self, scale: float) -> "AssetBundle":
        """Return a new bundle with all assets scaled by *scale*.

        The resulting assets preserve their aspect ratio. Sizes are rounded to
        the nearest integer and clamped to a minimum of 1px to avoid zero-sized
        images when heavily downscaling.
        """

        if scale == 1.0:
            return self

        def _resize(image: Image.Image) -> Image.Image:
            width = max(1, int(round(image.width * scale)))
            height = max(1, int(round(image.height * scale)))
            return image.resize((width, height), RESAMPLE_FILTER)

        return AssetBundle(
            _resize(self.module),
            _resize(self.finder_inner),
            _resize(self.finder_outer),
        )

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


class QRCodeLike(Protocol):
    """Protocol describing the subset of QRCode API that we rely on."""

    def add_data(self, data: str) -> None:
        """Add data payload to the QR code."""

    def make(self, fit: bool = ...) -> None:
        """Finalize the QR code matrix generation."""

    def get_matrix(self) -> list[list[bool]]:
        """Return the generated matrix."""


QRCodeFactory = Callable[..., QRCodeLike]


def _load_qrcode_backend() -> Tuple[QRCodeFactory, int]:
    """Locate a QRCode implementation from qrcode or django_qrcode."""

    try:
        from qrcode.main import QRCode as qrcode_cls  # type: ignore
        from qrcode.constants import ERROR_CORRECT_H  # type: ignore
    except ModuleNotFoundError:
        qrcode_cls = None
        error_constant = None
    else:
        return cast(QRCodeFactory, qrcode_cls), int(ERROR_CORRECT_H)

    module_candidates = (
        "django_qrcode.qrcode",
        "django_qrcode.qrcode.qrcode",
        "django_qrcode",
    )

    for module_name in module_candidates:
        try:
            module = import_module(module_name)
        except ModuleNotFoundError:
            continue

        qr_cls = getattr(module, "QRCode", None)
        constants = getattr(module, "constants", None)

        if qr_cls is None and hasattr(module, "qrcode"):
            nested_module = getattr(module, "qrcode")
            qr_cls = getattr(nested_module, "QRCode", None)
            constants = getattr(nested_module, "constants", constants)
            module = nested_module

        if qr_cls is None:
            try:
                nested_module = import_module(f"{module.__name__}.qrcode")
            except ModuleNotFoundError:
                nested_module = None
            if nested_module is not None:
                qr_cls = getattr(nested_module, "QRCode", None)
                constants = getattr(nested_module, "constants", constants)
                module = nested_module

        if qr_cls is None:
            continue

        if constants is None:
            try:
                constants = import_module(f"{module.__name__}.constants")
            except ModuleNotFoundError:
                constants = None

        error_constant = int(getattr(constants, "ERROR_CORRECT_H", 2))
        return cast(QRCodeFactory, qr_cls), error_constant

    raise ModuleNotFoundError(
        "Unable to locate a QRCode backend. Install either 'qrcode' or 'django_qrcode'."
    )


QRCode, ERROR_CORRECT_H = _load_qrcode_backend()


def modules_to_version(modules: int) -> int:
    """Convert a requested module count to the corresponding QR version.

    Standard QR codes are square with sizes: 21×21 for version 1 and
    increasing by 4 modules per side for each subsequent version. This helper
    validates the requested module count and returns the necessary version
    number for the QRCode backend.
    """

    if modules < 21:
        raise ValueError("Минимальный размер стандартного QR-кода — 21×21 модуля (версия 1)")
    if (modules - 21) % 4 != 0:
        raise ValueError(
            "Размер QR-кода должен быть нечётным числом из прогрессии 21, 25, 29, … (по стандарту QR)"
        )

    return (modules - 21) // 4 + 1


def version_to_modules(version: int) -> int:
    """Return the module count for a QR version."""

    if version < 1:
        raise ValueError("QR version must be 1 or greater")
    return 21 + 4 * (version - 1)


def create_qr_matrix(data: str, *, version: int | None = None) -> Tuple[list[list[bool]], int]:
    qr = QRCode(error_correction=ERROR_CORRECT_H, box_size=1, border=4, version=version)
    qr.add_data(data)

    try:
        qr.make(fit=version is None)
    except DataOverflowError:
        auto_qr = QRCode(error_correction=ERROR_CORRECT_H, box_size=1, border=4)
        auto_qr.add_data(data)
        auto_qr.make(fit=True)
        qr = auto_qr

    matrix = qr.get_matrix()
    return matrix, int(qr.border)


def is_finder_module(row: int, col: int, size: int, border: int) -> bool:
    for origin_col, origin_row in finder_pattern_origins(size, border):
        if origin_row <= row < origin_row + 7 and origin_col <= col < origin_col + 7:
            return True
    return False


def paste_modules(
    canvas: Image.Image,
    matrix: Sequence[Sequence[bool]],
    assets: AssetBundle,
    margin: int,
    border: int,
) -> None:
    module_img = assets.module
    module_size = assets.module_size
    sized_module = module_img.resize((module_size, module_size), RESAMPLE_FILTER)
    for row_index, row in enumerate(matrix):
        for col_index, cell in enumerate(row):
            if not cell or is_finder_module(row_index, col_index, len(matrix), border):
                continue
            canvas.alpha_composite(sized_module, (margin + col_index * module_size, margin + row_index * module_size))


def paste_finder_patterns(
    canvas: Image.Image,
    matrix_size: int,
    assets: AssetBundle,
    margin: int,
    border: int,
) -> None:
    module_size = assets.module_size
    finder_span = 7 * module_size

    outer = assets.finder_outer
    inner = assets.finder_inner

    outer_offset = (finder_span - outer.width) // 2
    inner_offset = (finder_span - inner.width) // 2

    for origin_col, origin_row in finder_pattern_origins(matrix_size, border):
        top_left_x = margin + origin_col * module_size
        top_left_y = margin + origin_row * module_size
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
    output_size: int | None,
    matrix_modules: int | None,
) -> Path:
    assets = AssetBundle.from_paths(module_asset, finder_inner_asset, finder_outer_asset)
    matrix_version = modules_to_version(matrix_modules) if matrix_modules is not None else None
    matrix, border = create_qr_matrix(data, version=matrix_version)

    if matrix_modules is not None and matrix_modules != len(matrix):
        scale = matrix_modules / len(matrix)
        assets = assets.resized(scale)

    canvas, margin = build_canvas(len(matrix), assets, background_color)
    paste_modules(canvas, matrix, assets, margin, border)
    paste_finder_patterns(canvas, len(matrix), assets, margin, border)

    if output_size is not None:
        if output_size <= 0:
            raise ValueError("Output size must be a positive integer")
        canvas = canvas.resize((output_size, output_size), RESAMPLE_FILTER)

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
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR / "qr.png",
        type=Path,
        help="Where to write the generated PNG",
    )
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
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Final square image size in pixels (e.g., 16 for 16x16, 32 for 32x32)",
    )
    parser.add_argument(
        "--modules",
        type=int,
        default=None,
        help=(
            "Размер матрицы QR-кода в модулях (например, 21, 25, 29). "
            "Поддерживаются только стандартные размеры QR (21 + 4*n)."
        ),
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
        parsed.size,
        parsed.modules,
    )
    print(f"QR code saved to {output_path}")


if __name__ == "__main__":
    main()
