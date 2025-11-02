"""Convenience entry point for running the QR code generator without CLI flags.

Update the configuration variables below to control what data is encoded, where
the resulting file is written, and which assets are used to render the QR
code. When you run ``main.py`` (for example from PyCharm) the script will use
these values and immediately generate the QR code image.
"""

from __future__ import annotations

from pathlib import Path

from src.generate_qr import DEFAULT_OUTPUT_DIR, generate_qr, parse_color

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Text or URL that should be encoded inside the QR code.
DATA_TO_ENCODE = "https://example.com"

# Directory where the generated QR code will be saved.
DOWNLOAD_DIRECTORY = DEFAULT_OUTPUT_DIR

# File name for the generated QR image inside DOWNLOAD_DIRECTORY.
OUTPUT_FILENAME = "qr.png"

# Paths to the artwork assets used when rendering modules and finder patterns.
# Replace the placeholder .txt files with your PNG artwork and update the file
# names below if necessary.
MODULE_ASSET_PATH = Path("assets/modules/default.txt")
FINDER_INNER_ASSET_PATH = Path("assets/finder_inner/default.txt")
FINDER_OUTER_ASSET_PATH = Path("assets/finder_outer/default.txt")

# Background color for the QR code image. Accepts #RRGGBB or #RRGGBBAA.
BACKGROUND_COLOR = "#FFFFFFFF"


def main() -> None:
    """Generate a QR code using the configuration specified above."""

    output_path = Path(DOWNLOAD_DIRECTORY) / OUTPUT_FILENAME
    background = parse_color(BACKGROUND_COLOR)

    saved_path = generate_qr(
        DATA_TO_ENCODE,
        output_path,
        Path(MODULE_ASSET_PATH),
        Path(FINDER_INNER_ASSET_PATH),
        Path(FINDER_OUTER_ASSET_PATH),
        background,
    )

    print(f"QR code saved to {saved_path}")


if __name__ == "__main__":
    main()
