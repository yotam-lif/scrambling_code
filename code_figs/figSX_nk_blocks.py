"""Export the editable Inkscape schematic for supplementary Figure SX.

Edit ext_svg/figSX_nk_blocks.svg in Inkscape, then run this script to
refresh the vector PDF in figs_paper. Use --png for a preview alongside it.
The SVG is the source of truth; this script does not overwrite manual edits.
"""

import argparse
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", action="store_true", help="also export a 300-dpi PNG")
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    source = script_dir / "ext_svg" / "figSX_nk_blocks.svg"
    out_dir = script_dir.parent / "figs_paper"
    inkscape = shutil.which("inkscape")
    if inkscape is None:
        mac_app = Path("/Applications/Inkscape.app/Contents/MacOS/inkscape")
        if mac_app.is_file():
            inkscape = str(mac_app)
        else:
            parser.error("Inkscape is required; install it or add it to PATH.")
    out_dir.mkdir(exist_ok=True)
    for fmt in (["pdf", "png"] if args.png else ["pdf"]):
        target = out_dir / f"{source.stem}.{fmt}"
        subprocess.run(
            [inkscape, str(source), f"--export-type={fmt}",
             f"--export-filename={target}", "--export-area-page",
             "--export-background=white", "--export-background-opacity=1",
             "--export-dpi=300", "--export-text-to-path"],
            check=True,
        )
        print(f"Saved: {target}")


if __name__ == "__main__":
    main()
