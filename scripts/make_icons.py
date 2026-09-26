"""アプリのアイコン（SVG と PNG）を作る開発用スクリプト。

できたファイルは src/stemapp/web/icons/ にコミットしてある。デザインを変えたときだけ実行する。
    C:\\mine\\stem\\.venv\\Scripts\\python.exe scripts\\make_icons.py

図柄: 暗い背景に、波形（stem のレベル）に見える縦の棒5本。差し色は赤系の2色。
「赤い丸」は録音ボタンに見えるので使わない。PNG 作成には Pillow（venv に入っているもの）を使う。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "src" / "stemapp" / "web" / "icons"
BG = "#0b0b0e"
ACCENT = "#ff3b4e"
ACCENT_DEEP = "#9e1b2c"
UNIT = 512
BAR_W = 44
GAP = 26
HEIGHTS = (150, 250, 330, 210, 130)
COLORS = (ACCENT_DEEP, ACCENT, ACCENT, ACCENT, ACCENT_DEEP)


def bars() -> list[tuple[float, float, float, float, str]]:
    total = len(HEIGHTS) * BAR_W + (len(HEIGHTS) - 1) * GAP
    x = (UNIT - total) / 2
    out = []
    for h, color in zip(HEIGHTS, COLORS, strict=True):
        y = (UNIT - h) / 2
        out.append((x, y, BAR_W, h, color))
        x += BAR_W + GAP
    return out


def svg(rounded: bool) -> str:
    rx = 112 if rounded else 0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {UNIT} {UNIT}">',
        f'<rect width="{UNIT}" height="{UNIT}" rx="{rx}" fill="{BG}"/>',
    ]
    for x, y, w, h, color in bars():
        parts.append(
            f'<rect x="{x:g}" y="{y:g}" width="{w}" height="{h}" rx="{w / 2:g}" fill="{color}"/>'
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def png(size: int, path: Path) -> None:
    scale = 4
    big = size * scale
    k = big / UNIT
    img = Image.new("RGB", (big, big), BG)  # 全面を塗る（iPhone が角を丸める）
    draw = ImageDraw.Draw(img)
    for x, y, w, h, color in bars():
        draw.rounded_rectangle(
            (x * k, y * k, (x + w) * k, (y + h) * k), radius=w * k / 2, fill=color
        )
    img.resize((size, size), Image.Resampling.LANCZOS).save(path, optimize=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "icon.svg").write_text(svg(rounded=True), encoding="utf-8", newline="\n")
    png(180, OUT / "apple-touch-icon.png")
    png(192, OUT / "icon-192.png")
    png(512, OUT / "icon-512.png")
    print(f"{OUT} に書き出しました。")


if __name__ == "__main__":
    main()
