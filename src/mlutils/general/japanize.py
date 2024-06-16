from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


def japanize_matplotlib(japanese_font_family: list[str] = ['Bizin Gothic Discord', 'Noto Sans CJK JP'], japanese_ttf_dir: Path = Path("/usr/share/fonts/truetype")):
    # 日本語フォントの設定
    # https://qiita.com/mikan3rd/items/791e3cd7f75e010c8f9f
    font_files = fm.findSystemFonts(str(japanese_ttf_dir))

    for font_file in font_files:
        try:
            fm.fontManager.addfont(font_file)
        except RuntimeError:
            pass
    plt.rcParams['font.family'] = japanese_font_family
    plt.rcParams['font.sans-serif'] = 'sans-serif'



if __name__ == "__main__":
    japanize_matplotlib()
    plt.plot([1, 2, 3], [4, 5, 6])
    plt.title("日本語タイトル")
    plt.savefig("image.png")