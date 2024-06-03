import matplotlib.pyplot as plt


def japanize_matplotlib():
    # 日本語フォントの設定
    import matplotlib.font_manager as fm
    plt.rcParams['font.family'] = [f.name for f in fm.fontManager.ttflist]
    plt.rcParams['font.sans-serif'] = 'sans-serif'

