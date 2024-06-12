# -*- coding: utf-8 -*-
"""披露宴席次最適化_GW.ipynb
# 披露宴席次最適化
[披露宴の席次をGromov-Wasserstein最適輸送で最適化した話](https://zenn.dev/akira_t/articles/seat-opt-via-gw)の再現用コードです
"""
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np

from mlutils.general.japanize import japanize_matplotlib  # noqa

"""## GWで解く"""


# 正方行列と X および Y のラベルの行列を渡す
# https://qiita.com/ynakayama/items/7dc01f45caf6d87a981b より
def heat_mat(data, rowlabels=None, collabels=None):
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap=plt.cm.Blues)

    ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)

    ax.invert_yaxis()
    ax.xaxis.tick_top()

    if not rowlabels:
        rowlabels = list(range(data.shape[0]))
    if not collabels:
        collabels = list(range(data.shape[1]))
    ax.set_xticklabels(collabels, minor=False, rotation=30)
    ax.set_yticklabels(rowlabels, minor=False)
    plt.show()

    return heatmap


def fillzero(s):
    return (
        s.replace("\n\t", "\n0\t")
        .replace("\t\n", "\t0\n")
        .replace("\t\t", "\t0\t")
        .replace("\t\t", "\t0\t")
    )


guest_name = """新郎1
新郎2
幼馴染
父
母
弟
友人（オケ）1
友人（オケ）2
友人（オケ）3
級友（高校）1
級友（高校）2
友人（ロボ）１
友人（ロボ）2
先輩
友人（ロボ）3
同僚１
友人（大学）
同僚3
同僚4
先生""".split(
    "\n"
)

N = len(guest_name)

# Group affiliation
# 参加者のグループへの所属(度)
# スプレッドシートなどで設計して貼り付ける
Group = np.loadtxt(
    StringIO(
        fillzero(
            """
	2				0.5	1	1
		1	1	1
2
2
2
2
			1
			1								1
			1							1
		1	1	1								1
		1		1
				1
				1						1
	2							1	1
				1				1
					1	1			1
						1					1	1
					1
					1
							1
"""
        )
    ),
    delimiter="\t",
)

heat_mat(Group, guest_name)

G2 = np.sqrt(Group @ (Group).transpose())
heat_mat(G2, guest_name)

# 席間の近さの上半分
D_upper = np.loadtxt(
    StringIO(
        fillzero(
            """
	2	1			1	1
		2	1		1	1	1
			2	1		1	1	1
				2			1	1	1
								1	1
						2	1
							2	1
								2	1
									2

											2	1			1	1
												2	1		1	1	1
													2	1		1	1	1
														2			1	1	1
																		1	1
																2	1
																	2	1
																		2	1
																			2

"""
        )
    ),
    delimiter="\t",
)

D = D_upper + D_upper.transpose()
heat_mat(D)

# similarity -> normalized distance
dist_G2 = (G2.max() - G2) / (G2.max() - G2.min())
np.fill_diagonal(dist_G2, 0)

heat_mat(dist_G2, rowlabels=guest_name)

plt.hist(dist_G2.reshape((-1)))

# dist_D = 1/(D+1.)
dist_D = (D.max() - D) / (D.max() - D.min())
np.fill_diagonal(dist_D, 0)
heat_mat(dist_D)

plt.hist(dist_D.reshape((-1)))

G0 = np.zeros([N, N])
G0[0, 0] = 1.0
G0[1, 12:] = 1.0 / 8
G0[2:, 1:12] = 1.0 / (N - 2)
G0[2:, 12:] = 7 / 8 / (N - 2)
G0 /= N
G0.sum(axis=0)  # check [1/N ,… 1/N]

heat_mat(G0, guest_name)

import ot

p = ot.unif(N)
q = ot.unif(N)
gw, log = ot.gromov.gromov_wasserstein(
    dist_G2, dist_D, p, q, "square_loss", log=True, verbose=True, G0=G0
)
heat_mat(gw, guest_name)

gw, log = ot.gromov.gromov_wasserstein(
    dist_G2, dist_D, p, q, "kl_loss", log=True, verbose=True, G0=G0
)
heat_mat(gw, guest_name)


def harden_assignment(gw):
    T = gw.copy()
    ordered_attendee = np.zeros(N, dtype=np.int32)

    # harden
    for _ in range(N):
        id_max = np.ndarray.argmax(T)
        i = id_max // N
        j = id_max % N
        ordered_attendee[j] = i
        T[i, :] = 0
        T[:, j] = 0
        # i,j
    return ordered_attendee


ordered_attendee = harden_assignment(gw)

"""ordered_attendee[0]=0
ordered_attendee[7]=17
ordered_attendee[18]=10
"""

ordered_names = [guest_name[ordered_attendee[i]] for i in range(N)]

import pandas as pd

pd.DataFrame(np.array(ordered_names).reshape((4, 5)))

"""## 解のRefinement

GW解をもとに貪欲法で元のQP問題を解く
"""

eval_gw = lambda gw: gw.reshape((-1)) @ (np.kron(G2, D) @ gw.reshape((-1)))

assign_random = np.zeros([20, 20])
rand_perm = np.random.permutation(20)
for i in range(20):
    assign_random[i, rand_perm[i]] = 0.05

# ランダム割当の解の質
lst = []
for i in range(100):
    assign_random = np.zeros([20, 20])
    rand_perm = np.random.permutation(20)
    for i in range(20):
        assign_random[i, rand_perm[i]] = 0.05
    lst.append(eval_gw(assign_random))
lst = np.array(lst)
lst.mean(), lst.std()

eval_gw(assign_random), eval_gw(gw)

"""ランダムより評価値が良ければGWの解がQPの目的関数の意味でもある程度良いということ"""

gw_refine = gw.copy()
# gw_refine = assign_random.copy()  # ランダム初期値を用いる場合

while True:
    i_greed, j_greed = 0, 0
    current_value = eval_gw(gw_refine)
    for i in range(N - 1):
        for j in range(i + 1, N, 1):
            gw_refine_work = gw_refine.copy()
            gw_refine_work[[i, j], :] = gw_refine_work[[j, i], :]
            value_working = eval_gw(gw_refine_work)
            if value_working > current_value:
                current_value = value_working
                i_greed, j_greed = i, j

    if (i_greed, j_greed) != (0, 0):
        print((i_greed, j_greed))
        gw_refine[[i_greed, j_greed], :] = gw_refine[[j_greed, i_greed], :]
    else:
        break

eval_gw(gw_refine)

heat_mat(gw_refine, guest_name)

ordered_attendee = harden_assignment(gw_refine)
ordered_names = [guest_name[ordered_attendee[i]] for i in range(N)]

seat_chart = pd.DataFrame(np.array(ordered_names).reshape((4, 5)))
seat_chart[:2]

seat_chart[2:]
# %%