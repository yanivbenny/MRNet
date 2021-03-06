bar = 0
pgf = 0

import matplotlib

if pgf:
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
else:
    matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


w = 6
h = 5
b = 0.8
s = 2
fig = plt.figure()
fig.set_size_inches(w=w / s - b, h=h / s)
ax = plt.gca()
# ax = plt.axes([0.15, 0.1, 0.7, 0.8])
# # axbar = plt.axes([0.15, 0.1, 0.7, 0.03])
if bar:
    # ticks = [0, 20, 40, 60, 80, 100]
    ticks = [0, 25, 50, 75, 100]
    figbar = plt.figure()
    figbar.set_size_inches(w=b, h=h / s)
    eps = 0.05
    axbar = plt.axes([0.5 - eps, 0.25, 2 * eps, 0.5])
    # axbar2 = plt.axes([0.5-eps, 0.3, 2*eps, 0.6])

mode = 1
m = 0.0
# PGM:
if mode == 1:
    title = 'PGM without auxiliary (%)'
    labels = ['L-type', 'L-color', 'S-color', 'S-num', 'S-pos', 'S-size', 'S-type']
    vals_arr = np.array([[100, 98.1044, 15.6856, 100, 99.85466667, 65.6974, 15.6484],
                         [54.36225, 12.598, 14.9972, 13.228, 99.85466667, 60.2492, 14.4942],
                         [13.8475, 12.0174, 14.3058, 100, 12.63766667, 13.325, 12.2406],
                         [26.0045, 95.7244, 12.4772, 13.0805, 14.71233333, 14.0242, 11.1256]])

# PGM AUX:
if mode == 2:
    title = 'PGM with auxiliary (%)'
    labels = ['L-type', 'L-color', 'S-color', 'S-num', 'S-pos', 'S-size', 'S-type']
    vals_arr = np.array([[99.97, 97.5894, 41.2866, 99.779, 99.85633333, 66.7308, 62.9044],
                         [63.06475, 11.6704, 34.8912, 14.266, 94.61833333, 58.8014, 55.9824],
                         [13.63725, 12.3518, 12.2516, 99.705, 68.808, 12.5758, 14.31],
                         [42.72975, 92.3366, 13.4066, 13.375, 12.526, 13.1106, 14.1404]])

# RAVEN 1:
if mode == 3:
    title = 'RAVEN single stage (%)'
    labels = ['All', 'Center', '2x2Grid', '3x3Grid', 'L-R', 'U-D', 'OS-IS', 'OS-IG']
    vals_arr = np.array([[80.58, 87.71, 64.71, 64.28, 94.19, 94.34, 90.93, 67.95],
                         [72.03, 84.09, 47.17, 48.34, 89.99, 91.48, 87.59, 55.48],
                         [73.30, 90.49, 56.08, 53.42, 83.51, 83.03, 82.49, 64.09],
                         [58.91, 97.74, 62.17, 62.81, 42.25, 42.51, 56.83, 48.09]])

if mode == 4:
    title = 'RAVEN double stage (%)'
    labels = ['All', 'Center', '2x2Grid', '3x3Grid', 'L-R', 'U-D', 'OS-IS', 'OS-IG']
    vals_arr = np.array([[80.58, 87.71, 64.71, 64.28, 94.19, 94.34, 90.93, 67.95],
                         [76.55, 98.28, 67.48, 65.38, 81.57, 83.38, 81.50, 58.30],
                         [78.26, 90.98, 63.16, 64.71, 88.78, 92.18, 87.69, 60.32],
                         [77.72, 82.79, 61.71, 59.48, 91.68, 92.23, 89.93, 66.28]])

vals_arr /= 100
vmin = 0
size = 0.15
N = len(vals_arr[0])
x = np.array([100 / N] * N)

print('cmap')
cmap = plt.get_cmap("bwr")
cmap = truncate_colormap(cmap, 0.5, 1)
colors = np.ones(N)
colors = cmap(colors)
print(colors)

print('axes')

colors_full = colors.copy()
# vals = vals_full / 100
colors_full[:, 3] = colors_full[:, 3] * ((vals_arr[0] - vmin + m) / (1 - vmin + m))
ax.pie(x, radius=1 - 3 * size - 0.25, colors=colors_full,
       wedgeprops=dict(width=1 - 3 * size - 0.25, edgecolor='w'))

colors_high = colors.copy()
# vals = vals_high / 100
colors_high[:, 3] = colors_high[:, 3] * ((vals_arr[1] - vmin + m) / (1 - vmin + m))
ax.pie(x, radius=1 - size, colors=colors_high, labels=[labels[i] for i in range(N)],
       wedgeprops=dict(width=size, edgecolor='w'))
colors_mid = colors.copy()
# vals = vals_mid / 100
colors_mid[:, 3] = colors_mid[:, 3] * ((vals_arr[2] - vmin + m) / (1 - vmin + m))
ax.pie(x, radius=1 - 2 * size, colors=colors_mid,
       wedgeprops=dict(width=size, edgecolor='w'))
colors_low = colors.copy()
# vals = vals_low / 100
colors_low[:, 3] = colors_low[:, 3] * ((vals_arr[3] - vmin + m) / (1 - vmin + m))
pcm = ax.pie(x, radius=1 - 3 * size, colors=colors_low,
             wedgeprops=dict(width=size, edgecolor='w'))

print('set')

# cmap = cmap()
# cmap = matplotlib.cm.cool
norm = matplotlib.colors.Normalize(vmin=vmin * 100, vmax=100)

# axbar = plt.axes([0.85, 0.17, 0.03, 0.63])
# axbar = plt.axes([0.15, 0.1, 0.7, 0.03])
if bar:
    # matplotlib.colorbar.ColorbarBase(axbar, cmap=cmap, extend='both', orientation='vertical', norm=norm)
    matplotlib.colorbar.ColorbarBase(axbar, cmap=cmap, orientation='vertical', norm=norm)

print('show')

# ax.set(aspect="equal", title=title)

# fig.set_size_inches(w=3.5/1.3, h=2.5/1.3)
if bar:
    # axbar.set_yticks([0, 20, 40, 60, 80, 100])
    axbar.yaxis.set_ticks_position('left')
    # cax2 = axbar.twinx()
    # cax2.set_ylim(-0, 100)
    # cax2.set_yticks(ticks)
    # cax2.patch.set_visible(False)
if pgf:
    if not bar:
        fig.savefig(f'piechart_{mode}c.pgf')
    else:
        figbar.savefig(f'piechart_colorbar_l.pgf')
plt.show()
print('done')