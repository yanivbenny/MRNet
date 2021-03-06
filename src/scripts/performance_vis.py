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

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick


def y_fmt(tick_val, pos):
    if tick_val >= 1000000:
        val = int(tick_val) / 1000000
        if val == int(val):
            return '{:d} M'.format(int(val))
        else:
            return '{:.01f} M'.format(val)
    elif tick_val >= 1000:
        val = int(tick_val) / 1000
        if val == int(val):
            return '{:d} k'.format(int(val))
        else:
            return '{:.01f} k'.format(val)
    else:
        assert tick_val <= 0 or tick_val > 100, f'{tick_val}'
        val = tick_val - (tick_val % 100)
        return val

# Specify path to file
file = ''
assert len(file), 'performance.pickle file was not specified'
with open(file, 'rb') as fp:
    d = pickle.load(fp)

keys = ['shape-color', 'shape-type', 'shape-size', 'shape-number', 'shape-position', 'line-color', 'line-type']
colors = plt.cm.tab10(np.linspace(0, 1, 10))[:7]
# colors = ['r', 'b', 'g', 'o', '']
vals = [None] * len(keys)
counts = [0] * len(keys)
for key, val in d['acc_regime'].items():
    for i, k in enumerate(keys):
        if k in key and val[i] is not None:
            if vals[i] is None:
                vals[i] = val.copy()
            else:
                for j in range(len(vals[i])):
                    vals[i][j] += val[j]
            counts[i] += 1
            break

for i in range(len(vals)):
    for j in range(len(vals[i])):
        vals[i][j] /= counts[i]

N = 0
if N > 1:
    for i in range(len(vals)):
        vals[i] = [np.mean(vals[i][j - N:j]) for j in range(N, len(vals[i]))]
    t = d['t'][N:]
    test_acc = [np.mean(d['test_acc'][j - N:j]) for j in range(N, len(d['test_acc']))]
else:
    t = d['t']
    test_acc = d['test_acc']

fig = plt.figure(figsize=(5.5, 3.5))
plt.plot(t, test_acc, color='k', linestyle='--', label='total')
for i in range(len(keys)):
    plt.plot(t, vals[i], label=keys[i])
plt.legend(loc=8, ncol=4)
fig.gca().xaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
plt.show()

label_count = {}
for key in keys:
    label_count[key] = 1

fig = plt.figure(figsize=(6, 4.6))
ax = plt.axes([0.1, 0.1, 0.8, 0.6])
# ax = plt.subplot(111)
# ax = plt.axes()
for key, val in d['acc_regime'].items():
    label = None
    if key.endswith('prog'):
        marker = '-'
        label = key.split('-prog')[0]
    elif key.endswith('union'):
        marker = '--'
        label = key.split('-union')[0]
    elif key.endswith('xor'):
        marker = ':'
        label = key.split('-xor')[0]
    elif key.endswith('or'):
        marker = '-.'
        label = key.split('-or')[0]
    elif key.endswith('and'):
        marker = '-.'
        label = key.split('-and')[0]
    if any([k in key for k in keys]) and val[0] is not None:
        if label_count[label]:
            label_count[label] -= 1
        else:
            label = None
        i = np.where([k in key for k in keys])[0][0]
        if N > 1:
            val_i = [np.mean(val[j - N:j]) for j in range(N, len(val))]
        else:
            val_i = val
        plt.plot(t, val_i, color=colors[i], linestyle=marker, label=label)
plt.xlabel('iteration')
plt.ylabel('Accuracy')
plt.grid()

# make legend
import matplotlib.lines as mlines

legend_lines = []
marker = '-'
legend_lines.append(mlines.Line2D([], [], color='k', linestyle=marker, label='progression'))
marker = '--'
legend_lines.append(mlines.Line2D([], [], color='k', linestyle=marker, label='union'))
marker = ':'
legend_lines.append(mlines.Line2D([], [], color='k', linestyle=marker, label='xor'))
marker = '-.'
legend_lines.append(mlines.Line2D([], [], color='k', linestyle=marker, label='or/and'))
for i, key in enumerate(keys):
    legend_lines.append(mlines.Line2D([], [], color=colors[i], label=key))

plt.legend(handles=legend_lines, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.4))
plt.show()
# plt.legend(loc=6, ncol=1)
# ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1),
#           ncol=4, fancybox=True, shadow=True)
fig.gca().xaxis.set_major_formatter(tick.FuncFormatter(y_fmt))

if pgf:
    fig.savefig(f'performance2.pgf')
else:
    plt.show()
print('pause')