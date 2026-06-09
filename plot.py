import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import os
from SRNN import model_srnn
from SRNN import inference_network
from SRNN import train
from sklearn.metrics import mean_squared_error

np.random.seed(131)
torch.manual_seed(131)

# ── Load saved results ──────────────────────────────────────────────────────
result_path = './result/sim_model_hidden8_fold0.pt'
m1 = torch.load(result_path, weights_only=False, map_location='cpu')

device = torch.device('cpu')
dtype  = torch.float32

y_test_np    = m1['y_test']
X_test_np    = m1['X_test']
label_test   = m1['label_test']   # (trials, time, 1)

y_test  = torch.tensor(y_test_np,       dtype=dtype, device=device)
X_test  = torch.tensor(X_test_np,       dtype=dtype, device=device)

# ── Rebuild model ───────────────────────────────────────────────────────────
input_shape         = X_test.shape[2]
num_tv              = m1['num_tv']
hidden_shape        = m1['hidden_shape']
neural_private_shape = m1['neural_private_shape']

model    = model_srnn.Model(input_shape, num_tv, hidden_shape, neural_private_shape).to(device)
rnninfer = inference_network.RNNInfer(input_shape, hidden_shape).to(device)

model.load_state_dict(m1['model_state_dict'])
rnninfer.load_state_dict(m1['rnninfer_state_dict'])

# ── Run eval ────────────────────────────────────────────────────────────────
y_pred_np, _, pos_test, _, _, _, _, _ = train.eval_(model, rnninfer, X_test, y_test, device)
# pos_test: (trials, time, num_tv) — log posteriors
inferred_states = np.exp(pos_test).argmax(axis=-1)   # (trials, time)
true_states     = label_test[:, :, 0].astype(int)    # (trials, time)

os.makedirs('./plot', exist_ok=True)

# ── Plot 1: Neural reconstruction ───────────────────────────────────────────
mse_per_trial = np.array([
    mean_squared_error(y_test_np[i].ravel(), y_pred_np[i].ravel())
    for i in range(len(y_test_np))
])
trial = int(np.argmin(mse_per_trial))

y_true = y_test_np[trial].T   # (neurons, time)
y_pred = y_pred_np[trial].T

gap      = np.full((5, y_true.shape[1]), np.nan)
combined = np.concatenate([y_true, gap, y_pred], axis=0)

cmap_img = plt.cm.cividis.copy()
cmap_img.set_bad(color='white')
vmin = min(y_true.min(), y_pred.min())
vmax = max(y_true.max(), y_pred.max())

fig, ax = plt.subplots(figsize=(4, 3))
im = ax.imshow(combined, aspect='auto', cmap=cmap_img, vmin=vmin, vmax=vmax)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title('Neural Reconstruction', fontsize=10)
ax.set_xlabel('Time')

n_true, n_pred = y_true.shape[0], y_pred.shape[0]
ax.text(-8, n_true / 2,           'True\nNeurons',      va='center', ha='center', fontsize=9, rotation=90)
ax.text(-8, n_true + 5 + n_pred / 2, 'Predicted\nNeurons', va='center', ha='center', fontsize=9, rotation=90)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])
cbar.set_ticklabels(['Low', 'Mid', 'High'])
plt.tight_layout()
plt.savefig('./plot/neural_recon.png', dpi=150, bbox_inches='tight')
print('Saved: ./plot/neural_recon.png')
plt.close()

# ── Plot 2: True vs Inferred States ─────────────────────────────────────────
color_list = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442']
cmap_states = mcolors.ListedColormap(color_list[:num_tv])

fig, axes = plt.subplots(2, 1, figsize=(5, 3), sharex=True)

axes[0].imshow(true_states,     aspect='auto', cmap=cmap_states, vmin=0, vmax=num_tv-1, interpolation='nearest')
axes[0].set_title('True States',     fontsize=10)
axes[0].set_ylabel('Trial'); axes[0].set_xticks([])

axes[1].imshow(inferred_states, aspect='auto', cmap=cmap_states, vmin=0, vmax=num_tv-1, interpolation='nearest')
axes[1].set_title('Inferred States', fontsize=10)
axes[1].set_ylabel('Trial'); axes[1].set_xlabel('Time')

plt.tight_layout()
plt.savefig('./plot/states.png', dpi=150, bbox_inches='tight')
print('Saved: ./plot/states.png')
plt.close()
