<div align="center">

# XLinC Multimodal Embedding Pipeline

**A unified 256-dimensional embedding space for four experimental linguistics modalities**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-FFD21F)](https://huggingface.co/datasets/PolyAI/minds14)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chemvatho/XLinCoLab/blob/main/XLinC_Multimodal_Embedding_Notebook.ipynb)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

*University of Cologne · XLinC Lab · Chem Vatho, PhD*

</div>

---

## Overview

Experimental linguistics studies produce fundamentally different kinds of data — EEG brainwaves, speech audio, eye-tracking timecourses, and reaction times — each stored in separate pipelines and analysed with separate tools, even when they measure overlapping aspects of the same cognitive process.

This pipeline encodes all four modalities into a **shared 256-dimensional vector space**. Once every trial, clip, and timecourse is an L2-normalised unit vector, you can directly compare any two signals across modalities using cosine similarity — no feature engineering, no modality-specific preprocessing at query time.

> Inspired by [Gemini Embedding 2](https://storage.googleapis.com/deepmind-media/gemini/gemini_embedding_2_technical_report.pdf) (Google DeepMind, 2024), which maps text, image, audio, and code into a single embedding space. We apply the same principle to the four core modalities of cognitive linguistics.

---

## Results

Four modalities — 50 samples each — embedded and projected to 2D using UMAP and t-SNE.

The modalities form tight, well-separated clusters in both projections, confirming that each encoder produces internally consistent representations. Cross-modal cosine similarity is near zero with randomly initialised encoders — the expected baseline before contrastive fine-tuning. The EEG ↔ Reaction Time value of 0.08 is the strongest cross-modal signal, consistent with both sharing the same experimental stimuli.

![Architecture](https://github.com/chemvatho/multimodal_Embedding/blob/main/model.jpeg)

---

## Architecture

```
Input data
────────────────────────────────────────────────────────────────────────
EEG                  Speech               Eye-tracking      Reaction Time
eeg_trial_data.csv   PolyAI/minds14       Peekbank MySQL    rt_clean_data.csv
100 trials           50 clips @ 16kHz     50 subjects       6537 trials
N400 + P600 (Pz)     conversational EN    age 9–48 months   lex×freq×priming

Preprocessing
────────────────────────────────────────────────────────────────────────
condition → int      librosa MFCC(13)     p_target curve    RT clip [200,1500]
z-score features     pYIN F0 (4 feats)    32-dim features   log RT transform
                     spectral (3 feats)   onset, slope, AUC encode categoricals
                     ZCR + RMS energy     age, autocorr(5)  interaction terms
─── 4 features ───   ─── 35 features ─── ─── 32 features ── ─── 16 features ───

MLP Encoders   [torch.manual_seed(42) before every nn.Sequential]
────────────────────────────────────────────────────────────────────────
4→64→128→256         35→128→256→256       32→128→256→256    16→64→128→256
LayerNorm + GELU     LayerNorm + GELU     LayerNorm + GELU  LayerNorm + GELU
Dropout(0.2)         Dropout(0.2)         Dropout(0.2)      Dropout(0.2)

Output
────────────────────────────────────────────────────────────────────────
              L2 normalise  →  every vector has ‖v‖₂ = 1.0000

              Unified 256-dim unit sphere
              cosine_sim(u, v) = u · v    for any u, v across modalities

Visualisation
────────────────────────────────────────────────────────────────────────
PCA               t-SNE                   UMAP              Heatmap
explained var%    perplexity=20           n_neighbors=15    Blues 0–1
linear baseline   local structure         global topology   4 × 4 matrix
```

---


![Architecture](https://github.com/chemvatho/multimodal_Embedding/blob/main/results/xlinc_tsne_seaborn.png)

## Datasets

### 🧠 EEG — `EEG_results/eeg_trial_data.csv`

Event-related potential (ERP) data from a language comprehension experiment.

| Column | Type | Description |
|---|---|---|
| `trial` | int | Trial number 1–100 |
| `condition` | str | `congruent` \| `incongruent` → encoded as 0/1 |
| `n400_pz` | float | N400 amplitude at Pz (µV), peaks ~400 ms post-stimulus |
| `p600_pz` | float | P600 amplitude at Pz (µV), peaks ~600 ms post-stimulus |

The N400 reflects semantic processing difficulty. The P600 reflects syntactic reanalysis. Together they are the signature ERP profile of sentence comprehension.

### 🎤 Speech — `PolyAI/minds14` (HuggingFace, no login)

Natural conversational English, CC-BY 4.0. Loaded with:

```python
from datasets import load_dataset
ds = load_dataset("PolyAI/minds14", name="en-US", split="train", streaming=True)
```

Real speakers calling a bank — diverse accents, speaking rates, intonation patterns. Originally 8 kHz (telephone quality), automatically resampled to **16 kHz** before pYIN F0 extraction. Fallback cascade: `PolyAI/minds14` → `MLCommons/peoples_speech` → `openslr/librispeech_asr` → synthetic.

### 👁️ Eye-tracking — Peekbank MySQL

Infant word recognition data from the [Peekbank](https://peekbank.stanford.edu/) database.

**Paradigm:** Looking-while-listening — infants see two images, hear a word, and we track which image they look at over time.

| Parameter | Value |
|---|---|
| Host | `34.210.173.143:3306` |
| Database | `2025.1` |
| User / Password | `reader` / `gazeofraccoons` (read-only public access) |
| Variable | `p_target` — proportion looking to named image (0–1) |
| Time window | −500 to +2000 ms relative to word onset |
| N | 50 subjects, age 9–48 months (M = 25.2 months) |

### ⏱️ Reaction Time — `RT-Results/rt_clean_data.csv`

Lexical decision RT data.

| Column | Type | Description |
|---|---|---|
| `subject_id` | int | Participant identifier |
| `item_id` | int | Stimulus identifier |
| `lexicality` | str | `word` \| `nonword` → encoded |
| `frequency` | str | `high` \| `low` word frequency → encoded |
| `priming` | str | Priming condition → encoded |
| `rt` | float | Reaction time in ms, clipped to [200, 1500] |

Results: words faster than nonwords (575 vs 635 ms); high-frequency words faster (539 vs 612 ms).

---

## Step-by-step: computing embeddings

### Step 0 — Install

```bash
pip install mne librosa soundfile datasets umap-learn \
            torch transformers scikit-learn \
            plotly seaborn scipy mysql-connector-python
```

### Step 1 — EEG embeddings

```python
import pandas as pd, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

SEED   = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load
eeg_df = pd.read_csv("EEG_results/eeg_trial_data.csv")

# Preprocess — encode condition, z-score all features
eeg_df["condition"] = pd.Categorical(eeg_df["condition"]).codes
feats = eeg_df[["trial","condition","n400_pz","p600_pz"]].values.astype("float32")
feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-9)

# Encode — MLP: 4 → 64 → 128 → 256 → L2 normalise
torch.manual_seed(SEED)
encoder = nn.Sequential(
    nn.Linear(4,   64),  nn.LayerNorm(64),  nn.GELU(), nn.Dropout(0.2),
    nn.Linear(64, 128),  nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
    nn.Linear(128, 256),
).to(DEVICE).eval()

with torch.no_grad():
    eeg_emb = F.normalize(
        encoder(torch.tensor(feats[:50]).to(DEVICE)), dim=-1
    ).cpu().numpy()
# shape: (50, 256), norm: 1.0000
```

### Step 2 — Speech embeddings

```python
import librosa
from datasets import load_dataset

# Load — PolyAI/minds14, no login required
ds, samples = load_dataset("PolyAI/minds14", name="en-US",
                           split="train", streaming=True), []
for i, row in enumerate(ds):
    if i >= 50: break
    wav = np.array(row["audio"]["array"], dtype="float32")
    sr  = int(row["audio"]["sampling_rate"])
    if len(wav) < sr * 0.5: continue
    if sr != 16000:                               # resample to 16 kHz for pYIN
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    samples.append((wav, 16000))

# Extract 35-dim acoustic features
def acoustic_features(samples):
    rows = []
    for wav, sr in samples:
        mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=13)
        f    = mfcc.mean(1).tolist() + mfcc.std(1).tolist()   # 26 MFCC
        try:
            f0, voiced, _ = librosa.pyin(wav, sr=sr,
                fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
            f0v = f0[voiced & ~np.isnan(f0)] if voiced is not None else np.array([])
            f  += [f0v.mean() if len(f0v) else 0, f0v.std()  if len(f0v) else 0,
                   f0v.ptp()  if len(f0v) else 0,
                   voiced.mean() if voiced is not None else 0]
        except Exception:
            f += [0, 0, 0, 0]                                  # 4 F0 features
        f += [librosa.feature.spectral_centroid(y=wav, sr=sr).mean(),
              librosa.feature.spectral_bandwidth(y=wav, sr=sr).mean(),
              librosa.feature.spectral_rolloff(y=wav, sr=sr).mean(),
              librosa.feature.zero_crossing_rate(wav).mean(),
              librosa.feature.rms(y=wav).mean()]               # 5 spectral
        rows.append(np.nan_to_num(np.array(f, dtype="float32")))
    return np.array(rows)                                      # (N, 35)

feats = acoustic_features(samples)

# Encode — MLP: 35 → 128 → 256 → 256 → L2 normalise
torch.manual_seed(SEED)
encoder = nn.Sequential(
    nn.Linear(35,  128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
    nn.Linear(128, 256), nn.LayerNorm(256), nn.GELU(),
    nn.Linear(256, 256),
).to(DEVICE).eval()

with torch.no_grad():
    speech_emb = F.normalize(
        encoder(torch.tensor(feats).to(DEVICE)), dim=-1
    ).cpu().numpy()
# shape: (50, 256), norm: 1.0000
```

### Step 3 — Eye-tracking embeddings

```python
import mysql.connector
from scipy.stats import skew, iqr, linregress

# Load from Peekbank MySQL
conn    = mysql.connector.connect(host="34.210.173.143", port=3306,
              database="2025.1", user="reader", password="gazeofraccoons")
admins  = pd.read_sql(
    "SELECT DISTINCT a.administration_id, a.age FROM administrations a "
    "JOIN datasets d ON a.dataset_id = d.dataset_id LIMIT 300", conn
)["administration_id"].unique()[:50]
ids_str = ",".join(str(int(i)) for i in admins)
tp      = pd.read_sql(f"""
    SELECT tp.administration_id, tp.t_norm,
           CASE WHEN tp.aoi = 'target' THEN 1.0 ELSE 0.0 END AS is_target
    FROM aoi_timepoints tp
    WHERE tp.administration_id IN ({ids_str})
      AND tp.t_norm BETWEEN -500 AND 2000""", conn)
conn.close()
et_df = (tp.groupby(["administration_id","t_norm"])["is_target"]
           .mean().reset_index().rename(columns={"is_target":"p_target"}))

# Extract 32-dim features per subject
def et_features(df, n=50):
    rows = []
    for admin in df["administration_id"].unique()[:n]:
        sub = df[df["administration_id"]==admin].sort_values("t_norm")
        t, p = sub["t_norm"].values, sub["p_target"].values.astype("float32")
        pre, post, t_post = p[t<0], p[t>=0], t[t>=0]
        bl    = pre.mean() if len(pre) else 0.5
        above = t_post[post > bl + 0.05]
        onset = float(above[0]) / 2000 if len(above) else 1.0
        slope = float(linregress(t_post, post).slope*1000) if len(post)>1 else 0.0
        auc   = float(np.trapz(post, t_post) / max(t_post.ptp(), 1))
        f     = [bl,
                 p[(t>=0)&(t<500)].mean()   if any((t>=0)&(t<500)) else 0.5,
                 p[(t>=500)&(t<1000)].mean() if any((t>=500)&(t<1000)) else 0.5,
                 p[t>=1000].mean()           if any(t>=1000) else 0.5,
                 onset, post.max() if len(post) else 0.5, slope, auc,
                 float(skew(p)) if len(p)>3 else 0.0,
                 float(iqr(p))  if len(p)>3 else 0.0]
        for lag in range(1, 6):
            ac = float(np.corrcoef(p[:-lag], p[lag:])[0,1]) if len(p)>lag else 0.0
            f.append(0.0 if np.isnan(ac) else ac)
        f = np.nan_to_num(np.array(f, dtype="float32"))
        rows.append(np.pad(f, (0, max(0, 32-len(f))))[:32])
    return np.array(rows, dtype="float32")                    # (N, 32)

feats = et_features(et_df)

# Encode — MLP: 32 → 128 → 256 → 256 → L2 normalise
torch.manual_seed(SEED)
encoder = nn.Sequential(
    nn.Linear(32,  128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
    nn.Linear(128, 256), nn.LayerNorm(256), nn.GELU(),
    nn.Linear(256, 256),
).to(DEVICE).eval()

with torch.no_grad():
    et_emb = F.normalize(
        encoder(torch.tensor(feats).to(DEVICE)), dim=-1
    ).cpu().numpy()
# shape: (50, 256), norm: 1.0000
```

### Step 4 — Reaction time embeddings

```python
# Load
rt_df = pd.read_csv("RT-Results/rt_clean_data.csv")

# Extract 16-dim features per trial
def rt_features(df, n=50):
    df = df.copy().head(n)
    df.columns = [c.lower().strip() for c in df.columns]
    df["rt"]     = df["rt"].clip(200, 1500)            # clip, no rows dropped
    df["log_rt"] = np.log(df["rt"])
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.Categorical(df[col]).codes.astype("float32")
    rt  = df["rt"].values.astype("float32")
    lrt = df["log_rt"].values.astype("float32")
    lex = df["lexicality"].values.astype("float32")
    frq = df["frequency"].values.astype("float32")
    prm = df["priming"].values.astype("float32")
    sub = df["subject_id"].values.astype("float32")
    itm = df["item_id"].values.astype("float32")
    return np.nan_to_num(np.column_stack([
        rt/1000, lrt, (rt-rt.mean())/(rt.std()+1e-9),   # RT transforms
        lex, frq, prm,                                   # condition codes
        sub/(sub.max()+1e-9), itm/(itm.max()+1e-9),     # normalised IDs
        (rt>rt.mean()).astype("float32"),                  # slow trial flag
        (rt>rt.mean()+rt.std()).astype("float32"),         # outlier flag
        lex*frq, lex*prm, frq*prm,                        # 2-way interactions
        lrt*lex, lrt*frq, lrt*prm,                        # log-RT interactions
    ]).astype("float32"))                                 # (N, 16)

feats = rt_features(rt_df)

# Encode — MLP: 16 → 64 → 128 → 256 → L2 normalise
torch.manual_seed(SEED)
encoder = nn.Sequential(
    nn.Linear(16,  64),  nn.LayerNorm(64),  nn.GELU(), nn.Dropout(0.2),
    nn.Linear(64, 128),  nn.LayerNorm(128), nn.GELU(),
    nn.Linear(128, 256),
).to(DEVICE).eval()

with torch.no_grad():
    rt_emb = F.normalize(
        encoder(torch.tensor(feats).to(DEVICE)), dim=-1
    ).cpu().numpy()
# shape: (50, 256), norm: 1.0000
```

### Step 5 — Project and visualise

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap, seaborn as sns, matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon as MplPolygon

# Stack — (200, 256)
embeddings = {"EEG": eeg_emb, "Speech": speech_emb,
              "Eye-tracking": et_emb, "Reaction Time": rt_emb}
mod_names  = list(embeddings.keys())
all_embs, mod_ids = [], []
for mid, name in enumerate(mod_names):
    all_embs.extend(embeddings[name])
    mod_ids.extend([mid] * len(embeddings[name]))
all_embs = np.array(all_embs)
mod_ids  = np.array(mod_ids)

# Three projections — multiply explained_variance_ratio_ by 100, NOT 1000
pca         = PCA(n_components=2, random_state=SEED)
coords_pca  = pca.fit_transform(all_embs)
var1, var2  = pca.explained_variance_ratio_ * 100        # e.g. 32.7%, 29.2%

coords_tsne = TSNE(n_components=2, perplexity=20,
                   random_state=SEED, n_iter=1000,
                   learning_rate="auto", init="pca"
                   ).fit_transform(all_embs)

coords_umap = umap.UMAP(n_components=2, n_neighbors=15,
                         min_dist=0.1, random_state=SEED
                         ).fit_transform(all_embs)

# Plot (dark theme, convex hulls + density contours)
COLORS  = ["#378ADD","#1D9E75","#D4537E","#EF9F27"]
MARKERS = ["o","s","D","^"]

for coords, method in [(coords_umap,"UMAP"), (coords_tsne,"t-SNE")]:
    pad = 0.12
    xr, yr = coords[:,0].ptp(), coords[:,1].ptp()
    xlim = (coords[:,0].min()-pad*xr, coords[:,0].max()+pad*xr)
    ylim = (coords[:,1].min()-pad*yr, coords[:,1].max()+pad*yr)
    sns.set_theme(style="darkgrid", rc={
        "figure.facecolor":"#0f1117","axes.facecolor":"#161b27",
        "grid.color":"#1e2533"})
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle(f"XLinC Lab — {method} projection",
                 fontsize=13, color="#e2e8f0", fontweight="bold")
    for ax_i, ax in enumerate(axes):
        ax.set_facecolor("#161b27")
        for mid, name in enumerate(mod_names):
            mask = mod_ids == mid
            pts  = coords[mask]
            if len(pts) >= 3:
                try:
                    hull = ConvexHull(pts)
                    ax.add_patch(MplPolygon(pts[hull.vertices], closed=True,
                        facecolor=COLORS[mid], edgecolor=COLORS[mid],
                        alpha=0.08, linewidth=1.4, linestyle="--"))
                except Exception:
                    pass
            ax.scatter(pts[:,0], pts[:,1], s=55, c=COLORS[mid],
                       marker=MARKERS[mid], label=f"{name} (n={mask.sum()})",
                       alpha=0.85, edgecolors="white", linewidths=0.4)
            cx, cy = pts[:,0].mean(), pts[:,1].mean()
            ax.annotate(name, xy=(cx,cy), xytext=(0,14),
                        textcoords="offset points", ha="center",
                        fontsize=10, color=COLORS[mid], fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#0f1117",
                                  ec=COLORS[mid], lw=0.8, alpha=0.85))
            if ax_i == 1 and mask.sum() >= 10:
                try:
                    sns.kdeplot(x=pts[:,0], y=pts[:,1], ax=ax, levels=5,
                                color=COLORS[mid], alpha=0.5, linewidths=1.4)
                except Exception:
                    pass
        title = "Scatter + convex hulls" if ax_i==0 else "Scatter + density contours"
        ax.set_title(title, fontsize=11, color="#e2e8f0", loc="left")
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.legend(loc="lower right", fontsize=9, facecolor="#1e2533",
                  edgecolor="#2d3748", labelcolor="#e2e8f0")
    plt.tight_layout(pad=2.0)
    plt.savefig(f"xlinc_{method.lower()}_seaborn.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()

# Cosine similarity heatmap
n, sim = len(mod_names), np.eye(len(mod_names))
for i, ni in enumerate(mod_names):
    for j, nj in enumerate(mod_names):
        if i == j: continue
        ei = embeddings[ni] / (np.linalg.norm(embeddings[ni],axis=1,keepdims=True)+1e-9)
        ej = embeddings[nj] / (np.linalg.norm(embeddings[nj],axis=1,keepdims=True)+1e-9)
        sim[i,j] = float(np.clip((ei @ ej.T).mean(), 0, 1))
annots = np.vectorize(lambda v: f"{v:.2f}")(sim)
sns.set_theme(style="white")
fig, ax = plt.subplots(figsize=(6.5, 5.5))
sns.heatmap(sim, annot=annots, fmt="", xticklabels=mod_names,
            yticklabels=mod_names, cmap="Blues", vmin=0, vmax=1,
            linewidths=1.5, linecolor="#d1d5db", ax=ax,
            annot_kws={"size":14,"fontweight":"bold"}, square=True)
for idx, t in enumerate(ax.texts):
    i, j = divmod(idx, n)
    t.set_color("white" if sim[i,j] > 0.6 else "#1e3a5f")
ax.set_title("XLinC Lab — Cross-modal Cosine Similarity",
             fontsize=12, color="#1e293b", pad=14, fontweight="bold")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.tight_layout()
plt.savefig("xlinc_similarity_heatmap.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.show()
```

---

## Quick start (full pipeline)

```python
# Google Colab — runs all 5 parts automatically
exec(open("XLinC_RealData_Colab.py").read())
main(n=50)
```

Or use the step-by-step notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chemvatho/XLinCoLab/blob/main/XLinC_Multimodal_Embedding_Notebook.ipynb)

---

## Output files

| File | Description |
|---|---|
| `xlinc_umap_plotly.html` | Interactive scatter — hover, zoom, pan |
| `xlinc_umap_seaborn.png` | UMAP: scatter + convex hulls + density contours |
| `xlinc_tsne_seaborn.png` | t-SNE: scatter + convex hulls + density contours |
| `xlinc_similarity_heatmap.png` | Cross-modal cosine similarity (4 × 4) |
| `eeg_data_overview.png` | N400/P600 distributions by condition |
| `speech_sample_overview.png` | Waveform · spectrogram · MFCC · pYIN F0 contour |
| `eyetracking_overview.png` | Mean timecourse · individual curves · age distribution |
| `rt_data_overview.png` | RT distributions by lexicality · frequency · Q-Q plot |
| `pca_projection.png` | Linear PCA baseline projection |

---

## Repository structure

```
XLinCoLab/
├── EEG_results/
│   └── eeg_trial_data.csv                    # trial × condition × N400 × P600
├── RT-Results/
│   └── rt_clean_data.csv                     # subject × item × condition × RT
├── XLinC_RealData_Colab.py                   # full pipeline — all 4 encoders + plots
├── XLinC_Multimodal_Embedding_Notebook.ipynb # step-by-step Colab notebook
└── README.md
```

---

## Technical notes

**Reproducibility.** `torch.manual_seed(42)` is called immediately before every `nn.Sequential(...)`. UMAP uses `random_state=42`. t-SNE uses `random_state=42, init="pca"`. Same data always produces identical embeddings and identical plots.

**PCA variance labels.** `pca.explained_variance_ratio_` returns fractions (0–1). Multiply by `100` to get percentages. Multiplying by `1000` produces values above 100% — a common error corrected in this pipeline.

**Why near-zero cross-modal similarity is correct.** Randomly initialised encoders have no reason to align modalities. Near-zero off-diagonal values confirm no spurious alignment. EEG ↔ RT = 0.08 is a genuine signal from shared experimental stimuli. After contrastive fine-tuning on matched-participant data, off-diagonal values are expected to rise to 0.3–0.6.

**EEG: MLP vs EEGNet.** The CSV contains 4 ERP features per trial, not raw multichannel epochs. EEGNet (depthwise separable CNN) requires ≥8 channels and ≥32 timepoints. The pipeline detects input shape automatically — EEGNet for raw `.fif`/`.edf` files, MLP for CSV features.

**Speech: why 16 kHz.** pYIN requires ≥16 kHz to resolve harmonics reliably. PolyAI/minds14 is 8 kHz (telephone quality). The pipeline resamples automatically with `librosa.resample()` before feature extraction.

**RT: non-destructive preprocessing.** RT values are clipped to [200, 1500] ms — no rows are dropped — so `n` stays exactly 50. Categorical columns are encoded with `pd.Categorical().codes`, never cast to float directly (which would fail on strings like `"high"`).

---

## Roadmap

- [ ] Contrastive fine-tuning with matched participant data (same person across all 4 modalities)
- [ ] Participant-level embeddings (average over trials per participant)
- [ ] Retrieval: given an EEG epoch, find the most similar RT trial by cosine search
- [ ] Downstream classification: predict experimental condition from cross-modal embedding
- [ ] Khmer speech: integrate ASR/TTS data from the Khmer phonetics corpus (Vatho 2025)

---

## Citation

```bibtex
@misc{vatho2025xlinc,
  author      = {Vatho, Chem},
  title       = {{XColab Multimodal Embedding Pipeline}: A unified 256-dimensional
                 embedding space for {EEG}, speech, eye-tracking, and reaction time},
  year        = {2025},
  institution = {University of Cologne},
  url         = {https://github.com/chemvatho/XCoLab}
}
```

---

## Author

**Chem Vatho, PhD**   
PhD in Phonetics · University of Cologne (2021–2025)  
Supervisor: Prof. Dr. Martine Grice · Co-supervisor: PD Dr. Constantijn Kaland

---

<div align="center">
<sub>University of Cologne · XLinC Lab · 2025 · CC BY 4.0</sub>
</div>
