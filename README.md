# Supervised Switching Recurrent Neural Networks (Supervised_SRNN)

This repository contains the official implementation of **Supervised Switching Recurrent Neural Networks (Supervised_SRNN)**. 

---

## Installation

### 1. Clone the repository
```bash
git clone <repo-url>
cd Supervised_SRNN
```

### 2. Create the conda environment

**Windows (NVIDIA GPU):**
```bash
conda env create -f environment.yml
conda activate SSRNN
```

```

> **Note:** If the environment creation fails, manually install any missing packages via `pip install <package>`. All dependencies are standard scientific Python packages.

---

## Running the Model

### Single fold (example)
```bash
python array_hidden8.py --config config.yaml --fold 0
```

### Full 5-fold run
```bash
for %i in (0 1 2 3 4) do python array_hidden8.py --config config.yaml --fold %i
```
This sequentially runs all 5 folds. Note that a full run may take a significant amount of time depending on your hardware.

---

## Post-run Analysis

After completing the 5-fold run, generate figures by running:
```bash
python plot.py
```
Output figures will be saved to the `plot/` directory.

---
