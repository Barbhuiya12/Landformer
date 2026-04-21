# Quick Start Guide - Swin Surrogate Model

## 🚀 Running Your First Training

### Step 1: Update Configuration

Edit `config_swin.yaml` and set your data paths:

```yaml
data:
  paths:
    forcing: 'YOUR_FORCING_PATH_HERE'
    parameters: 'YOUR_PARAMETERS_PATH_HERE'
    target_files:
      evap: 'YOUR_EVAP_PATH_HERE'
      sm: 'YOUR_SM_PATH_HERE'
      Q: 'YOUR_Q_PATH_HERE'

output:
  base_dir: './output'
  experiment_name: 'my_first_experiment'

hardware:
  device: 'cuda:0'  # Change to 'cpu' if no GPU
```

### Step 2: Run Training

```bash
# Navigate to the directory
cd /home/siddik/my_data/storage/Surrogate_final/surro_final_swin

# Run training
python main.py --config config_swin.yaml
```

### Step 3: Monitor Progress

Training will display:
- Progress bars for each epoch
- Loss values (train and validation)
- Metrics (NSE, KGE, Correlation)
- ETA for completion

### Step 4: View Results

After training, check:
```bash
ls output/my_first_experiment/
```

You'll find:
- `models/best_model.pth` - Your trained model
- `training.log` - Detailed logs
- `metrics/*.json` - Performance metrics
- `plots/*.png` - Training visualizations

---

## 🎛️ Key Configuration Options

### For Quick Training (Testing)
```yaml
training:
  epochs: 5
  batch_size: 50

data:
  grid_sampling:
    train_grids: 1000
    validation_grids: 50
```

### For Full Training (Production)
```yaml
training:
  epochs: 40
  batch_size: 100

data:
  grid_sampling:
    train_grids: 10000
    validation_grids: 100
```

### For Multi-Task Learning
```yaml
model:
  model_type: 'multi_task_swin'

data:
  variables:
    target: ['evap', 'sm', 'Q']
```

---

## 🔍 Common Commands

```bash
# Basic training
python main.py

# With custom config
python main.py --config my_config.yaml

# Resume from checkpoint
python main.py --resume

# Validate config only
python main.py --validate

# Show system info
python main.py --info
```

---

## 📊 Expected Training Time

| Grids | Epochs | GPU | Approximate Time |
|-------|--------|-----|------------------|
| 1,000 | 10 | V100 | ~30 minutes |
| 10,000 | 40 | V100 | ~4-6 hours |
| 10,000 | 40 | A100 | ~2-3 hours |

*Times vary based on sequence length and model size*

---

## ✅ Verify Installation

```bash
# Check if all files are present
ls -la

# Expected output:
# - main.py
# - config_swin.yaml
# - requirements.txt
# - README.md
# - src/

# Test imports
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import xarray; print('xarray:', xarray.__version__)"
```

---

## 🆘 Quick Fixes

### "Module not found"
```bash
pip install -r requirements.txt
```

### "CUDA out of memory"
Reduce batch size in `config_swin.yaml`:
```yaml
training:
  batch_size: 32  # Reduce from 100
```

### "Configuration file not found"
Make sure you're in the right directory:
```bash
pwd  # Should end with surro_final_swin
```

---

## 📁 Important Files

| File | Purpose |
|------|---------|
| `config_swin.yaml` | **Main configuration** - Edit this! |
| `main.py` | Training entry point |
| `src/models/swin_transformer.py` | Model architecture |
| `src/training/swin_trainer.py` | Training loop |
| `README.md` | Full documentation |

---

**Need help?** Check the full [README.md](README.md) for detailed documentation.
