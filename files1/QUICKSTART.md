# Quick Start Guide
## RNN Fileless Malware Detection - Ubuntu

### ⚡ 5-Minute Setup

```bash
# 1. Extract the package
tar -xzf rnn-malware-detection.tar.gz
cd rnn-malware-detection

# 2. Run automated setup
chmod +x setup.sh
./setup.sh

# 3. Copy your data
cp /path/to/your/newDataset.csv data/
cp /path/to/your/analysis_file.xlsx data/

# 4. Run analysis
./run.sh
```

### 📊 What You'll Get

After running, check `outputs/` for:
- **model_comparison.png** - Performance comparison
- **roc_curves_comparison.png** - ROC curves
- **confusion_matrix_*.png** - 4 confusion matrices
- **training_history_*.png** - Training curves
- **model_comparison.csv** - Metrics table
- **ANALYSIS_REPORT.md** - Full analysis

### 🎯 Expected Results

Based on test data (50 samples):
- **GRU Model:** 100% accuracy (best)
- **Bidirectional LSTM:** 93.3% accuracy
- **Simple RNN:** 93.3% accuracy
- **LSTM:** 66.7% accuracy

Training time: 3-4 minutes on modern CPU

### 🔧 Customization

Edit `rnn_fileless_malware_detection.py` to:
- Change dataset paths (line ~450)
- Adjust training parameters (line ~470)
- Select specific models (line ~467)

### 🐛 Troubleshooting

**Import Error:**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**File Not Found:**
```bash
# Ensure data is in correct location
ls -la data/
```

**Memory Error:**
```python
# In script, reduce batch_size
batch_size=4  # Line ~470
```

### 📚 Full Documentation

See `README.md` for complete documentation.

### 🚀 Quick Test

Run with example data:
```bash
source venv/bin/activate
python3 -c "
import pandas as pd
from rnn_fileless_malware_detection import FilelessMalwareRNNClassifier

# Quick test
print('Testing RNN system...')
classifier = FilelessMalwareRNNClassifier()
print('✓ System ready!')
"
```

---

**Questions?** Check README.md or ANALYSIS_REPORT.md
