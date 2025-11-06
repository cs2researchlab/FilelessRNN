# RNN-Based Fileless Malware Detection System
## Comprehensive Analysis Report

**Date:** November 2, 2025  
**Analyst:** Hunter  
**System:** TensorFlow/Keras RNN Implementation

---

## Executive Summary

This report presents the results of training and comparing four different Recurrent Neural Network (RNN) architectures for fileless malware detection. The models were trained on a combined dataset of 50 samples (26 malicious, 24 benign) containing 33 behavioral and system-level features.

### Key Findings

**🏆 Best Performing Model: GRU (Gated Recurrent Unit)**
- **Accuracy:** 100%
- **Precision:** 100%
- **Recall:** 100%
- **F1 Score:** 100%
- **AUC-ROC:** 1.000

The GRU model achieved perfect classification on the test set with zero false positives and zero false negatives.

---

## Dataset Details

### Dataset Composition
- **Total Samples:** 50
- **Training Set:** 35 samples (70%)
- **Test Set:** 15 samples (30%)
- **Features:** 33 system and behavioral indicators
- **Class Distribution:**
  - Malicious (Label=0): 26 samples (52%)
  - Benign (Label=1): 24 samples (48%)

### Feature Categories
The dataset includes comprehensive system artifacts:
- Process handles and DLL modules
- Registry events (read/write/delete)
- Network connections (TCP/UDP)
- HTTP(S) requests and DNS queries
- Process memory characteristics
- Mutex and thread information
- Service scan results

### Data Sources
1. **newDataset.csv** - 45 samples (baseline dataset)
2. **analysis_of_new_malware_samples_to_be_added_to_the_dataset.xlsx** - 5 new malicious samples
   - Sorebrect
   - PowerSniff
   - Lemon Duck
   - Novter
   - DarkWatchman

---

## Model Architectures

### 1. Simple RNN
**Architecture:**
- Layer 1: SimpleRNN (64 units, return_sequences=True)
- Dropout (30%)
- Layer 2: SimpleRNN (32 units)
- Dropout (30%)
- Dense (32 units, ReLU)
- Dropout (30%)
- Output (1 unit, Sigmoid)
- **Total Parameters:** 8,417 (32.88 KB)

### 2. LSTM (Long Short-Term Memory)
**Architecture:**
- Layer 1: LSTM (64 units, return_sequences=True)
- Dropout (30%)
- Layer 2: LSTM (32 units)
- Dropout (30%)
- Dense (32 units, ReLU)
- Dropout (30%)
- Output (1 unit, Sigmoid)
- **Total Parameters:** 30,401 (118.75 KB)

### 3. GRU (Gated Recurrent Unit)
**Architecture:**
- Layer 1: GRU (64 units, return_sequences=True)
- Dropout (30%)
- Layer 2: GRU (32 units)
- Dropout (30%)
- Dense (32 units, ReLU)
- Dropout (30%)
- Output (1 unit, Sigmoid)
- **Total Parameters:** 23,361 (91.25 KB)

### 4. Bidirectional LSTM
**Architecture:**
- Layer 1: Bidirectional LSTM (64 units, return_sequences=True)
- Dropout (30%)
- Layer 2: Bidirectional LSTM (32 units)
- Dropout (30%)
- Dense (32 units, ReLU)
- Dropout (30%)
- Output (1 unit, Sigmoid)
- **Total Parameters:** 77,121 (301.25 KB)

---

## Comparative Performance Results

### Model Rankings (by AUC-ROC)

| Rank | Model               | Accuracy | Precision | Recall | F1 Score | AUC-ROC | Parameters |
|------|---------------------|----------|-----------|--------|----------|---------|------------|
| 🥇 1  | **GRU**            | 1.000    | 1.000     | 1.000  | 1.000    | 1.000   | 23,361     |
| 🥈 2  | **Bidirectional LSTM** | 0.933 | 0.875     | 1.000  | 0.933    | 0.982   | 77,121     |
| 🥉 3  | **Simple RNN**     | 0.933    | 0.875     | 1.000  | 0.933    | 0.946   | 8,417      |
| 4    | **LSTM**           | 0.667    | 0.625     | 0.714  | 0.667    | 0.679   | 30,401     |

### Performance Analysis

#### GRU Model (Best Performer)
- **Perfect Classification:** Achieved 100% across all metrics
- **Confusion Matrix:** Zero misclassifications
  - True Positives (Malicious): 8/8 (100%)
  - True Negatives (Benign): 7/7 (100%)
- **Training Characteristics:**
  - Early stopping at epoch 23 (best epoch: 8)
  - Converged quickly with stable validation performance
  - Optimal balance between model complexity and performance

#### Bidirectional LSTM (Second Best)
- **Strong Performance:** 93.3% accuracy, 98.2% AUC
- **Confusion Matrix:**
  - 1 false positive (classified benign as malicious)
  - 0 false negatives
- **Highest Model Complexity:** 77,121 parameters
- **Perfect Recall:** Detected all malicious samples

#### Simple RNN (Third Place)
- **Excellent Performance:** 93.3% accuracy, 94.6% AUC
- **Most Efficient:** Only 8,417 parameters (smallest model)
- **Best Parameter Efficiency:** Excellent performance with minimal complexity
- **Perfect Recall:** Detected all malicious samples

#### LSTM (Fourth Place)
- **Moderate Performance:** 66.7% accuracy, 67.9% AUC
- **Confusion Matrix:**
  - 3 false positives
  - 2 false negatives
- **Training Issues:** Early convergence suggests potential overfitting or underfitting
- **Recommendation:** May benefit from hyperparameter tuning

---

## Training Configuration

### Common Hyperparameters
- **Optimizer:** Adam (learning rate: 0.001)
- **Loss Function:** Binary Crossentropy
- **Metrics:** Accuracy, Precision, Recall
- **Epochs:** 100 (with early stopping)
- **Batch Size:** 8
- **Validation Split:** 20% of training data
- **Dropout Rate:** 30%
- **Early Stopping Patience:** 15 epochs

### Training Time & Efficiency
- **GRU:** 23 epochs (early stopped at epoch 23)
- **Bidirectional LSTM:** 25 epochs (early stopped at epoch 25)
- **Simple RNN:** 88 epochs (early stopped at epoch 88)
- **LSTM:** 16 epochs (early stopped at epoch 16)

---

## Key Insights & Recommendations

### 1. Model Selection
**Recommended Model: GRU**
- Perfect detection rate with zero false positives/negatives
- Moderate model size (23,361 parameters)
- Fast training convergence
- Robust generalization to test data

### 2. Feature Engineering Effectiveness
The 33 behavioral and system-level features proved highly effective for fileless malware detection:
- Process-level indicators (handles, DLLs, modules)
- Network behavior (connections, DNS, HTTP requests)
- Registry activity patterns
- Memory characteristics

### 3. Model Complexity Trade-offs
- **Simple RNN** demonstrated excellent parameter efficiency
- **Bidirectional LSTM** showed strong performance but with 3x more parameters than GRU
- **Standard LSTM** underperformed, suggesting architectural or training issues

### 4. Recall Performance
All models except LSTM achieved **100% recall** (no false negatives), which is critical for malware detection where missing threats is more costly than false alarms.

---

## Deployment Considerations

### Production Recommendations

1. **Primary Model:** Deploy GRU for production use
   - Perfect detection rate
   - Fast inference time
   - Moderate resource requirements

2. **Ensemble Option:** Combine GRU + Bidirectional LSTM
   - Use voting or weighted averaging
   - May improve robustness on unseen samples

3. **Monitoring Strategy:**
   - Track false positive rate in production
   - Collect misclassified samples for model retraining
   - Monitor performance on new malware families

### Model Limitations

1. **Small Dataset:** 50 samples is limited; recommend expanding dataset
2. **Class Balance:** Nearly balanced (52% malicious, 48% benign) but may need adjustment for real-world distribution
3. **Temporal Considerations:** Malware evolves; implement continuous retraining pipeline

---

## Future Work

### Dataset Enhancement
1. **Expand Sample Size:** Target 500-1000+ samples
2. **Include More Malware Families:** Add samples from the 54 fileless malware list
3. **Temporal Analysis:** Include time-series features for behavioral analysis

### Model Improvements
1. **Attention Mechanisms:** Implement attention layers to identify critical features
2. **Hybrid Architecture:** Combine RNN with CNN for pattern recognition
3. **Transfer Learning:** Pre-train on larger malware datasets

### Feature Engineering
1. **Feature Importance Analysis:** Identify most discriminative features
2. **Dimensionality Reduction:** Apply PCA/t-SNE for visualization
3. **Additional Features:** Include API call sequences, file operations

---

## Technical Details

### Data Preprocessing
- **Normalization:** StandardScaler (z-score normalization)
- **Sequence Formatting:** Reshaped features as (samples, timesteps, features)
- **Train/Test Split:** 70/30 stratified split
- **Random Seed:** 42 (for reproducibility)

### Callbacks Used
1. **EarlyStopping:**
   - Monitor: validation loss
   - Patience: 15 epochs
   - Restore best weights

2. **ReduceLROnPlateau:**
   - Monitor: validation loss
   - Factor: 0.5
   - Patience: 7 epochs
   - Min LR: 0.00001

---

## Conclusion

The GRU-based RNN model demonstrated exceptional performance for fileless malware detection, achieving perfect classification on the test set. The combination of comprehensive behavioral features and appropriate model architecture proved highly effective for this security application.

**Key Takeaways:**
1. ✅ GRU provides optimal balance of performance and efficiency
2. ✅ Behavioral features are highly discriminative for fileless malware
3. ✅ 100% recall achieved by top 3 models (critical for security)
4. ✅ Simple RNN shows excellent parameter efficiency

**Next Steps:**
1. Deploy GRU model to production environment
2. Expand dataset with additional malware samples
3. Implement continuous monitoring and retraining pipeline
4. Conduct adversarial robustness testing

---

## Generated Artifacts

### Visualizations
- `model_comparison.png` - Performance metrics comparison
- `roc_curves_comparison.png` - ROC curves for all models
- `confusion_matrix_*.png` - Confusion matrices (4 models)
- `training_history_*.png` - Training history plots (4 models)

### Data Files
- `model_comparison.csv` - Tabular performance metrics
- `rnn_fileless_malware_detection.py` - Complete source code

---

**Report Generated:** November 2, 2025  
**Framework:** TensorFlow 2.x with Keras  
**Python Version:** 3.x  
**Environment:** CPU-based training (CUDA not available)
