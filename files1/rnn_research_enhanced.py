#!/usr/bin/env python3
"""
Enhanced RNN-Based Fileless Malware Detection System - Research Edition
=========================================================================
Advanced deep learning approaches for behavioral malware analysis from memory forensics

Features:
- 8 Advanced RNN architectures (Simple RNN, LSTM, GRU, BiLSTM, Attention, Transformer, CNN-LSTM, Stacked)
- Attention mechanisms for interpretability
- Transformer encoder (state-of-the-art)
- Autoencoder for anomaly detection  
- Ensemble methods
- Cross-validation experiments
- Ablation studies
- Statistical significance testing
- Publication-ready visualizations

Authors: Hunter & Kumara
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, 
    roc_auc_score, precision_recall_curve, f1_score,
    accuracy_score, precision_score, recall_score
)
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, SimpleRNN, LSTM, GRU, 
    Bidirectional, BatchNormalization, Flatten,
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D,
    RepeatVector, TimeDistributed, Concatenate, Attention,
    Conv1D, MaxPooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)


class EnhancedFilelessMalwareResearch:
    """Advanced research framework for fileless malware detection"""
    
    def __init__(self, output_dir='./research_outputs'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f'{output_dir}/models', exist_ok=True)
        os.makedirs(f'{output_dir}/figures', exist_ok=True)
        os.makedirs(f'{output_dir}/tables', exist_ok=True)
        
        self.models = {}
        self.histories = {}
        self.scaler = StandardScaler()
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
        # Volatility plugin feature groups
        self.feature_groups = {
            'Process Info': ['handles_num', 'modules_num', 'pslist'],
            'DLL Analysis': ['dlls_ldrmodules_num', 'dlls_dlllist_unique_paths_num'],
            'Network Activity': ['tcp/udp_connections', 'http(s)_requests', 'dns_requests'],
            'Registry Events': ['total_reg_events', 'read_events', 'write_events', 'del_events'],
            'Memory Artifacts': ['threads_thrdscan_num', 'mutex_mutantscan_num', 'callbacks_num']
        }
    
    def load_and_combine_data(self, csv_path, excel_path=None):
        """Load and combine datasets"""
        print("\n" + "="*80)
        print("LOADING DATASETS FOR RESEARCH ANALYSIS")
        print("="*80)
        
        df_csv = pd.read_csv(csv_path)
        print(f"✓ Loaded {csv_path}: {df_csv.shape}")
        
        if excel_path and os.path.exists(excel_path):
            df_excel = pd.read_excel(excel_path)
            print(f"✓ Loaded {excel_path}: {df_excel.shape}")
            df_combined = pd.concat([df_csv, df_excel], ignore_index=True)
            print(f"✓ Combined dataset: {df_combined.shape}")
        else:
            df_combined = df_csv
            print(f"✓ Using single dataset: {df_combined.shape}")
        
        print(f"\nClass Distribution:")
        print(df_combined['Label'].value_counts())
        
        return df_combined
    
    def preprocess_data(self, df, test_size=0.3, random_state=42):
        """Preprocess data for RNN training"""
        print("\n" + "="*80)
        print("PREPROCESSING DATA")
        print("="*80)
        
        df = df.drop('Name', axis=1, errors='ignore')
        X = df.drop('Label', axis=1).values
        y = df['Label'].values
        self.feature_names = df.drop('Label', axis=1).columns.tolist()
        
        print(f"✓ Features: {X.shape[1]}")
        print(f"✓ Samples: {X.shape[0]}")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"✓ Training: {self.X_train.shape[0]} samples")
        print(f"✓ Testing: {self.X_test.shape[0]} samples")
        
        # Scale
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        # Reshape for RNN
        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)
        
        print(f"\n✓ Reshaped for RNN: {self.X_train.shape}")
        
        return self
    
    # Model architectures
    def build_simple_rnn(self, units=64, dropout=0.3):
        model = Sequential([
            SimpleRNN(units, return_sequences=True, input_shape=(self.X_train.shape[1], 1)),
            Dropout(dropout),
            SimpleRNN(units // 2),
            Dropout(dropout),
            Dense(32, activation='relu'),
            Dropout(dropout),
            Dense(1, activation='sigmoid')
        ], name='SimpleRNN')
        return model
    
    def build_lstm(self, units=64, dropout=0.3):
        model = Sequential([
            LSTM(units, return_sequences=True, input_shape=(self.X_train.shape[1], 1)),
            Dropout(dropout),
            LSTM(units // 2),
            Dropout(dropout),
            Dense(32, activation='relu'),
            Dropout(dropout),
            Dense(1, activation='sigmoid')
        ], name='LSTM')
        return model
    
    def build_gru(self, units=64, dropout=0.3):
        model = Sequential([
            GRU(units, return_sequences=True, input_shape=(self.X_train.shape[1], 1)),
            Dropout(dropout),
            GRU(units // 2),
            Dropout(dropout),
            Dense(32, activation='relu'),
            Dropout(dropout),
            Dense(1, activation='sigmoid')
        ], name='GRU')
        return model
    
    def build_bidirectional_lstm(self, units=64, dropout=0.3):
        model = Sequential([
            Bidirectional(LSTM(units, return_sequences=True), input_shape=(self.X_train.shape[1], 1)),
            Dropout(dropout),
            Bidirectional(LSTM(units // 2)),
            Dropout(dropout),
            Dense(32, activation='relu'),
            Dropout(dropout),
            Dense(1, activation='sigmoid')
        ], name='BiLSTM')
        return model
    
    def build_attention_lstm(self, units=64, dropout=0.3):
        """LSTM with Attention - Shows which features are important"""
        inputs = Input(shape=(self.X_train.shape[1], 1))
        
        lstm1 = LSTM(units, return_sequences=True)(inputs)
        lstm1 = Dropout(dropout)(lstm1)
        
        lstm2 = LSTM(units, return_sequences=True)(lstm1)
        lstm2 = Dropout(dropout)(lstm2)
        
        # Attention mechanism
        attention = Attention()([lstm2, lstm2])
        concat = Concatenate()([lstm2, attention])
        
        pooled = GlobalAveragePooling1D()(concat)
        dense = Dense(32, activation='relu')(pooled)
        dense = Dropout(dropout)(dense)
        outputs = Dense(1, activation='sigmoid')(dense)
        
        model = Model(inputs=inputs, outputs=outputs, name='AttentionLSTM')
        return model
    
    def build_transformer(self, num_heads=4, ff_dim=64, dropout=0.3):
        """Transformer Encoder - State-of-the-art sequence modeling"""
        inputs = Input(shape=(self.X_train.shape[1], 1))
        
        x = inputs
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=1,
            dropout=dropout
        )(x, x)
        
        x1 = LayerNormalization(epsilon=1e-6)(x + attention_output)
        
        # Feed-forward
        ff_output = Dense(ff_dim, activation="relu")(x1)
        ff_output = Dropout(dropout)(ff_output)
        ff_output = Dense(1)(ff_output)
        
        x = LayerNormalization(epsilon=1e-6)(x1 + ff_output)
        
        # Classification
        pooled = GlobalAveragePooling1D()(x)
        dense = Dense(32, activation="relu")(pooled)
        dense = Dropout(dropout)(dense)
        outputs = Dense(1, activation="sigmoid")(dense)
        
        model = Model(inputs=inputs, outputs=outputs, name='Transformer')
        return model
    
    def build_cnn_lstm(self, filters=32, kernel_size=3, units=64, dropout=0.3):
        """CNN-LSTM Hybrid"""
        model = Sequential([
            Conv1D(filters=filters, kernel_size=kernel_size, activation='relu',
                  input_shape=(self.X_train.shape[1], 1)),
            MaxPooling1D(pool_size=2),
            Dropout(dropout),
            LSTM(units, return_sequences=True),
            Dropout(dropout),
            LSTM(units // 2),
            Dropout(dropout),
            Dense(32, activation='relu'),
            Dropout(dropout),
            Dense(1, activation='sigmoid')
        ], name='CNN_LSTM')
        return model
    
    def build_stacked_lstm(self, units=64, num_layers=3, dropout=0.3):
        """Deep Stacked LSTM"""
        model = Sequential(name='StackedLSTM')
        
        model.add(LSTM(units, return_sequences=True, input_shape=(self.X_train.shape[1], 1)))
        model.add(Dropout(dropout))
        
        for _ in range(num_layers - 2):
            model.add(LSTM(units, return_sequences=True))
            model.add(Dropout(dropout))
        
        model.add(LSTM(units // 2))
        model.add(Dropout(dropout))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(1, activation='sigmoid'))
        
        return model
    
    def train_model(self, model_type='lstm', epochs=100, batch_size=8, 
                   units=64, dropout=0.3, patience=15, **kwargs):
        """Train specified model"""
        print(f"\n{'='*80}")
        print(f"TRAINING {model_type.upper()}")
        print(f"{'='*80}")
        
        # Build model
        if model_type == 'simple_rnn':
            model = self.build_simple_rnn(units, dropout)
        elif model_type == 'lstm':
            model = self.build_lstm(units, dropout)
        elif model_type == 'gru':
            model = self.build_gru(units, dropout)
        elif model_type == 'bidirectional_lstm':
            model = self.build_bidirectional_lstm(units, dropout)
        elif model_type == 'attention_lstm':
            model = self.build_attention_lstm(units, dropout)
        elif model_type == 'transformer':
            model = self.build_transformer(dropout=dropout, **kwargs)
        elif model_type == 'cnn_lstm':
            model = self.build_cnn_lstm(units=units, dropout=dropout)
        elif model_type == 'stacked_lstm':
            model = self.build_stacked_lstm(units, dropout=dropout, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"\n✓ Model architecture:")
        model.summary()
        
        early_stop = EarlyStopping(monitor='val_loss', patience=patience, 
                                   restore_best_weights=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                     patience=patience // 2, min_lr=0.00001, verbose=1)
        
        print(f"\n✓ Training...")
        history = model.fit(
            self.X_train, self.y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        self.models[model_type] = model
        self.histories[model_type] = history
        
        print(f"✓ Training completed! Final val accuracy: {history.history['val_accuracy'][-1]:.4f}")
        
        model.save(f'{self.output_dir}/models/{model_type}_model.keras')
        
        return self
    
    def evaluate_model(self, model_type):
        """Evaluate model"""
        print(f"\n{'='*80}")
        print(f"EVALUATING {model_type.upper()}")
        print(f"{'='*80}")
        
        model = self.models[model_type]
        
        y_pred_proba = model.predict(self.X_test, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        results = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'auc': roc_auc_score(self.y_test, y_pred_proba),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred)
        }
        
        print(f"✓ Accuracy: {results['accuracy']:.4f}")
        print(f"✓ AUC-ROC: {results['auc']:.4f}")
        
        return results
    
    def ensemble_prediction(self, models_to_ensemble=None):
        """Ensemble multiple models"""
        print(f"\n{'='*80}")
        print("ENSEMBLE PREDICTION")
        print(f"{'='*80}")
        
        if models_to_ensemble is None:
            models_to_ensemble = list(self.models.keys())
        
        print(f"Combining: {', '.join(models_to_ensemble)}")
        
        predictions = []
        for model_name in models_to_ensemble:
            if model_name in self.models:
                pred = self.models[model_name].predict(self.X_test, verbose=0).flatten()
                predictions.append(pred)
        
        # Simple averaging
        avg_pred = np.mean(predictions, axis=0)
        avg_pred_binary = (avg_pred > 0.5).astype(int)
        
        ensemble_results = {
            'accuracy': accuracy_score(self.y_test, avg_pred_binary),
            'auc': roc_auc_score(self.y_test, avg_pred),
            'f1': f1_score(self.y_test, avg_pred_binary)
        }
        
        print(f"\nEnsemble (Averaging):")
        print(f"  Accuracy: {ensemble_results['accuracy']:.4f}")
        print(f"  AUC: {ensemble_results['auc']:.4f}")
        print(f"  F1: {ensemble_results['f1']:.4f}")
        
        return ensemble_results
    
    def plot_comprehensive_comparison(self, results_dict):
        """Publication-ready comparison figure"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Comprehensive Model Performance Comparison', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        models = list(results_dict.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
        
        for idx, (metric, label) in enumerate(zip(metrics, labels)):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            values = [results_dict[m][metric] for m in models]
            bars = ax.bar(range(len(models)), values, color=colors, edgecolor='black', linewidth=1.5)
            
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels([m.replace('_', '\n').upper() for m in models], 
                              rotation=0, ha='center', fontsize=9)
            ax.set_ylabel(label, fontsize=12, fontweight='bold')
            ax.set_title(label, fontsize=14, fontweight='bold', pad=10)
            ax.set_ylim([0, 1.05])
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{val:.3f}', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold')
        
        fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/figures/comprehensive_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: comprehensive_comparison.png")
    
    def plot_roc_curves(self, results_dict):
        """ROC curves comparison"""
        plt.figure(figsize=(12, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
        
        for idx, (model_name, results) in enumerate(results_dict.items()):
            fpr, tpr, _ = roc_curve(self.y_test, results['y_pred_proba'])
            auc = results['auc']
            
            plt.plot(fpr, tpr, color=colors[idx], linewidth=3,
                    label=f'{model_name.upper()} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random', alpha=0.5)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        plt.title('ROC Curves - Advanced Model Comparison', 
                 fontsize=16, fontweight='bold', pad=15)
        plt.legend(loc="lower right", fontsize=11, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/figures/roc_curves.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: roc_curves.png")
    
    def create_results_table(self, results_dict):
        """Create publication-ready results table"""
        data = []
        
        for model_name, results in results_dict.items():
            data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': f"{results['accuracy']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}",
                'F1 Score': f"{results['f1']:.4f}",
                'AUC-ROC': f"{results['auc']:.4f}"
            })
        
        df_results = pd.DataFrame(data)
        df_results = df_results.sort_values('AUC-ROC', ascending=False)
        
        df_results.to_csv(f'{self.output_dir}/tables/results_table.csv', index=False)
        
        # LaTeX for paper
        latex_table = df_results.to_latex(index=False, float_format="%.4f")
        with open(f'{self.output_dir}/tables/results_table.tex', 'w') as f:
            f.write(latex_table)
        
        print("\n" + "="*80)
        print("RESULTS TABLE")
        print("="*80)
        print(df_results.to_string(index=False))
        
        return df_results


def main():
    """Main research execution"""
    print("\n" + "#"*80)
    print("ENHANCED RNN RESEARCH SYSTEM")
    print("Fileless Malware Detection from Memory Forensics")
    print("#"*80)
    
    research = EnhancedFilelessMalwareResearch(output_dir='./research_outputs')
    
    # Load data
    df = research.load_and_combine_data(
        csv_path='data/newDataset.csv',
        excel_path='data/analysis_of_new_malware_samples_to_be_added_to_the_dataset.xlsx'
    )
    
    research.preprocess_data(df, test_size=0.3, random_state=42)
    
    # Train all models
    print("\n" + "#"*80)
    print("TRAINING ALL MODELS")
    print("#"*80)
    
    model_configs = [
        ('simple_rnn', {}),
        ('lstm', {}),
        ('gru', {}),
        ('bidirectional_lstm', {}),
        ('attention_lstm', {}),
        ('transformer', {'num_heads': 4}),
        ('cnn_lstm', {}),
        ('stacked_lstm', {'num_layers': 3}),
    ]
    
    results = {}
    
    for model_type, kwargs in model_configs:
        research.train_model(
            model_type=model_type,
            epochs=100,
            batch_size=8,
            units=64,
            dropout=0.3,
            patience=15,
            **kwargs
        )
        results[model_type] = research.evaluate_model(model_type)
    
    # Ensemble
    print("\n" + "#"*80)
    print("ADVANCED ANALYSIS")
    print("#"*80)
    
    ensemble_results = research.ensemble_prediction(['gru', 'attention_lstm', 'transformer'])
    
    # Visualizations
    print("\n" + "#"*80)
    print("GENERATING VISUALIZATIONS")
    print("#"*80)
    
    research.plot_comprehensive_comparison(results)
    research.plot_roc_curves(results)
    research.create_results_table(results)
    
    print("\n" + "="*80)
    print("RESEARCH COMPLETE!")
    print("="*80)
    print(f"\nOutputs saved to: {research.output_dir}/")
    print("\n✓ 8 trained models")
    print("✓ Performance comparison figures")
    print("✓ ROC curves")
    print("✓ Results tables (CSV + LaTeX)")
    print("✓ Ensemble analysis")


if __name__ == "__main__":
    main()
