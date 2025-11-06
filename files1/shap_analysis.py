#!/usr/bin/env python3
"""
SHAP Feature Importance Analysis for RNN Malware Detection
===========================================================
Interprets which Volatility memory features are most important for detection

This shows which behavioral indicators from memory forensics are most
discriminative for identifying fileless malware.

Critical for research papers to show:
- Which memory artifacts matter most
- How the model makes decisions
- Validation of feature engineering choices
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import tensorflow as tf
from tensorflow import keras

class SHAPAnalysis:
    """SHAP-based feature importance for RNN models"""
    
    def __init__(self, model, X_train, X_test, feature_names, output_dir='./research_outputs/shap'):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.output_dir = output_dir
        
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def compute_shap_values(self, background_samples=100, test_samples=50):
        """
        Compute SHAP values showing feature importance
        
        Args:
            background_samples: Number of training samples to use as background
            test_samples: Number of test samples to explain
        """
        print("\n" + "="*80)
        print("COMPUTING SHAP VALUES FOR FEATURE IMPORTANCE")
        print("="*80)
        
        # Select background data
        background = self.X_train[:background_samples]
        test_data = self.X_test[:test_samples]
        
        print(f"\n✓ Using {background_samples} background samples")
        print(f"✓ Explaining {test_samples} test samples")
        print("✓ This may take a few minutes...")
        
        # Create explainer
        explainer = shap.DeepExplainer(self.model, background)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(test_data)
        
        print("✓ SHAP values computed successfully")
        
        return shap_values, test_data
    
    def plot_feature_importance(self, shap_values, test_data, top_n=15):
        """
        Plot top N most important features
        
        Shows which Volatility plugins/features are most critical
        """
        # Reshape SHAP values for plotting
        # From (samples, timesteps, 1) to (samples, timesteps)
        shap_2d = shap_values[0].reshape(shap_values[0].shape[0], -1)
        test_2d = test_data.reshape(test_data.shape[0], -1)
        
        # Summary plot
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_2d, 
            test_2d,
            feature_names=self.feature_names,
            max_display=top_n,
            show=False
        )
        plt.title('Top Features by SHAP Importance\n(Volatility Memory Artifacts)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('SHAP Value (Impact on Model Output)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: shap_summary.png")
        
        # Bar plot of mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_2d), axis=0)
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': mean_abs_shap
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 10))
        top_features = importance_df.head(top_n)
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_features)))
        
        bars = plt.barh(range(len(top_features)), top_features['Importance'], color=colors)
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Most Important Volatility Features', 
                 fontsize=14, fontweight='bold', pad=15)
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_features['Importance'])):
            plt.text(val, i, f' {val:.4f}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/shap_importance_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: shap_importance_bar.png")
        
        # Save importance table
        importance_df.to_csv(f'{self.output_dir}/feature_importance.csv', index=False)
        print(f"✓ Saved: feature_importance.csv")
        
        return importance_df
    
    def plot_dependence(self, shap_values, test_data, feature_idx, interaction_idx=None):
        """
        Plot SHAP dependence for a specific feature
        Shows how feature values affect predictions
        """
        shap_2d = shap_values[0].reshape(shap_values[0].shape[0], -1)
        test_2d = test_data.reshape(test_data.shape[0], -1)
        
        plt.figure(figsize=(10, 8))
        shap.dependence_plot(
            feature_idx,
            shap_2d,
            test_2d,
            feature_names=self.feature_names,
            interaction_index=interaction_idx,
            show=False
        )
        plt.title(f'SHAP Dependence: {self.feature_names[feature_idx]}',
                 fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/shap_dependence_{feature_idx}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: shap_dependence_{feature_idx}.png")
    
    def generate_interpretation_report(self, importance_df, top_n=10):
        """
        Generate textual interpretation of results
        Perfect for research paper discussion section
        """
        report = []
        report.append("="*80)
        report.append("SHAP FEATURE IMPORTANCE INTERPRETATION")
        report.append("="*80)
        report.append("\nMost Discriminative Volatility Features for Malware Detection:")
        report.append("")
        
        top_features = importance_df.head(top_n)
        
        for idx, (_, row) in enumerate(top_features.iterrows(), 1):
            feature = row['Feature']
            importance = row['Importance']
            
            # Categorize features
            category = self._categorize_feature(feature)
            
            report.append(f"{idx}. {feature}")
            report.append(f"   Importance: {importance:.4f}")
            report.append(f"   Category: {category}")
            report.append(f"   Interpretation: {self._interpret_feature(feature, category)}")
            report.append("")
        
        report.append("\n" + "="*80)
        report.append("KEY INSIGHTS FOR RESEARCH PAPER")
        report.append("="*80)
        report.append("")
        report.append(self._generate_insights(top_features))
        
        report_text = "\n".join(report)
        
        with open(f'{self.output_dir}/interpretation_report.txt', 'w') as f:
            f.write(report_text)
        
        print("\n" + report_text)
        print(f"\n✓ Saved: interpretation_report.txt")
        
        return report_text
    
    def _categorize_feature(self, feature):
        """Categorize Volatility features"""
        feature_lower = feature.lower()
        
        if any(x in feature_lower for x in ['dll', 'ldrmodules']):
            return "DLL Analysis"
        elif any(x in feature_lower for x in ['handle', 'process', 'pslist', 'psxview']):
            return "Process Information"
        elif any(x in feature_lower for x in ['tcp', 'udp', 'connection', 'dns', 'http']):
            return "Network Activity"
        elif any(x in feature_lower for x in ['registry', 'reg_', 'read_events', 'write_events']):
            return "Registry Operations"
        elif any(x in feature_lower for x in ['thread', 'mutex', 'callback']):
            return "Memory Artifacts"
        elif any(x in feature_lower for x in ['service', 'svcscan']):
            return "System Services"
        else:
            return "Other"
    
    def _interpret_feature(self, feature, category):
        """Provide interpretation for feature"""
        interpretations = {
            "DLL Analysis": "Malware often loads unusual DLLs or hides DLL information",
            "Process Information": "Process hiding and manipulation is common in fileless attacks",
            "Network Activity": "C2 communication and data exfiltration indicators",
            "Registry Operations": "Persistence mechanisms and configuration storage",
            "Memory Artifacts": "Threading and synchronization patterns differ in malware",
            "System Services": "Service injection and manipulation for persistence"
        }
        return interpretations.get(category, "Behavioral indicator in memory")
    
    def _generate_insights(self, top_features):
        """Generate research insights"""
        categories = top_features.apply(
            lambda row: self._categorize_feature(row['Feature']), 
            axis=1
        ).value_counts()
        
        insights = []
        insights.append("1. FEATURE CATEGORY DISTRIBUTION:")
        for cat, count in categories.items():
            insights.append(f"   - {cat}: {count} features in top 10")
        
        insights.append("\n2. RESEARCH IMPLICATIONS:")
        if "DLL Analysis" in categories.index[:2]:
            insights.append("   - DLL-related features are highly discriminative")
            insights.append("   - Volatility's ldrmodules plugin is critical for detection")
        
        if "Network Activity" in categories.index[:3]:
            insights.append("   - Network indicators remain important despite 'fileless' nature")
            insights.append("   - C2 communication patterns are detectable in memory")
        
        if "Process Information" in categories.index[:3]:
            insights.append("   - Process hiding techniques are key malware indicators")
            insights.append("   - psxview and pslist plugins provide valuable signals")
        
        insights.append("\n3. RECOMMENDED FOR PAPER:")
        insights.append("   - Focus on the dominance of " + str(categories.index[0]))
        insights.append("   - Discuss why these features are theoretically sound")
        insights.append("   - Compare with static analysis features")
        insights.append("   - Validate against known malware techniques")
        
        return "\n".join(insights)


def run_shap_analysis_for_model(model_path, X_train, X_test, feature_names, 
                                output_dir='./research_outputs/shap'):
    
    
    """
    #Main function to run SHAP analysis
    
    Usage:
        from shap_analysis import run_shap_analysis_for_model
        
        run_shap_analysis_for_model(
            model_path='./research_outputs/models/gru_model.keras',
            X_train=X_train,
            X_test=X_test,
            feature_names=feature_names
        )
    
    
    """
    print("\n" + "#"*80)
    print("SHAP FEATURE IMPORTANCE ANALYSIS")
    print("#"*80)
    
    # Load model
    print(f"\n✓ Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    
    # Create analyzer
    analyzer = SHAPAnalysis(model, X_train, X_test, feature_names, output_dir)
    
    # Compute SHAP values
    shap_values, test_data = analyzer.compute_shap_values(
        background_samples=100,
        test_samples=50
    )
    
    # Generate visualizations
    importance_df = analyzer.plot_feature_importance(shap_values, test_data, top_n=15)
    
    # Plot dependence for top feature
    top_feature_idx = importance_df.index[0]
    analyzer.plot_dependence(shap_values, test_data, top_feature_idx)
    
    # Generate interpretation
    analyzer.generate_interpretation_report(importance_df, top_n=10)
    
    print("\n" + "="*80)
    print("SHAP ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    
    return analyzer, importance_df


if __name__ == "__main__":
    print("This module provides SHAP analysis for RNN models.")
    print("Import and use run_shap_analysis_for_model() function.")
    print("\nExample:")
    print("  from shap_analysis import run_shap_analysis_for_model")
    print("  run_shap_analysis_for_model(model_path, X_train, X_test, feature_names)")
