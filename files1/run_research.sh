#!/bin/bash

#####################################################################
# Quick Start - Enhanced RNN Research System
# Trains 8 advanced models and generates complete analysis
#####################################################################

echo "======================================================================="
echo "  Enhanced RNN Research System for Fileless Malware Detection"
echo "======================================================================="
echo ""

# Check virtual environment
if [ ! -d "venv" ] && [ ! -d ".Hunter" ]; then
    echo "❌ No virtual environment found!"
    echo "Run: python3 -m venv venv && source venv/bin/activate"
    exit 1
fi

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".Hunter" ]; then
    source .Hunter/bin/activate
fi

echo "✓ Virtual environment activated"
echo ""

# Check data
if [ ! -f "data/newDataset.csv" ]; then
    echo "❌ Dataset not found: data/newDataset.csv"
    echo "Please place your dataset in the data/ directory"
    exit 1
fi

echo "✓ Dataset found"
echo ""

# Install research dependencies
echo "Installing research dependencies..."
pip install --quiet shap scipy 2>/dev/null

echo "✓ Dependencies ready"
echo ""

# Run enhanced research version
echo "======================================================================="
echo " Starting Enhanced Research Analysis"
echo " This will train 8 advanced RNN models and take 10-15 minutes"
echo "======================================================================="
echo ""

python3 rnn_research_enhanced.py

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "======================================================================="
    echo " ✓ RESEARCH ANALYSIS COMPLETE!"
    echo "======================================================================="
    echo ""
    echo "Results saved to: research_outputs/"
    echo ""
    echo "Generated:"
    echo "  📁 models/     - 8 trained models (.keras files)"
    echo "  📊 figures/    - Publication-ready visualizations"
    echo "  📋 tables/     - Results tables (CSV + LaTeX)"
    echo ""
    echo "Key outputs:"
    echo "  • comprehensive_comparison.png - Performance across all models"
    echo "  • roc_curves.png - ROC curves for all 8 models"
    echo "  • results_table.csv - Metrics for all models"
    echo "  • results_table.tex - LaTeX table for paper"
    echo ""
    echo "Next steps:"
    echo "  1. Review results: cd research_outputs && ls -R"
    echo "  2. Run SHAP analysis: python3 -c 'from shap_analysis import run_shap_analysis_for_model; ...'"
    echo "  3. See RESEARCH_GUIDE.md for paper writing tips"
    echo ""
else
    echo ""
    echo "❌ Analysis failed with exit code: $exit_code"
    echo "Check error messages above"
    exit $exit_code
fi

# Deactivate
deactivate 2>/dev/null

echo "Done! Check research_outputs/ for all results."
