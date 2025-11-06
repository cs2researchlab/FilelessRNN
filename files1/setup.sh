#!/bin/bash

################################################################################
# RNN Fileless Malware Detection - Ubuntu Setup Script
# Automated installation and configuration
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check if running on Ubuntu
check_os() {
    print_header "Checking Operating System"
    
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        if [[ "$ID" == "ubuntu" ]]; then
            print_success "Running on Ubuntu $VERSION"
        else
            print_warning "Not running on Ubuntu. This script is optimized for Ubuntu but may work."
        fi
    else
        print_warning "Cannot detect OS. Proceeding anyway..."
    fi
}

# Check Python version
check_python() {
    print_header "Checking Python Installation"
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        print_success "Python $PYTHON_VERSION found"
        
        if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_success "Python version is compatible (3.8+)"
        else
            print_error "Python 3.8 or higher required. Found: $PYTHON_VERSION"
            print_info "Install Python 3.8+ with: sudo apt install python3.8"
            exit 1
        fi
    else
        print_error "Python 3 not found"
        print_info "Install with: sudo apt update && sudo apt install python3 python3-pip python3-venv"
        exit 1
    fi
}

# Check pip
check_pip() {
    print_header "Checking pip Installation"
    
    if command -v pip3 &> /dev/null; then
        PIP_VERSION=$(pip3 --version | cut -d' ' -f2)
        print_success "pip $PIP_VERSION found"
    else
        print_warning "pip not found. Installing..."
        sudo apt update
        sudo apt install -y python3-pip
        print_success "pip installed"
    fi
}

# Create virtual environment
setup_venv() {
    print_header "Setting Up Virtual Environment"
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists"
        read -p "Remove and recreate? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf venv
            print_info "Removed existing virtual environment"
        else
            print_info "Using existing virtual environment"
            return
        fi
    fi
    
    print_info "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
}

# Install dependencies
install_dependencies() {
    print_header "Installing Python Dependencies"
    
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found!"
        exit 1
    fi
    
    print_info "Activating virtual environment..."
    source venv/bin/activate
    
    print_info "Upgrading pip..."
    pip install --upgrade pip --quiet
    
    print_info "Installing dependencies (this may take a few minutes)..."
    pip install -r requirements.txt --quiet
    
    print_success "All dependencies installed"
    
    deactivate
}

# Create directory structure
setup_directories() {
    print_header "Creating Directory Structure"
    
    mkdir -p data
    print_success "Created: data/"
    
    mkdir -p outputs
    print_success "Created: outputs/"
    
    mkdir -p logs
    print_success "Created: logs/"
}

# Verify installation
verify_installation() {
    print_header "Verifying Installation"
    
    source venv/bin/activate
    
    # Check TensorFlow
    print_info "Checking TensorFlow..."
    python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)" 2>&1
    if [ $? -eq 0 ]; then
        print_success "TensorFlow working"
    else
        print_error "TensorFlow check failed"
    fi
    
    # Check other key packages
    print_info "Checking other packages..."
    python3 -c "import pandas, numpy, sklearn, matplotlib, seaborn" 2>&1
    if [ $? -eq 0 ]; then
        print_success "All core packages working"
    else
        print_error "Some packages failed to import"
    fi
    
    # Check GPU availability
    print_info "Checking GPU availability..."
    GPU_CHECK=$(python3 -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))" 2>&1)
    if [ "$GPU_CHECK" -gt 0 ]; then
        print_success "GPU detected and available for TensorFlow"
    else
        print_warning "No GPU detected. Training will use CPU (slower but functional)"
    fi
    
    deactivate
}

# Create example data file
create_example_data() {
    print_header "Creating Example Data Template"
    
    if [ ! -f "data/example_template.csv" ]; then
        cat > data/example_template.csv << 'EOF'
Name,Label,handles_num,hiveList,dlls_ldrmodules_num,dlls_ldrmodules_unique_mappedpaths_num,dlls_ldrmodules_InInit_fales_num,dlls_ldrmodules_InLoad_false_num,dlls_ldrmodules_InMem_False_num,dlls_ldrmodules_all_false_num,modules_num,callbacks_num,processes_privs_enabled_not_default_num,processes_psxview_exited_num,processes_psxview_false_columns_num,processes_psxview_false_rows_num,processes_psxview_num,processes_psxview_pslist_true_num,processes_psxview_psscan_true_num,services_svcscan_num,services_svcscan_running_num,services_svcscan_stopped_num,dlls_dlllist_unique_paths_num,mutex_mutantscan_num,threads_thrdscan_num,pslist,tcp/udp_connections,total_reg_events,read_events,write_events,del_events,executable_files,unknown_types,http(s)_requests,dns_requests
Sample1,1,100,568,2635,467,130,78,78,78,156,10,7,5,60,60,60,0,393,139,254,15453,446,791,25,24,1,0,0,0,0,0,0,12,44
Sample2,0,95,450,1800,380,110,65,65,65,140,8,5,8,70,70,70,0,350,120,220,12000,380,650,30,35,15,5,2,3,1,2,1,18,52
EOF
        print_success "Created example data template: data/example_template.csv"
    else
        print_info "Example template already exists"
    fi
}

# Create run script
create_run_script() {
    print_header "Creating Run Script"
    
    cat > run.sh << 'EOF'
#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Check if data exists
if [ ! -f "data/newDataset.csv" ]; then
    echo "Error: data/newDataset.csv not found!"
    echo "Please place your dataset in the data/ directory"
    exit 1
fi

# Run the analysis
echo "Starting RNN Malware Detection Analysis..."
python3 rnn_fileless_malware_detection.py

# Deactivate virtual environment
deactivate

echo ""
echo "Analysis complete! Check the outputs/ directory for results."
EOF
    
    chmod +x run.sh
    print_success "Created run script: ./run.sh"
}

# Print next steps
print_next_steps() {
    print_header "Installation Complete!"
    
    echo -e "${GREEN}Setup successful!${NC}\n"
    
    echo "📁 Directory structure:"
    echo "   ├── data/          - Place your datasets here"
    echo "   ├── outputs/       - Generated results will appear here"
    echo "   ├── logs/          - Log files"
    echo "   └── venv/          - Python virtual environment"
    echo ""
    
    echo "📊 Next steps:"
    echo ""
    echo "1. Place your datasets in the data/ directory:"
    echo "   ${YELLOW}cp /path/to/newDataset.csv data/${NC}"
    echo "   ${YELLOW}cp /path/to/new_malware_samples.xlsx data/${NC}"
    echo ""
    echo "2. Activate the virtual environment:"
    echo "   ${YELLOW}source venv/bin/activate${NC}"
    echo ""
    echo "3. Run the analysis:"
    echo "   ${YELLOW}python3 rnn_fileless_malware_detection.py${NC}"
    echo ""
    echo "   Or use the convenience script:"
    echo "   ${YELLOW}./run.sh${NC}"
    echo ""
    echo "4. Check results in outputs/ directory"
    echo ""
    echo "📖 For detailed usage instructions, see README.md"
    echo ""
    
    print_success "Ready to use!"
}

# Main execution
main() {
    clear
    echo -e "${BLUE}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════╗
║   RNN Fileless Malware Detection System - Setup      ║
║                Ubuntu Installation                    ║
╚═══════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    
    check_os
    check_python
    check_pip
    setup_venv
    install_dependencies
    setup_directories
    verify_installation
    create_example_data
    create_run_script
    print_next_steps
}

# Run main function
main
