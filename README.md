# MLOps course project

This repository contains the ML experiments code for the news topic classification task.

## Development Workflow:

GitHub flow:
ML experiments will be conducted in separate branches. Once the research is completed, the report and source code will be merged into the main branch.

## Project Setup Guide 🛠️

This project uses **Rye** (modern Python manager) and **uv** (ultra-fast installer) for dependency management. Follow these steps to get started:

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ installed
- Curl available in terminal

### 1. Install Rye
```bash
# Install Rye manager
curl -sSf https://rye-up.com/get | bash

# Add to PATH
echo 'source "$HOME/.rye/env"' >> ~/.bashrc  # or ~/.zshrc for Zsh
source ~/.bashrc # or restart terminal instead
```

### 2. Clone & Setup
    git clone https://github.com/leksminure/itmo_mlops_course.git
    cd itmo_mlops_course

#### Install all dependencies (including dev tools)
    rye sync

### 3. Activate Environment
    source .venv/bin/activate

## Git LFS setup

**1. Install & Setup**  
```bash
# Install
brew install git-lfs       # macOS
sudo apt-get install git-lfs  # Ubuntu
git lfs install            # Initialize
```
**2. Track Files**  
```bash
git lfs track "*.pt" "*.h5" "data/**" "models/**"  # ML files/dirs
git add .gitattributes && git commit -m "Add LFS tracking"
```
**3. Standard Workflow**  
```bash
git add large_file.pt   # LFS auto-handles tracked files
git commit -m "Add model"
git push
```
