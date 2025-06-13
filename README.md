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
## How to use Snakemake:
    snakemake --cores 1 --snakefile src/mlops_course_project/Snakefile

## Setup ClearML server

1. See
   [installation guide](https://clear.ml/docs/latest/docs/deploying_clearml/clearml_server_linux_mac/)

2. Run the `docker-compose` to start the server.
3. Initialize `ClearML` client (firstly, you need to install the python
   dependencies using your package manager):

```bash
clearml-init
```

## Examples

### How to start?

1. Create project:

```bash
clearml-data create --project PROJECT_NAME --name DATASET_NAME
```

2. Create your first experiment and track artifacts by:

```bash
from clearml import Task
task = Task.init(project_name="Onboarding", task_name="First Task")
task.logger.report_scalar("Demo", "Value", 0.95, 1)
task.close()
```

3. Get artifacts using:

```bash
# get instance of task that created artifact, using task ID
preprocess_task = Task.get_task(task_id='the_preprocessing_task_id')
# access artifact
local_csv = preprocess_task.artifacts['data'].get_local_copy()
```

4. Navigate to the `ClearML` web interface and see the results. By default, it
   is available on `http://localhost:8080`.

## Deploy via Triton Inference Server

### Prerequisites
- Docker and Docker Compose installed
- Models directory configured (already set up)

### Step-by-Step Deployment

1. **Start Triton Server**  
   Build and launch the service:
   ```bash
   docker-compose up --build -d
   ```

2. **Run Business Logic Client**  
   Execute the client script:
   ```bash
   python triton_server/client/bls.py
   ```

3. **Configure Model Selection**  
   Edit the configuration file to select your model:  
   `triton_server/models/topic_classification_bls/1/config.json`  
   Set `"model_name"` to either `"soft"`, `"medium"`, or `"rigid"`

4. **Measure Quality Metrics**  
   Test model accuracy and F1-score:
   ```bash
   python triton_server/client/test_performance.py
   ```

5. **Measure Efficiency Metrics**  
   Run performance analyzer for a specific model:
   ```bash
perf_analyzer -m "medium" -u "localhost:8000" —concurrency-range=1 —measurement "20000" —input-data "/client/test_data.json" —shape=INPUT:1,1 -f perf_analyze_medium_result.csv.csv —verbose-csv —collect-metrics —metrics-url "localhost:8002/metrics"
   ```

### Key Configuration Notes
- **Model Selection**: The `config.json` file controls which model version is used
- **Test Data**: `test_data.json` contains sample inputs for performance testing
- **Output Files**: Results are saved as CSV or .md files (e.g., `perf_analyze_soft_result.csv` and `perfomance_report.md`)