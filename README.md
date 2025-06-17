# Sepsis Prediction using Enhanced Grey Wolf Optimization (E-GWO)

This repository contains the complete implementation of a machine learning pipeline for early sepsis prediction, developed as part of a research study. The solution leverages an Enhanced Grey Wolf Optimization (E-GWO) algorithm to tune Decision Tree hyperparameters for improved prediction accuracy.

## 🧠 Overview

Sepsis is a life-threatening medical condition that requires early and accurate detection to improve patient outcomes. This project participates in a sepsis prediction challenge using synthetic patient data from hospitals in Uganda. The core focus of the study is on optimizing Decision Tree models using E-GWO, and benchmarking against other metaheuristic algorithms such as:

- Genetic Algorithm (GA)
- Particle Swarm Optimization (PSO)
- Ant Colony Optimization (ACO)

## ⚙️ Getting Started

### 1. Clone the repository
```bash
git clone (https://github.com/AkifaKhan/GeniusMinds-Sepsis_Challenge)
2. Run with Docker (recommended)
Ensure Docker and Docker Compose are installed.
docker-compose up --build
This will spin up a container that runs the complete pipeline including data loading, training, and evaluation.

3. Run Locally (without Docker)
Create a virtual environment and install dependencies:

python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate     # For Windows

pip install -r requirements.txt
python src/team_code.py

### Dataset
The dataset used in this project is synthetic patient data provided as part of a sepsis prediction challenge. The data simulates patient records from hospitals in Uganda and includes features such as vital signs, lab values, and clinical assessments.

Note: Due to privacy concerns, the data is synthetic but follows the statistical characteristics of real-world clinical data.

### Performance Metrics
The model is evaluated using:

Area Under the ROC Curve (AUC)

Area Under the Precision-Recall Curve (AUPRC)

Sensitivity

Net Clinical Benefit

### Reproducibility
All code and configurations are encapsulated in a Docker container to ensure consistency across environments and ease of deployment.

### Author
Akifa Khan
Institute of Business Administration (IBA), Pakistan
Email: akifakhan001@hotmail.com
