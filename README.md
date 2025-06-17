# Sepsis Prediction using Enhanced Grey Wolf Optimization (E-GWO)

This repository contains the complete implementation of a machine learning pipeline for early sepsis prediction, developed as part of a research study. The solution leverages an Enhanced Grey Wolf Optimization (E-GWO) algorithm to tune Decision Tree hyperparameters for improved prediction accuracy.

## Overview

Sepsis is a life-threatening medical condition that requires early and accurate detection to improve patient outcomes. This project participates in a sepsis prediction challenge using synthetic patient data from hospitals in Uganda. The core focus of the study is on optimizing Decision Tree models using E-GWO, and benchmarking against other metaheuristic algorithms such as:

- Genetic Algorithm (GA)
- Particle Swarm Optimization (PSO)
- Ant Colony Optimization (ACO)

## Getting Started

### 1. Clone the repository
     git clone (https://github.com/AkifaKhan/GeniusMinds-Sepsis_Challenge)
### 2. Run with Docker (recommended)
Ensure Docker and Docker Compose are installed.

     docker-compose up --build
     
This will spin up a container that runs the complete pipeline including data loading, training, and evaluation.

## Dataset
This project is built around the 2024 Pediatric Sepsis Data Challenge, a global competition organized to advance predictive modeling for sepsis in low-resource settings.

The aim of the challenge is to develop open-source algorithms to predict in-hospital mortality in children aged 6 to 60 months who were admitted with suspected or confirmed sepsis across six Ugandan hospitals between 2017 and 2020.

The training dataset provided is synthetic, generated from a random subset of real clinical data, and is used to train and tune models.
The test dataset is non-synthetic, used for evaluation based on standard diagnostic criteria (e.g., AUC, sensitivity, specificity).
The complete dataset includes 3,837 pediatric patient records with clinical and vital signs information.
The overarching goal is to develop robust, generalizable algorithms that could improve sepsis outcomes in low- and middle-income countries (LMICs) through data-driven clinical decision support.

This repository contributes a Dockerized machine learning pipeline using Enhanced Grey Wolf Optimization (E-GWO) for model tuning, with the potential to aid in global health research and clinical implementation.

Note: Due to privacy concerns, the data is synthetic but follows the statistical characteristics of real-world clinical data.

## Performance Metrics
The model is evaluated using:
- Area Under the ROC Curve (AUC)
- Area Under the Precision-Recall Curve (AUPRC)
- Sensitivity
- Net Clinical Benefit

## Reproducibility
All code and configurations are encapsulated in a Docker container to ensure consistency across environments and ease of deployment.

## Author
Akifa Khan

Institute of Business Administration (IBA), Pakistan

Email: akifakhan001@hotmail.com
