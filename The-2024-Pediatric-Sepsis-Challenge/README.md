# Phase 2 Submission

### This is the Phase 2 example submission. In this phase, we have made few improvements and updates over the previous submission. These changes enhance reproducibility, consistency, and robustness of our evaluation. 
---

## What's New in Phase 2

Overall major function arguments remains same, some new functionality have been added as follows.

- **Enhanced Data Loading I/O:**  
  - The `load_challenge_testdata` function accepts an optional list of selected columns. This ensures that during inference, only the necessary columns are loaded from the CSV.
  - New helper functions have been added to read selected variables either from the model dictionary or from a file (e.g., `selected_variables.txt`) in the model folder or root directory.

- **Threshold Handling Improvements:**  
  - A new mechanism is implemented to read the classification threshold from a file. If the file is missing or cannot be read, the system computes an approximate threshold from the prediction probabilities and binary predictions.

- **Parsimony Score Calculation:**  
  - The parsimony score is computed as the number of selected variables divided by the total number of available predictors (136 by default after excluding identifiers). This metric helps quantify model complexity.

- **Compute Calculation:**  
  - We now track and report compute resource usage—including memory usage and CPU time—using `psutil`. This data is recorded during inference and included in the evaluation metrics.

---


## Submission Structure

```bash
submission/
├── Dockerfile                # REQUIRED: Defines the container image and entry point.
├── requirements.txt          # REQUIRED: Lists all Python dependencies.
├── run_model.py              # Not required: Script to run inference and evaluation.
├── team_code.py              # REQUIRED: Contains your training and inference functions.
├── helper_code.py            # Not required: Contains helper functions used by your code.
├── threshold.txt             # REQUIRED: Contains the probability threshold (e.g., 0.5), either get calculated during training or model run or hard coded.  
└── model/
    ├── model.sav             # REQUIRED: Serialized trained model (includes imputer, prediction_model, etc.).
    ├── dummy_columns.txt     # OPTIONAL: List of dummy‐encoded columns (used to align test data).
    └──selected_variables.txt  # OPTIONAL: Contains the raw selected variables used for training, if not given/calculated explicitely all features with be considered as used for parsimony.
```

## File Descriptions

- **Dockerfile**  
  Defines your container image. It installs dependencies from `requirements.txt` and sets the entry point (for example, to run `run_model.py`).

- **requirements.txt**  
  Lists all necessary Python libraries (e.g., numpy, pandas, scikit-learn, joblib, mne, psutil).

- **run_model.py**  
  Loads the trained model, runs inference on test data, computes inference time and resource usage, and writes outputs including the parsimony score.

- **team_code.py**  
  Contains your team's training and inference functions. The training code now saves both the raw selected features and the dummy‑encoded columns, while the inference code uses these files for consistency.

- **helper_code.py**  
  Contains utility functions for loading data, saving outputs, and computing resource usage.

- **threshold.txt**  
  Contains the classification probability threshold (e.g., `0.5`). This value can be computed during training or hard coded.

- **selected_variables.txt**  
  Contains a list of the raw variables selected for training. This file is used during inference to load only the necessary columns and to compute the parsimony score. If this file is absent, all features will be used.

- **dummy_columns.txt**  
  Contains the list of dummy‑encoded feature names, which are used to align test data with the training features. Alternatively, these may be stored in `columns.txt` within the model folder.

- **model/**  
  - **model.sav:** The serialized model including the imputer and prediction model.
  - **dummy_columns.txt:** Required file listing the dummy‑encoded columns used during training.
  - **selected_variables.txt:** (Optional) A copy of `selected_variables.txt` for reference.

---
---

## How to Run the Phase-2 Challenge Example Code in Docker

Follow these steps to run the example code in a Docker environment: 

1. **Prepare Your Local Directory Structure:**

   Create a directory (e.g., `~/example`) with the required subfolders:

   ```bash
   mkdir -p ~/example/{mkdir training_data test_data model test_outputs}

2. **Clone or Download the Repository:**
  
    Clone the updated repository containing the example submission submission:
    ```bash
    git clone --branch Phase2 https://github.com/Kamaleswaran-Lab/The-2024-Pediatric-Sepsis-Challenge.git
    cd The-2024-Pediatric-Sepsis-Challenge
    ```
    We have provided example train_data.csv, test_data.csv and labels.csv on the above cloned example repo.
    You may consider placing them or your own CSVs in those folders for testing and debuggig. 
    - Place train_data.csv files into ~/example/training_data folder
    - Place test_data.csv files into ~/example/test_data folder
    - Place labels.csv files into ~/example/test_data folder
3. Build the Docker Image:
    
    In the root of the repository, build the Docker image using the provided Dockerfile:
    ```bash
    docker build -t image .
    ```
3. Run the Docker Container with Volume Mappings:
   
   Start the Docker container and map your local directories into the container. For example:
    ```bash
    docker run -it \
    -v ~/example/model:/challenge/model \
    -v ~/example/test_data:/challenge/test_data \
    -v ~/example/test_outputs:/challenge/test_outputs \
    -v ~/example/training_data:/challenge/training_data \
    image bash
    ```

4. Inside the Container:
  
    Once inside the container, verify that the directory structure is correct:
    ```bash  
    root@[...]:/challenge# ls
        Dockerfile             README.md         test_outputs
        evaluate_2024.py      requirements.txt  training_data
        helper_code.py         team_code.py      train_model.py
        LICENSE                run_model.py    dummy_data_split.py

    ```

    **Train Your Model:**
    Run the training script to build and save your model:
    ```bash
    python train_model.py training_data/training_data.csv model
    ```
    **Run Your Trained Model**:
    Execute the inference script to generate predictions:
    ```bash
    python run_model.py model test_data/test_data.csv test_outputs
    ```
    **Evaluate Your Model:**
    Finally, run the evaluation script to compute performance metrics (e.g., AUC, AUPRC, Sensitivity parsimony score, compute usage):
    ```bash
    python evaluate_2024.py test_data/labels.csv test_outputs/outputs.txt test_outputs/inference_time.txt threshold.txt score.json
    ```
    The evaluation output will be saved in score.json.
    
    **Exit the Container:**
    Once finished, simply type:
    ```bash
    exit
    ```
These steps ensure that you run the complete Phase 2 example code in a reproducible Docker environment. The code includes flexible thresholding, compute resource tracking, and parsimony score calculation.
