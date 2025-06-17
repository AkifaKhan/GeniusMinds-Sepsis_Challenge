#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, os, sys
import pandas as pd
from sklearn.impute import SimpleImputer
import joblib
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.special import gamma
from scipy.stats import levy


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################


# === Enhanced GWO Optimizer (from earlier) ===
# === GWO with Lévy ===
class GreyWolfOptimizerWithLevy:
    def __init__(self, lower_bounds, upper_bounds, fitness_function,
                 population_size=10, max_iterations=100, early_stopping_rounds=10):
        self.lower_bounds = np.array(lower_bounds)
        self.upper_bounds = np.array(upper_bounds)
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.early_stopping_rounds = early_stopping_rounds
        self.dimension = len(lower_bounds)
        self.positions = np.random.uniform(self.lower_bounds, self.upper_bounds,
                                           (self.population_size, self.dimension))

        # Integer indices: [max_depth, min_samples_split, min_samples_leaf]
        self.int_indices = [0, 1, 2]

    def levy_flight(self, size):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        return u / np.abs(v) ** (1 / beta)

    def optimize(self):
        best_fitness = np.inf
        best_position = None
        patience_counter = 0

        for iteration in range(self.max_iterations):
            fitness_values = np.array([self.fitness_function(self._apply_constraints(p)) for p in self.positions])
            sorted_indices = np.argsort(fitness_values)
            fitness_values = fitness_values[sorted_indices]
            self.positions = self.positions[sorted_indices]
            self.alpha = self.positions[0]
            self.beta = self.positions[1]
            self.delta = self.positions[2]

            if fitness_values[0] < best_fitness:
                best_fitness = fitness_values[0]
                best_position = self.alpha.copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_rounds:
                print(f"Early stopping at iteration {iteration}")
                break

            a = 2 - iteration * (2 / self.max_iterations)

            for i in range(self.population_size):
                for j in range(self.dimension):
                    r1, r2 = np.random.random(), np.random.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha[j] - self.positions[i][j])
                    X1 = self.alpha[j] - A1 * D_alpha

                    r1, r2 = np.random.random(), np.random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta[j] - self.positions[i][j])
                    X2 = self.beta[j] - A2 * D_beta

                    r1, r2 = np.random.random(), np.random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta[j] - self.positions[i][j])
                    X3 = self.delta[j] - A3 * D_delta

                    new_value = (X1 + X2 + X3) / 3

                    if iteration < self.max_iterations * 0.5:
                        levy_step = self.levy_flight(1)[0] * (1.0 / (iteration + 1) ** 1.5)
                        new_value += levy_step

                    new_value = np.clip(new_value, self.lower_bounds[j], self.upper_bounds[j])
                    self.positions[i][j] = new_value

        return self._apply_constraints(best_position), best_fitness

    def _apply_constraints(self, position):
        constrained = position.copy()
        for idx in self.int_indices:
            constrained[idx] = int(round(constrained[idx]))
        return constrained


# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find the Challenge data.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')
        
    patient_ids, data, label, features = load_challenge_data(data_folder)
    num_patients = len(patient_ids)

    if num_patients == 0:
        raise FileNotFoundError('No data is provided.')
        
    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    
    # Train the models.
    if verbose >= 1:
        print('Training the Challenge models on the Challenge data...')


    # ================================
    # Feature selection (column dropping)
    # ================================
    # For example, the team may select a subset of variables. Put your feature selection code if needed. Here, we simply use all raw columns.

    selected_variables = ['agecalc_adm','height_cm_adm','hr_bpm_adm','glucose_mmolpl_adm','diasbp_mmhg_adm',
    'weight_kg_adm','bcseye_adm','sysbp_mmhg_adm','rr_brpm_app_adm','lactate_mmolpl_adm',
    'hematocrit_gpdl_adm','bcsverbal_adm','temp_c_adm','spo2site2_pc_oxi_adm',
    'exclbreastfed_adm','feedingstatus_adm','watersource_adm','deadchildren_adm',
    'spo2site1_pc_oxi_adm','deliveryloc_adm','priorweekantimal_adm','malariastatuspos_adm',
    'vaccdpt_adm','waterpure_adm','sex_adm','sqi2_perc_oxi_adm','respdistress_adm','muac_mm_adm',
    'momage_adm','birthdetail_adm___5','birthdetail_adm___4','vaccpneumoc_adm','birthdetail_adm___1',
    'birthdetail_adm___2','birthdetail_adm___3','sqi1_perc_oxi_adm','oxygenavail_adm',
    'symptoms_adm___3','priorweekabx_adm','bcsmotor_adm','bcgscar_adm','symptoms_adm___9']
    

    data = data[selected_variables]

    # Save the selected features to file.
    with open(os.path.join(model_folder, 'selected_variables.txt'), 'w') as f:
        f.write("\n".join(selected_variables))

    # ================================
    # Preprocessing: dummy encoding     
    # # ================================
    data = pd.get_dummies(data)
    dummy_columns = list(data.columns)
    #Saving the dummy-encoded column names for later alignment during inference.
    with open(os.path.join(model_folder, 'dummy_columns.txt'), 'w') as f:
        f.write("\n".join(dummy_columns))

    # Impute any missing features; use the mean value by default.
    imputer = SimpleImputer().fit(data)

    # Train the models.
    data_imputed = imputer.transform(data)
  # ================================
    # Oversampling
    # ================================

# Step 1: Show original class distribution
    if verbose >= 1:
        print('Original class distribution:')
        print(pd.Series(label.ravel()).value_counts())

# Step 2: Apply Borderline-SMOTE with sampling_strategy=0.3
    bl_smote = BorderlineSMOTE(sampling_strategy=0.25, random_state=42)
    X_resampled, y_resampled = bl_smote.fit_resample(data_imputed, label.ravel())

# Step 3: Show new class distribution
    if verbose >= 1:
        print('After Borderline-SMOTE class distribution (25% minority):')
        print(pd.Series(y_resampled).value_counts())

# TRAIN MODEL

    # === FITNESS FUNCTION ===
    def fitness_function_dt(params):
        max_depth = int(params[0])
        min_samples_split = int(params[1])
        min_samples_leaf = int(params[2])
        max_features = float(params[3])

        dt = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42
        )

        scores = cross_val_score(dt, X_resampled, y_resampled, cv=5, scoring='accuracy')
        return -np.mean(scores)  # Minimize negative accuracy


# === HYPERPARAMETER BOUNDS ===
    lower_bounds = [3, 2, 1, 0.1]   # max_depth, min_samples_split, min_samples_leaf, max_features
    upper_bounds = [50, 20, 10, 0.9]

# === RUN GWO OPTIMIZATION ===
    gwo = GreyWolfOptimizerWithLevy(
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        fitness_function=fitness_function_dt,
        population_size=30,
        max_iterations=50,
        early_stopping_rounds=10
    )

    best_params_dt, best_fitness = gwo.optimize()
    print("Best Hyperparameters found by GWO:", best_params_dt)


# === TRAIN FINAL MODEL ===
    best_dt = DecisionTreeClassifier(
        max_depth=int(best_params_dt[0]),
        min_samples_split=int(best_params_dt[1]),
        min_samples_leaf=int(best_params_dt[2]),
        max_features=float(best_params_dt[3]),
        class_weight='balanced',
        random_state=42
    )

    best_dt.fit(X_resampled, y_resampled)

    print("\nTrained Decision Tree Hyperparameters:")
    print(best_dt.get_params())

# === Save model ===
    save_challenge_model(model_folder, imputer, best_dt, selected_variables, dummy_columns)

    if verbose >= 1:
        print('\n✅ Optimized Decision Tree Model Training complete!')
        
# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    if verbose >= 1:
        print('Loading the model...')
    # Attempt to load the selected features from 'selected_variables.txt'
    try:
        with open(os.path.join(model_folder, 'selected_variables.txt'), 'r') as f:
            selected_variables = f.read().splitlines()
        if verbose:
            print("Loaded selected features from 'selected_variables.txt'")
    except Exception as e:
        if verbose:
            print("Warning: Could not load 'selected_variables.txt'. Using all features. Error:", e)
        selected_variables = None

    # Load the dummy-encoded columns (used during training)
    try:
        with open(os.path.join(model_folder, 'dummy_columns.txt'), 'r') as f:
            dummy_columns = f.read().splitlines()
    except Exception as e:
        if verbose:
            print("Warning: Could not load 'dummy_columns.txt'.", e)
        dummy_columns = None

    # Load the saved model.
    model = joblib.load(os.path.join(model_folder, 'model.sav'))
    # Save the selected features into the model dictionary under a standardized key.
    model['selected_variables'] = selected_variables
    model['columns'] = dummy_columns
    return model


def run_challenge_model(model, data_folder, verbose):
    imputer = model['imputer']
    prediction_model = model['prediction_model']
    dummy_columns = model['dummy_columns']
    selected_variable = model['selected_variables']
    
    
    # Load test data. If selected_variables is None, all columns are loaded.
    patient_ids, data, _ = load_challenge_testdata(data_folder, selected_variable)
    
    # Preprocess: apply dummy encoding and align with training dummy columns.
    data = pd.get_dummies(data)
    data = data.reindex(columns=dummy_columns, fill_value=0)
    
    # Impute missing data.
    data_imputed = imputer.transform(data)
    
    # Get prediction probabilities.
    prediction_probability = prediction_model.predict_proba(data_imputed)[:, 1]
    
    # Set a probability threshold (adjust or calculate as needed).
    threshold = 0.08
    
    # Compute binary predictions using the threshold.
    prediction_binary = (prediction_probability >= threshold).astype(int)
    
    # Write the threshold to a file called "threshold.txt".
    with open("threshold.txt", "w") as f:
        f.write(str(threshold))
    
    return patient_ids, prediction_binary, prediction_probability

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model along with the imputer, selected features, and dummy columns.
def save_challenge_model(model_folder, imputer, prediction_model, selected_variables, dummy_columns):
    d = {
        'imputer': imputer,
        'prediction_model': prediction_model,
        'selected_variables': selected_variables,
        'dummy_columns': dummy_columns,
    }
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)

