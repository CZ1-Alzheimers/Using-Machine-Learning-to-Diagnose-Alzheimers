import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from sklearn.metrics import roc_curve, auc as auc_calc

from sklearn.model_selection import KFold, GridSearchCV, GroupKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, f1_score, recall_score, make_scorer
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping

import joblib

matplotlib.use('Agg')

tf.random.set_seed(42)
np.random.seed(42)


# Load training data
def load_data():

    # Training CSV
    filename = ".venv/Scripts/Datasets/ADNI-Oasis_dataset.csv"  # Placeholder/default training data
    print(f"Loading training data from: {filename}")

    # Load dataset
    df = pd.read_csv(filename)

    # Name results path
    if "Oasis" in filename and "ADNI" not in filename and "AIBL" not in filename:
        data_source = "Oasis"
    elif "ADNI" in filename and "Oasis" not in filename and "AIBL" not in filename:
        data_source = "ADNI"
    elif "AIBL" in filename and "ADNI" not in filename and "Oasis" not in filename:
        data_source = "AIBL"
    elif "Oasis" in filename and "ADNI" in filename and "AIBL" not in filename:
        data_source = "ADNI-Oasis"
    elif "ADNI" in filename and "AIBL" in filename and "Oasis" not in filename:
        data_source = "ADNI-AIBL"
    elif "Oasis" in filename and "AIBL" in filename and "ADNI" not in filename:
        data_source = "Oasis-AIBL"
    else:
        data_source = "Complete"

    return df, data_source, filename

# Define models
def get_models_and_params(code_a, code_b):

    # Logistic Regression (NC vs MCI)
    # lr_config = {
    #     "LR (ElasticNet)": (
    #     LogisticRegression(solver='saga', max_iter=3000, random_state=42, class_weight='balanced', tol=1e-3),
    #     {'C': [0.1, 1, 10], 'penalty': ['elasticnet'], 'l1_ratio': [0.3, 0.5, 0.7]})
    # }

    # SVM (RBF) (NC vs AD, MCI vs AD)
    svm_config = {
        "SVM (RBF)": (SVC(probability=True, random_state=42, tol=1e-3, class_weight='balanced'),
                      {'C': [0.1, 1, 10, 100], 'kernel': ['rbf'], 'gamma': ['scale', 0.001, 0.01, 0.1]})
    }

    comparison = (code_a, code_b)

    # # Return model configuration
    # if comparison == (1, 2):  # NC (1) vs MCI (2)
    #     return lr_config
    # elif comparison == (1, 3) or comparison == (2, 3):  # NC (1) vs AD (3) or MCI (2) vs AD (3)
    #     return svm_config
    # else:
    #     print(f"Warning: No model configuration defined for comparison {comparison}. Returning empty.")
    #     return {}

    return svm_config


# Calculate CI 95
def calculate_ci_95(data_series):
    n = len(data_series)
    if n < 2:
        return np.nan, np.nan

    mean = data_series.mean()
    std_err = data_series.std(ddof=1) / np.sqrt(n)
    margin = stats.t.ppf(0.975, df=n - 1) * std_err

    return mean - margin, mean + margin

# Plot ROC curves
def plot_mean_roc_curves(model_roc_data, results_path, file_suffix):
    plt.figure(figsize=(12, 10))
    base_fpr = np.linspace(0, 1, 101)

    for name, fold_data in model_roc_data.items():
        tprs = []
        aucs = []

        if not fold_data:
            print(f"Warning: No ROC data for model {name}. Skipping plot.")
            continue

        for y_test, y_proba in fold_data:
            if len(np.unique(y_test)) < 2:
                continue

            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc_calc(fpr, tpr)

            tpr_interp = np.interp(base_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)
            aucs.append(roc_auc)

        if not tprs:
            print(f"Warning: No valid folds for model {name}. Skipping plot.")
            continue

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc_calc(base_fpr, mean_tpr)
        std_auc = np.std(aucs)

        std_tpr = np.std(tprs, axis=0)
        tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
        tpr_lower = np.maximum(mean_tpr - std_tpr, 0)

        plt.plot(base_fpr, mean_tpr,
                 label=f'{name} (AUC = {mean_auc:.3f} \u00B1 {std_auc:.3f})')
        plt.fill_between(base_fpr, tpr_lower, tpr_upper, alpha=0.2)

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.500)')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'Mean ROC Curves (5-Fold CV) - {file_suffix}')
    plt.legend(loc='lower right', fontsize='small')
    plt.grid(True)

    os.makedirs(results_path, exist_ok=True)
    plot_filename = f"{results_path}/Mean_ROC_Curves_{file_suffix}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved mean ROC plot to {plot_filename}")

# Evaluate model performance
def evaluate_model(y_true, y_pred, y_proba):
    y_proba_pos = y_proba.flatten()

    if len(np.unique(y_true)) < 2:
        return 0.0, 0.5, 0.0, 0.0, 0.0, 0, 0, 0, 0

    try:
        if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
            return 0.0, 0.5, 0.0, 0.0, 0.0, 0, 0, 0, 0

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except ValueError:
        return 0.0, 0.5, 0.0, 0.0, 0.0, 0, 0, 0, 0

    accuracy = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_proba_pos)
    except ValueError:
        auc = 0.5

    sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    return accuracy, auc, sensitivity, specificity, f1, tn, fp, fn, tp


def test_models(X_sub, y_sub, groups, save_path, code_a, code_b, label_a, label_b, stage_name, pca_num):
    kf = GroupKFold(n_splits=5)
    models_and_params = get_models_and_params(code_a, code_b)
    cv_results = []
    model_roc_data = {}

    for name, (model, param_grid) in models_and_params.items():
        print(f"\n--- Starting Grid Search & 5-Fold CV for {name} ---")

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=list(kf.split(X_sub, y_sub, groups)),
            verbose=0,
            n_jobs=-1
        )

        grid_search.fit(X_sub, y_sub)
        best_model = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")

        os.makedirs(save_path, exist_ok=True)
        model_filename = (
            f"{save_path}/{label_a}_vs_{label_b}_{stage_name}_"
            f"{pca_num}D_{name.replace(' ', '_')}.pkl"
        )
        joblib.dump(best_model, model_filename)
        print(f"Saved trained model to: {model_filename}")

        all_fold_metrics = []
        fold_roc_data = []

        for fold, (train_index, test_index) in enumerate(kf.split(X_sub, y_sub, groups)):
            X_train_fold, X_test_fold = X_sub.iloc[train_index], X_sub.iloc[test_index]
            y_train_fold, y_test_fold = y_sub.iloc[train_index], y_sub.iloc[test_index]

            best_model.fit(X_train_fold, y_train_fold)
            y_pred_fold = best_model.predict(X_test_fold)

            if hasattr(best_model, "predict_proba"):
                try:
                    y_proba_full = best_model.predict_proba(X_test_fold)
                    if y_proba_full.shape[1] == 2:
                        y_proba_fold = y_proba_full[:, 1]
                    else:
                        y_proba_fold = y_pred_fold.astype(float)
                except Exception:
                    y_proba_fold = y_pred_fold.astype(float)
            else:
                y_proba_fold = y_pred_fold.astype(float)

            fold_roc_data.append((y_test_fold.to_numpy(), y_proba_fold))

            acc, auc, sens, spec, f1, tn, fp, fn, tp = evaluate_model(
                y_test_fold.to_numpy(), y_pred_fold, y_proba_fold
            )
            all_fold_metrics.append({
                'acc': acc, 'auc': auc, 'sens': sens, 'spec': spec, 'f1': f1,
                'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
            })

        model_roc_data[name] = fold_roc_data

        df_metrics = pd.DataFrame(all_fold_metrics)
        mean_metrics = df_metrics.mean()
        std_metrics = df_metrics.std(ddof=1)

        ci_metrics = {}
        for col in ['acc', 'auc', 'sens', 'spec', 'f1']:
            ci_low, ci_high = calculate_ci_95(df_metrics[col])
            ci_metrics[f'{col}_ci_low'] = ci_low
            ci_metrics[f'{col}_ci_high'] = ci_high

        cv_results.append(
            (name,
             mean_metrics['acc'], std_metrics['acc'], ci_metrics['acc_ci_low'], ci_metrics['acc_ci_high'],
             mean_metrics['auc'], std_metrics['auc'], ci_metrics['auc_ci_low'], ci_metrics['auc_ci_high'],
             mean_metrics['sens'], std_metrics['sens'], ci_metrics['sens_ci_low'], ci_metrics['sens_ci_high'],
             mean_metrics['spec'], std_metrics['spec'], ci_metrics['spec_ci_low'], ci_metrics['spec_ci_high'],
             mean_metrics['f1'], std_metrics['f1'], ci_metrics['f1_ci_low'], ci_metrics['f1_ci_high'],
             mean_metrics['tp'], mean_metrics['tn'], mean_metrics['fp'], mean_metrics['fn'])
        )

        print(f"{name} 5-Fold CV Results (Mean (Std) [95% CI]):")
        print(
            f"  Accuracy:    {mean_metrics['acc']:.3f} ({std_metrics['acc']:.3f}) [{ci_metrics['acc_ci_low']:.3f} - {ci_metrics['acc_ci_high']:.3f}]")
        print(
            f"  AUC:         {mean_metrics['auc']:.3f} ({std_metrics['auc']:.3f}) [{ci_metrics['auc_ci_low']:.3f} - {ci_metrics['auc_ci_high']:.3f}]")
        print(
            f"  Sensitivity: {mean_metrics['sens']:.3f} ({std_metrics['sens']:.3f}) [{ci_metrics['sens_ci_low']:.3f} - {ci_metrics['sens_ci_high']:.3f}]")
        print(
            f"  Specificity: {mean_metrics['spec']:.3f} ({std_metrics['spec']:.3f}) [{ci_metrics['spec_ci_low']:.3f} - {ci_metrics['spec_ci_high']:.3f}]")
        print(
            f"  F1-score:    {mean_metrics['f1']:.3f} ({std_metrics['f1']:.3f}) [{ci_metrics['f1_ci_low']:.3f} - {ci_metrics['f1_ci_high']:.3f}]")
        print(
            f"  Mean CM:     TP={mean_metrics['tp']:.1f}, TN={mean_metrics['tn']:.1f}, FP={mean_metrics['fp']:.1f}, FN={mean_metrics['fn']:.1f}")

    return cv_results, model_roc_data


def run_pca_and_models_cv(stage_name, X_sub_clean_df, y_sub, groups, pca_num, code_a, code_b, label_a, label_b):
    print(f"\n--- PCA ({stage_name}) ---")

    X_to_pca = X_sub_clean_df.copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_to_pca)

    max_pca = min(pca_num, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=max_pca, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"PCA reduced features to {X_pca.shape[1]} with total variance explained = {explained_var:.3f}")

    os.makedirs(results_path, exist_ok=True)
    pca_filename = f"{results_path}/pca_{pca_num}.pkl"
    scaler_filename = f"{results_path}/scaler_{pca_num}.pkl"

    clean_features = X_sub_clean_df.columns.tolist()
    feature_list_filename = f"{results_path}/clean_feature_names.pkl"

    joblib.dump(pca, pca_filename)
    joblib.dump(scaler, scaler_filename)
    joblib.dump(clean_features, feature_list_filename)

    # Save PCA, scaler, and feature list for testing
    print(f"Saved PCA to {pca_filename}")
    print(f"Saved Scaler to {scaler_filename}")
    print(f"Saved Clean Feature List to {feature_list_filename}")

    X_final_df = pd.DataFrame(X_pca)

    classical_cv_results, classical_roc_data = test_models(
        X_final_df, y_sub, groups, results_path, code_a, code_b, label_a, label_b, stage_name, pca_num
    )

    all_model_roc_data = classical_roc_data.copy()
    roc_plot_suffix = f"{label_a}_vs_{label_b}_{stage_name}_({pca_num}_dim)"
    plot_mean_roc_curves(all_model_roc_data, results_path, roc_plot_suffix)

    all_results = []
    for res_tuple in classical_cv_results:
        name = res_tuple[0]
        metrics = res_tuple[1:]
        all_results.append([f"{label_a} vs {label_b}", pca_num, stage_name, name, *metrics])

    return all_results


def binary_compare(data_row_series, comparison_label, pca_dim, results_path):

    # # Choose model type based on prediction
    # if comparison_label.startswith('NC_vs_MCI'):
    #     model_name = "LR_(ElasticNet)"
    # elif comparison_label.startswith('NC_vs_AD') or comparison_label.startswith('MCI_vs_AD'):
    #     model_name = "SVM_(RBF)"
    # else:
    #     print(f"Error: Unknown comparison label: {comparison_label}")
    #     return None, None

    model_name = "SVM_(RBF)"

    stage_name = "Before Scaling"

    # Define File Paths
    model_filename = f"{results_path}/{comparison_label}_{stage_name}_{pca_dim}D_{model_name}.pkl"
    scaler_filename = f"{results_path}/scaler_{pca_dim}.pkl"
    pca_filename = f"{results_path}/pca_{pca_dim}.pkl"
    feature_list_filename = f"{results_path}/clean_feature_names.pkl"

    print(f"Model file: {model_filename}")

    # Load Components
    try:
        scaler = joblib.load(scaler_filename)
        pca = joblib.load(pca_filename)
        best_model = joblib.load(model_filename)
        clean_features = joblib.load(feature_list_filename)
    except FileNotFoundError as e:
        return None, None
    except Exception as e:
        print(f"Error loading objects for {comparison_label}: {e}. Skipping prediction.")
        return None, None

    # Check input structure
    if isinstance(data_row_series, pd.Series):
        X_subject = data_row_series.to_frame().T
    elif isinstance(data_row_series, pd.DataFrame):
        X_subject = data_row_series.head(1).copy()
    else:
        print("Error: Input data must be a pandas Series or DataFrame row.")
        return None, None

    # Filter Features
    try:
        X_subject_clean = X_subject[clean_features]
    except KeyError as e:
        return None, None

    # Scale and PCA transform
    X_scaled = scaler.transform(X_subject_clean)
    X_pca = pca.transform(X_scaled)

    # Predict
    y_pred = best_model.predict(X_pca)
    y_proba = best_model.predict_proba(X_pca)

    predicted_class = y_pred[0]
    prob_class_1 = y_proba[0][1]

    return predicted_class, prob_class_1

# Predict final diagnosis label
# Predict final diagnosis label
def final_diagnosis(data_row_series, pca_dim, results_path):
    comparisons_to_run = {
        # Pos Class (1), Neg Class (0)
        "NC_vs_AD": ('AD', 'NC'),  # AD (1) vs NC (0)
        "NC_vs_MCI": ('MCI', 'NC'),  # MCI (1) vs NC (0)
        "MCI_vs_AD": ('AD', 'MCI')  # AD (1) vs MCI (0)
    }

    all_predictions = {}

    # Run all three binary models
    for label, (pos_class, neg_class) in comparisons_to_run.items():
        predicted_class_bin, probability_class_1 = binary_compare(
            data_row_series=data_row_series,
            comparison_label=label,
            pca_dim=pca_dim,
            results_path=results_path
        )

        if predicted_class_bin is None:
            return "Prediction Failed (Missing Model Files)"

        # 0 -> neg_class, 1 -> pos_class
        predicted_label = pos_class if predicted_class_bin == 1 else neg_class

        all_predictions[label] = {
            'predicted_class': predicted_label,
            'confidence_in_positive': probability_class_1
        }

    # --- MAX-VOTE ENSEMBLE DECISION LOGIC ---

    vote_counts = {'NC': 0, 'MCI': 0, 'AD': 0}

    # Cast votes based on predicted label
    vote_counts[all_predictions["NC_vs_MCI"]['predicted_class']] += 1
    vote_counts[all_predictions["NC_vs_AD"]['predicted_class']] += 1
    vote_counts[all_predictions["MCI_vs_AD"]['predicted_class']] += 1

    # Define diagnosis order
    diagnosis_order = ['AD', 'MCI', 'NC']
    max_votes = -1
    final_diagnosis_result = "Unknown"

    # Iterate in order of preference (AD, then MCI, then NC)
    for diagnosis in diagnosis_order:
        current_votes = vote_counts[diagnosis]
        if current_votes > max_votes:
            max_votes = current_votes
            final_diagnosis_result = diagnosis

    # Note: Removed the print statements here to avoid excessive output for large test files.
    # The summary is now printed only if running the single subject demo manually.

    return final_diagnosis_result


# Main Driver
if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')

    try:
        try:
            # Determine directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            os.chdir(script_dir)
            print(f"Changed directory to: {script_dir}")
        except NameError:
            print("Running in interactive mode. Using current directory.")
            script_dir = os.getcwd()

        pca_num = 50

        # Load training data
        df, data, filename = load_data()
        results_path = f"Results/{data}/PCA - {pca_num}"

        if "Subject" not in df.columns:
            raise ValueError("Dataset must contain a 'Subject' column for GroupKFold.")

        subjects = df["Subject"]
        df = df.drop(columns=["Image_ID", "Subject"], errors="ignore")

        # Convert labels to numbers
        df["Diagnosis"] = df["Diagnosis"].replace({"NC": 1, "MCI": 2, "AD": 3})
        X = df.drop(columns=["Diagnosis"])
        y = df["Diagnosis"]

        label_map = {1: "NC", 2: "MCI", 3: "AD"}
        comparisons = [(1, 2), (1, 3), (2, 3)]

        all_results = []

        # Test one or more images from csv file
        test_filename = ".venv/Scripts/Datasets/Aligned Datasets/ADNI_aligned_data.csv"

        try:
            test_df = pd.read_csv(test_filename)
            print(f"Loaded test data for multiple subjects from: {test_filename}")

            if "Subject" not in test_df.columns:
                raise ValueError("Test dataset must contain a 'Subject' column.")

            # Prepare an output DataFrame to store results
            test_results = []

            print("\n" + "#" * 50)
            print("Running Multi-Subject Diagnosis")

            # Iterate through each row in the test DataFrame
            for index, row in test_df.iterrows():
                subject_id = row['Subject']

                # Drop non-feature columns for prediction
                subject_data = row.drop(["Image_ID", "Subject", "Diagnosis"], errors="ignore")

                # Perform 3-way prediction
                final_pred = final_diagnosis(
                    data_row_series=subject_data,
                    pca_dim=pca_num,
                    results_path=results_path
                )

                diagnosis_to_code = {"NC": 1, "MCI": 2, "AD": 3}

                # Convert string prediction to numerical code (1: NC, 2: MCI, 3: AD)
                final_pred_num = diagnosis_to_code.get(final_pred, 0)

                test_results.append({
                    'Subject': subject_id,
                    'Image_ID': row.get('Image_ID', 'N/A'),
                    'Predicted_Diagnosis': final_pred_num,
                    'True_Diagnosis': row.get('Diagnosis', 'N/A')
                })

                print(f"Subject {subject_id}: Predicted -> {final_pred}")

            test_results_df = pd.DataFrame(test_results)
            test_output_filename = f".venv/Scripts/Results/Test Results/Multi_Subject_Prediction_Summary.csv"
            os.makedirs(os.path.dirname(test_output_filename), exist_ok=True)
            test_results_df.to_csv(test_output_filename, index=False)

            print("#" * 50)
            print(f"Final predictions saved to: {test_output_filename}")


        except FileNotFoundError:
            print(f"Error: Test file {test_filename} not found. Please ensure it is in the same directory.")
        except Exception as e:
            print(f"An error occurred during multi-subject prediction demo: {e}")
            import traceback

            traceback.print_exc()

        print("Program Terminated")


    except FileNotFoundError as e:
        print(f"Error: Could not find file {e.filename}. Please check your file path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()