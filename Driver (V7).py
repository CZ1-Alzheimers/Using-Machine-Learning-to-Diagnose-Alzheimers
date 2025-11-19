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

from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, accuracy_score, f1_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping

import joblib

matplotlib.use('Agg')

tf.random.set_seed(42)
np.random.seed(42)


def get_data_name(filename):
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

    return data_source


# Load training data
def load_data():
    # Training CSV
    filename = "Datasets/ADNI-Oasis-AIBL_dataset.csv"
    print(f"Loading training data from: {filename}")

    # Load dataset
    df = pd.read_csv(filename)

    data_source = get_data_name(filename)

    return df, data_source, filename


# Define models
def get_models_and_params(code_a, code_b):
    # SVM (RBF)
    svm_config = {
        "SVM (RBF)": (SVC(probability=True, random_state=42, tol=1e-3, class_weight='balanced'),
                      {'C': [0.1, 1, 10, 100], 'kernel': ['rbf'], 'gamma': ['scale', 0.001, 0.01, 0.1]})
    }

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


all_roc_data = []


def test_models(X_sub, y_sub, groups, save_path, code_a, code_b, label_a, label_b, stage_name, pca_num):
    kf = GroupKFold(n_splits=5)
    models_and_params = get_models_and_params(code_a, code_b)
    cv_results = []
    model_roc_data = {}

    for name, (model_estimator, param_grid) in models_and_params.items():
        print(f"\n--- Starting Grid Search & 5-Fold CV for {name} ---")

        # Create pipeline to reduce data leakage
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=pca_num, random_state=42)),
            ('model', model_estimator)  # The classifier
        ])

        # Parameter grid
        pipeline_param_grid = {}
        for key, value in param_grid.items():
            # Parameters like C, kernel, gamma belong to the 'model' step of the pipeline
            pipeline_param_grid[f'model__{key}'] = value

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=pipeline_param_grid,
            scoring='roc_auc',
            # Pass groups for GroupKFold to ensure subject IDs are respected
            cv=list(kf.split(X_sub, y_sub, groups)),
            verbose=0,
            n_jobs=-1
        )

        grid_search.fit(X_sub, y_sub)
        best_pipeline = grid_search.best_estimator_

        # Extract and print best model parameters
        best_params = {k.replace('model__', ''): v for k, v in grid_search.best_params_.items()}
        print(f"Best parameters for {name}: {best_params}")

        # Save best pipeline
        os.makedirs(save_path, exist_ok=True)
        model_filename = (
            f"{save_path}/{label_a}_vs_{label_b}_{stage_name}_"
            f"{pca_num}D_{name.replace(' ', '_')}_Pipeline.pkl"  # Save the full pipeline
        )
        joblib.dump(best_pipeline, model_filename)
        print(f"Saved trained pipeline to: {model_filename}")

        # Cross validation with best model
        all_fold_metrics = []
        fold_roc_data = []

        # Iterate over folds
        for fold, (train_index, test_index) in enumerate(kf.split(X_sub, y_sub, groups)):
            X_train_fold, X_test_fold = X_sub.iloc[train_index], X_sub.iloc[test_index]
            y_train_fold, y_test_fold = y_sub.iloc[train_index], y_sub.iloc[test_index]

            # Fit pipeline on training data
            best_pipeline.fit(X_train_fold, y_train_fold)

            # Predict on test data of this fold
            y_pred_fold = best_pipeline.predict(X_test_fold)

            if hasattr(best_pipeline, "predict_proba"):
                try:
                    y_proba_full = best_pipeline.predict_proba(X_test_fold)
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

    all_roc_data.append(model_roc_data)
    return cv_results, model_roc_data


# Plot all ROCs togetehr
def plot_combined_mean_roc(all_roc_dicts, labels, results_path, file_suffix):
    plt.figure(figsize=(12, 10))
    base_fpr = np.linspace(0, 1, 101)

    for roc_dict, label in zip(all_roc_dicts, labels):
        tprs_all = []
        aucs_all = []

        for model_name, fold_data in roc_dict.items():
            tprs = []
            aucs = []

            for y_test, y_proba in fold_data:
                if len(np.unique(y_test)) < 2:
                    continue

                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc_calc(fpr, tpr)

                tpr_interp = np.interp(base_fpr, fpr, tpr)
                tpr_interp[0] = 0.0
                tprs.append(tpr_interp)
                aucs.append(roc_auc)

            if tprs:
                tprs_all.extend(tprs)
                aucs_all.extend(aucs)

        if not tprs_all:
            print(f"Warning: No ROC data available for {label}. Skipping.")
            continue

        mean_tpr = np.mean(tprs_all, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc_calc(base_fpr, mean_tpr)
        std_auc = np.std(aucs_all)

        plt.plot(base_fpr, mean_tpr,
                 label=f'{label} (AUC = {mean_auc:.3f} ± {std_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'Combined Mean ROC Curves - {file_suffix}')
    plt.grid(True)
    plt.legend(loc="lower right")

    os.makedirs(results_path, exist_ok=True)
    output_file = f"{results_path}/Combined_ROC_All_Comparisons_{file_suffix}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved combined ROC plot to: {output_file}")


def run_pca_and_models_cv(stage_name, X_sub_clean_df, y_sub, groups, pca_num, code_a, code_b, label_a, label_b):
    print(f"\n--- Model Training & CV (Stage: {stage_name}) ---")

    # Save the feature list
    os.makedirs(results_path, exist_ok=True)
    clean_features = X_sub_clean_df.columns.tolist()
    feature_list_filename = f"{results_path}/clean_feature_names.pkl"
    joblib.dump(clean_features, feature_list_filename)
    print(f"Saved Clean Feature List to {feature_list_filename}")

    classical_cv_results, classical_roc_data = test_models(
        X_sub_clean_df, y_sub, groups, results_path, code_a, code_b, label_map[a], label_map[b], stage_name, pca_num
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


def plot_confusion_matrix(cm, class_names, results_path, filename_suffix):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {filename_suffix}")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # Add CM values
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    os.makedirs(results_path, exist_ok=True)
    out_file = f"{results_path}/Confusion_Matrix_{filename_suffix}.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved confusion matrix plot to: {out_file}")


def binary_compare(data_row_series, comparison_label, pca_dim, results_path):
    model_name = "SVM_(RBF)"
    stage_name = "Before Scaling"

    # Define File Paths
    pipeline_filename = (
        f"{results_path}/{comparison_label}_{stage_name}_{pca_dim}D_{model_name}_Pipeline.pkl"
    )
    feature_list_filename = f"{results_path}/clean_feature_names.pkl"

    # Load Components
    try:
        # Load the single object that contains the Scaler, PCA, and Model
        best_pipeline = joblib.load(pipeline_filename)
        clean_features = joblib.load(feature_list_filename)
    except FileNotFoundError as e:
        return "MODEL NOT FOUND", "MODEL NOT FOUND"
    except Exception as e:
        print(f"Error loading objects for {comparison_label}: {e}. Skipping prediction.")
        return "LOAD ERROR", "LOAD ERROR"

    # Check input structure
    if isinstance(data_row_series, pd.Series):
        X_subject = data_row_series.to_frame().T
    elif isinstance(data_row_series, pd.DataFrame):
        X_subject = data_row_series.head(1).copy()
    else:
        print("Error: Input data must be a pandas Series or DataFrame row.")
        return "INPUT ERROR", "INPUT ERROR"

    # Filter Features
    try:
        # Ensure input data only contains the features the model was trained on
        X_subject_clean = X_subject[clean_features]
    except KeyError as e:
        print(f"Error: Missing features in test data: {e}. Skipping prediction.")
        return "FEATURE_ERROR", "FEATURE_ERROR"

    # Predict using the single pipeline call
    try:
        y_pred = best_pipeline.predict(X_subject_clean)
        y_proba = best_pipeline.predict_proba(X_subject_clean)

        predicted_class = y_pred[0]  # 0 or 1
        prob_class_1 = y_proba[0][1]  # Probability of class 1

        return predicted_class, prob_class_1
    except Exception as e:
        print(f"Error during prediction for {comparison_label}: {e}. Skipping prediction.")
        return "PREDICTION ERROR", "PREDICTION ERROR"


# Predict final diagnosis label
# Mapping for saving results as numbers
diagnosis_to_code = {"NC": 1, "MCI": 2, "AD": 3, "Prediction Failed (Missing Model Files)": 0}


def final_diagnosis_probabilistic(data_row_series, pca_dim, results_path):
    comparisons_map = {
        # Comparison Labels
        "NC_vs_AD": ('AD', 'NC', 3, 1),
        "NC_vs_MCI": ('MCI', 'NC', 2, 1),
        "MCI_vs_AD": ('AD', 'MCI', 3, 2)
    }

    class_scores = {'NC': 0.0, 'MCI': 0.0, 'AD': 0.0}
    binary_results = {}

    # Run all three binary models and get scores
    for label, (pos_class, neg_class, pos_code, neg_code) in comparisons_map.items():
        predicted_class_bin, probability_class_1 = binary_compare(
            data_row_series=data_row_series,
            comparison_label=label,
            pca_dim=pca_dim,
            results_path=results_path
        )

        # Handle errors
        if isinstance(predicted_class_bin, str):
            # If binary prediction fails to load or run, the final prediction also fails.
            binary_results[f'{label}_Prediction'] = diagnosis_to_code[f'Prediction Failed (Missing Model Files)']
            binary_results[f'{label}_Probability'] = np.nan
            final_diagnosis_str = "Prediction Failed (Missing Model Files)"
            return final_diagnosis_str, binary_results

        # Predict Class
        if predicted_class_bin == 1:
            # Predicted 1
            predicted_diagnosis_str = pos_class
            predicted_diagnosis_code = pos_code
        else:
            # Predicted 0
            predicted_diagnosis_str = neg_class
            predicted_diagnosis_code = neg_code

        # Store binary results in the dictionary
        binary_results[f'{label}_Prediction'] = predicted_diagnosis_code
        binary_results[f'{label}_Probability'] = probability_class_1

        # Score 3-way diagnosis:
        class_scores[pos_class] += probability_class_1

        # Add confidence for 0 classification
        class_scores[neg_class] += (1 - probability_class_1)

    # Find the class with the highest accumulated score
    final_diagnosis_result = max(class_scores, key=class_scores.get)

    # Tie-breaking logic: AD > MCI > NC preference
    max_score = class_scores[final_diagnosis_result]

    if max_score == class_scores['AD']:
        final_diagnosis_str = 'AD'
    elif max_score == class_scores['MCI'] and max_score >= class_scores['AD']:
        final_diagnosis_str = 'MCI'
    elif max_score == class_scores['NC'] and max_score >= class_scores['MCI'] and max_score >= class_scores['AD']:
        final_diagnosis_str = 'NC'
    else:
        # Use max() if tiebreaker fails
        final_diagnosis_str = final_diagnosis_result

    return final_diagnosis_str, binary_results


def check_accuracy(test_results_path):
    print("\n" + "=" * 50)
    print("Checking Multi-Subject Prediction Accuracy")
    print("=" * 50)

    try:
        df_results = pd.read_csv(test_results_path)
    except FileNotFoundError:
        print(f"Error: Prediction results file not found at {test_results_path}")
        return

    # Data cleaning and preparation
    df_results['Predicted_Diagnosis'] = df_results['Predicted_Diagnosis'].astype(int)

    # Clean True_Diagnosis
    df_clean = df_results.dropna(subset=['True_Diagnosis']).copy()
    if len(df_results) != len(df_clean):
        print(
            f"Warning: Dropped {len(df_results) - len(df_clean)} rows with missing True_Diagnosis for accuracy calculation.")

    df_clean['True_Diagnosis'] = df_clean['True_Diagnosis'].astype(int)

    # Get true and predicted labels
    y_true = df_clean['True_Diagnosis']
    y_pred = df_clean['Predicted_Diagnosis']

    if y_true.empty:
        print("Error: No valid data remaining after cleaning to calculate accuracy.")
        return

    # Define the mapping for clarity (1=NC, 2=MCI, 3=AD)
    diagnosis_map = {1: "NC", 2: "MCI", 3: "AD", 0: "Failed"}

    # Calculate overall accuracy
    overall_accuracy = accuracy_score(y_true, y_pred)

    # Get unique labels
    # Only include labels present in the actual data for the report
    unique_labels = sorted(list(set(y_pred.unique()) | set(y_true.unique())))
    target_names = [diagnosis_map.get(l, str(l)) for l in unique_labels]

    # Calculate classification report and confusion matrix
    report = classification_report(y_true, y_pred, labels=unique_labels,
                                   target_names=target_names, output_dict=False, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    cm_df = pd.DataFrame(cm, index=[f'True {diagnosis_map.get(l, l)}' for l in unique_labels],
                         columns=[f'Predicted {diagnosis_map.get(l, l)}' for l in unique_labels])

    test_results_path = f"Results/Test Results/{data}"
    plot_confusion_matrix(cm, target_names, test_results_path, get_data_name(test_results_path))

    # Print results
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({len(y_true)} samples)")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm_df)


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

        # Convert labels to numbers (1: NC, 2: MCI, 3: AD)
        df["Diagnosis"] = df["Diagnosis"].replace({"NC": 1, "MCI": 2, "AD": 3})
        X = df.drop(columns=["Diagnosis"])
        y = df["Diagnosis"]

        label_map = {1: "NC", 2: "MCI", 3: "AD"}
        # Mapping for saving results as numbers
        diagnosis_to_code = {"NC": 1, "MCI": 2, "AD": 3, "Prediction Failed (Missing Model Files)": 0}

        comparisons = [(1, 2), (1, 3), (2, 3)]  # (NC vs MCI), (NC vs AD), (MCI vs AD)

        all_results = []

        # Train and cross validation
        for a, b in comparisons:
            print(f"\n{'=' * 70}")
            print(f"Classification Task: {label_map[a]} (0) vs {label_map[b]} (1)")
            print(f"{'=' * 70}")

            mask = y.isin([a, b])
            X_sub, y_sub = X[mask], y[mask]
            groups_sub = subjects[mask].reset_index(drop=True)

            class_counts = y_sub.value_counts()
            min_class_count = class_counts.min()

            if min_class_count < 5:
                print(
                    f"Skipping {label_map[a]} vs {label_map[b]} comparison: minimum class ({min_class_count}) < n_splits (5).")
                continue

            y_sub = y_sub.replace({a: 0, b: 1}).reset_index(drop=True)
            X_sub = X_sub.reset_index(drop=True)

            vt = VarianceThreshold(threshold=0.0)
            X_sub_clean = vt.fit_transform(X_sub)
            X_sub_clean_df = pd.DataFrame(X_sub_clean, columns=X_sub.columns[vt.get_support()])

            results_before = run_pca_and_models_cv(
                "Before Scaling", X_sub_clean_df, y_sub, groups_sub,
                pca_num, a, b, label_map[a], label_map[b]
            )

            all_results.extend(results_before)

        labels_list = ["NC vs MCI", "NC vs AD", "MCI vs AD"]
        plot_combined_mean_roc(all_roc_data, labels_list, results_path, "All_ROC.png")

        # Save results
        summary_columns = [
            "Comparison", "PCA_Num", "PCA Stage", "Model",
            "Accuracy_Mean", "Accuracy_Std", "Accuracy_95CI_Low", "Accuracy_95CI_High",
            "AUC_Mean", "AUC_Std", "AUC_95CI_Low", "AUC_95CI_High",
            "Sensitivity_Mean", "Sensitivity_Std", "Sensitivity_95CI_Low", "Sensitivity_95CI_High",
            "Specificity_Mean", "Specificity_Std", "Specificity_95CI_Low", "Specificity_95CI_High",
            "F1-score_Mean", "F1-score_Std", "F1-score_95CI_Low", "F1-score_95CI_High",
            "Mean_TP", "Mean_TN", "Mean_FP", "Mean_FN"
        ]

        summary_df = pd.DataFrame(all_results, columns=summary_columns)
        os.makedirs(results_path, exist_ok=True)
        summary_filename = f"{results_path}/model_comparison_summary_{pca_num}_dimensions_(Pipeline).csv"
        summary_df.to_csv(summary_filename, index=False)
        summary_df.to_csv("model_comparison_summary.csv", index=False)
        print(f"\nSaved all results into: {summary_filename}")

        # Test multiple images from a CSV file
        test_filename = "Datasets/Aligned Datasets/AIBL_aligned_data.csv"

        test_data = get_data_name(test_filename)

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
                demo_subject_data = row.drop(["Image_ID", "Subject", "Diagnosis"], errors="ignore")

                # Perform 3-way prediction (returns string and binary results dict)
                final_pred_str, binary_comp_results = final_diagnosis_probabilistic(
                    data_row_series=demo_subject_data,
                    pca_dim=pca_num,
                    results_path=results_path
                )

                # Convert string prediction to numerical code (1: NC, 2: MCI, 3: AD)
                final_pred_code = diagnosis_to_code.get(final_pred_str, 0)  # Use 0 for 'Unknown' or error

                result_entry = {
                    'Subject': subject_id,
                    'Image_ID': row.get('Image_ID', 'N/A'),
                    'Predicted_Diagnosis': final_pred_code,  # Save the numerical code
                    'True_Diagnosis': row.get('Diagnosis', 'N/A')
                }

                # Add binary comparison results
                result_entry.update(binary_comp_results)

                test_results.append(result_entry)

                print(f"Subject {subject_id}: Predicted -> {final_pred_str} ({final_pred_code})")

            # Test results path
            test_results_df = pd.DataFrame(test_results)

            # path = Results/ Test Results/ Training Dataset/ Test Dataset
            test_output_filename = f"Results/Test Results/{data}/Test_Prediction_{test_data}.csv"
            os.makedirs(os.path.dirname(test_output_filename), exist_ok=True)
            test_results_df.to_csv(test_output_filename, index=False)

            print("#" * 50)
            print(f"Final predictions saved to: {test_output_filename}")

            check_accuracy(test_output_filename)


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