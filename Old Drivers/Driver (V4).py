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

matplotlib.use('Agg')

tf.random.set_seed(42)
np.random.seed(42)

#calculates 95% confidence interval
def calculate_ci_95(data_series):
    """Calculates the 95% confidence interval for a pandas Series."""
    n = len(data_series)
    if n < 2:
        return np.nan, np.nan
    
    mean = data_series.mean()
    # Get the standard error
    std_err = data_series.std(ddof=1) / np.sqrt(n)
    
    # Get the t-statistic for a 95% CI
    # ppf(0.975) gives the t-value for the upper tail (2.5%)
    margin = stats.t.ppf(0.975, df=n - 1) * std_err
    
    return mean - margin, mean + margin


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

#Evaluates models (added confusion matrix metrics)
def evaluate_model(y_true, y_pred, y_proba):
    y_proba_pos = y_proba.flatten()

    if len(np.unique(y_true)) < 2:
        return 0.0, 0.5, 0.0, 0.0, 0.0, 0, 0, 0, 0  

    try:
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


def test_classical_models_cv(X_sub, y_sub, groups):

    kf = GroupKFold(n_splits=5)

    models_and_params = {
        "LR (L1)": (LogisticRegression(solver='liblinear', max_iter=1000, random_state=42, class_weight='balanced'),
                    {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1']}),
        
        "LR (L2)": (LogisticRegression(solver='liblinear', max_iter=1000, random_state=42, class_weight='balanced'),
                    {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l2']}),

        "LR (ElasticNet)": (LogisticRegression(solver='saga', max_iter=3000, random_state=42, class_weight='balanced', tol=1e-3),
                            {'C': [0.1, 1, 10], 'penalty': ['elasticnet'], 'l1_ratio': [0.3, 0.5, 0.7]}),

        "SVM (Linear)": (SVC(probability=True, random_state=42, tol=1e-3, class_weight='balanced'),
                         {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear']}),
        
        "SVM (RBF)": (SVC(probability=True, random_state=42, tol=1e-3, class_weight='balanced'),
                      {'C': [0.1, 1, 10, 100], 'kernel': ['rbf'], 'gamma': ['scale', 0.001, 0.01, 0.1]}),
        
        "SVM (Poly d2)": (SVC(probability=True, random_state=42, tol=1e-3, class_weight='balanced'),
                        {'C': [0.1, 1, 10], 'kernel': ['poly'], 'degree': [2], 'gamma': ['scale', 'auto']}),
        
        "SVM (Sigmoid)": (SVC(probability=True, random_state=42, tol=1e-3, class_weight='balanced'),
                         {'C': [0.1, 1, 10], 'kernel': ['sigmoid'], 'gamma': ['scale', 'auto']}),
    }
    
    cv_results = []
    model_roc_data = {} # To store data for the mean ROC plot

    for name, (model, param_grid) in models_and_params.items():
        print(f"\n--- Starting Grid Search & 5-Fold CV for {name} ---")

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=list(kf.split(X_sub, y_sub, groups)), # Pass iterator as list
            verbose=0,
            n_jobs=-1
        )

        grid_search.fit(X_sub, y_sub)
        best_model = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")

        # Evaluate the best model on all 5 folds
        all_fold_metrics = []
        fold_roc_data = [] # Store (y_test, y_proba) for each fold
        
        for fold, (train_index, test_index) in enumerate(kf.split(X_sub, y_sub, groups)):
            X_train_fold, X_test_fold = X_sub.iloc[train_index], X_sub.iloc[test_index]
            y_train_fold, y_test_fold = y_sub.iloc[train_index], y_sub.iloc[test_index]

            # Re-fit the best model on the fold's training data
            best_model.fit(X_train_fold, y_train_fold)
            y_pred_fold = best_model.predict(X_test_fold)

            # Robust handling of predict_proba
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
        print(f"  Accuracy:    {mean_metrics['acc']:.3f} ({std_metrics['acc']:.3f}) [{ci_metrics['acc_ci_low']:.3f} - {ci_metrics['acc_ci_high']:.3f}]")
        print(f"  AUC:         {mean_metrics['auc']:.3f} ({std_metrics['auc']:.3f}) [{ci_metrics['auc_ci_low']:.3f} - {ci_metrics['auc_ci_high']:.3f}]")
        print(f"  Sensitivity: {mean_metrics['sens']:.3f} ({std_metrics['sens']:.3f}) [{ci_metrics['sens_ci_low']:.3f} - {ci_metrics['sens_ci_high']:.3f}]")
        print(f"  Specificity: {mean_metrics['spec']:.3f} ({std_metrics['spec']:.3f}) [{ci_metrics['spec_ci_low']:.3f} - {ci_metrics['spec_ci_high']:.3f}]")
        print(f"  F1-score:    {mean_metrics['f1']:.3f} ({std_metrics['f1']:.3f}) [{ci_metrics['f1_ci_low']:.3f} - {ci_metrics['f1_ci_high']:.3f}]")
        print(f"  Mean CM:     TP={mean_metrics['tp']:.1f}, TN={mean_metrics['tn']:.1f}, FP={mean_metrics['fp']:.1f}, FN={mean_metrics['fn']:.1f}")

    return cv_results, model_roc_data


def run_pca_and_models_cv(stage_name, X_sub_clean_df, y_sub, groups, pca_num, label_a, label_b, initial_scaler=None):
    print(f"\n--- PCA ({stage_name}) ---")

    X_to_pca = X_sub_clean_df.copy()
    if initial_scaler:
        X_scaled = initial_scaler.fit_transform(X_to_pca)
    else:
        X_scaled = X_to_pca.values

    max_pca = min(pca_num, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=max_pca, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    final_scaler = StandardScaler()
    X_final_scaled = final_scaler.fit_transform(X_pca)
    X_final_df = pd.DataFrame(X_final_scaled)

    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"PCA reduced features to {X_final_df.shape[1]} with total variance explained = {explained_var:.3f}")

    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_sub), y=y_sub)
    class_weight_dict = dict(enumerate(class_weights))

    kf = GroupKFold(n_splits=5)

    # Classical ML models with Grid Search and CV
    classical_cv_results, classical_roc_data = test_classical_models_cv(X_final_df, y_sub, groups)
    
    # --- Plot Mean ROC Curves for ALL models ---
    all_model_roc_data = classical_roc_data.copy()
    
    roc_plot_suffix = f"{label_a}_vs_{label_b}_{stage_name}_({pca_num}_dim)"
    plot_mean_roc_curves(all_model_roc_data, results_path, roc_plot_suffix)


    # Collect all results
    all_results = []
    for res_tuple in classical_cv_results:
        # (name, acc_m, acc_s, acc_cil, acc_cih, auc_m, ...)
        name = res_tuple[0]
        metrics = res_tuple[1:]
        all_results.append([f"{label_a} vs {label_b}", pca_num, stage_name, name, *metrics])

    return all_results

# Main Driver
if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')

    try:
        # Change to the script's directory (if running as a script)
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            os.chdir(script_dir)
            print(f"Changed directory to: {script_dir}")
        except NameError:
            print("Running in interactive mode (e.g., Jupyter). Using current directory.")
            script_dir = os.getcwd()


        # Load dataset
        pca_num = 100
        filename = "Datasets/ADNI-Oasis_dataset.csv"

        # Name results path
        if "Oasis" in filename and "ADNI" not in filename and "AIBL" not in filename:
           data = "Oasis"
        elif "ADNI" in filename and "Oasis" not in filename and "AIBL" not in filename:
            data = "ADNI"
        elif "AIBL" in filename and "ADNI" not in filename and "Oasis" not in filename:
            data = "AIBL"
        elif "Oasis" in filename and "ADNI" in filename and "AIBL" not in filename:
            data = "ADNI-Oasis"
        elif "ADNI" in filename and "AIBL" in filename and "Oasis" not in filename:
            data = "ADNI-AIBL"
        elif "Oasis" in filename and "AIBL" in filename and "ADNI" not in filename:
            data = "Oasis-AIBL"
        else:
            data = "Complete"

        results_path = f"Results/{data}/PCA - {pca_num}"
        df = pd.read_csv(filename)
        
        if "Subject" not in df.columns:
            raise ValueError("Dataset must contain a 'Subject' column for GroupKFold.")
            
        subjects = df["Subject"] # Keep for grouping
        df = df.drop(columns=["Image_ID", "Subject"], errors="ignore")

        # Map string labels to numeric: NC=1 (Control), MCI=2, AD=3
        df["Diagnosis"] = df["Diagnosis"].replace({"NC": 1, "MCI": 2, "AD": 3})
        X = df.drop(columns=["Diagnosis"])
        y = df["Diagnosis"]

        label_map = {1: "NC", 2: "MCI", 3: "AD"}
        comparisons = [(1, 2), (1, 3), (2, 3)]

        initial_scaler = MinMaxScaler()

        all_results = []

        for a, b in comparisons:
            print(f"\n{'=' * 70}")
            print(f"Classification Task: {label_map[a]} (0) vs {label_map[b]} (1)")
            print(f"{'=' * 70}")

            # Prepare binary classification subset
            mask = y.isin([a, b])
            X_sub, y_sub = X[mask], y[mask]
            groups_sub = subjects[mask].reset_index(drop=True)

            class_counts = y_sub.value_counts()
            min_class_count = class_counts.min()
            print(f"Class distribution: {label_map[a]}={class_counts.get(a, 0)}, {label_map[b]}={class_counts.get(b, 0)}")

            if min_class_count < 10:
                print(f"Warning: Minimum class has only {min_class_count} samples. 5-fold CV may fail or be unstable.")

            if min_class_count < 5:
                print(f"Skipping {label_map[a]} vs {label_map[b]} comparison: minimum class ({min_class_count}) < n_splits (5).")
                continue
                
            y_sub = y_sub.replace({a: 0, b: 1}).reset_index(drop=True)
            X_sub = X_sub.reset_index(drop=True)

            vt = VarianceThreshold(threshold=0.0)
            X_sub_clean = vt.fit_transform(X_sub)
            X_sub_clean_df = pd.DataFrame(X_sub_clean, columns=X_sub.columns[vt.get_support()])
            print(f"Removed {X_sub.shape[1] - X_sub_clean_df.shape[1]} constant features.")

            # PCA Before Scaling
            results_before = run_pca_and_models_cv(
                "Before Scaling", X_sub_clean_df, y_sub, groups_sub,
                pca_num, label_map[a], label_map[b], initial_scaler=None
            )

            # PCA After Scaling (MinMaxScaler)
            results_after = run_pca_and_models_cv(
                "After Scaling", X_sub_clean_df, y_sub, groups_sub,
                pca_num, label_map[a], label_map[b], initial_scaler=initial_scaler
            )

            all_results.extend(results_before + results_after)

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
        
        summary_filename = f"{results_path}/model_comparison_summary_{pca_num}_dimensions.csv"
        summary_df.to_csv(summary_filename, index=False)
        print(f"\nSaved all results into: {summary_filename}")

    except FileNotFoundError as e:
        print(f"Error: Could not find file {e.filename}. Please check your file path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()