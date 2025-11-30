# Import libraries
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
from scipy import stats
from sklearn.metrics import roc_curve, auc as auc_calc
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, f1_score, recall_score, classification_report,
    balanced_accuracy_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

# Harmonization libraries
try:
    from skimage.exposure import match_histograms

    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False

try:
    from neuroHarmonize import harmonizationLearn, harmonizationApply

    NEUROHARMONIZE_AVAILABLE = True
except Exception:
    NEUROHARMONIZE_AVAILABLE = False

matplotlib.use('Agg')
np.random.seed(42)

# Configuration Settings
ENABLE_HISTOGRAM_MATCHING = True
ENABLE_COMBAT = True
ENABLE_SMOTE = True
ENABLE_SAMPLE_WEIGHT = True
ENABLE_CALIBRATION = True
CALIBRATION_METHOD = 'sigmoid'
SMOTE_K_NEIGHBORS = 5
MCI_PROB_BOOST = 1.53 # Helps MCI tuning

all_roc_data = []


# Rename models
def comparison_to_filename(a, b):
    mapping = {
        (1, 2): "NCvMCI.pkl",
        (1, 3): "NCvAD.pkl",
        (2, 3): "MCIvAD.pkl"
    }
    key = (min(a, b), max(a, b))
    return mapping.get(key, f"Pipeline_{a}v{b}.pkl")

# Get dataname from file
def get_data_name(filename):
    if "Oasis" in filename and "ADNI" not in filename and "AIBL" not in filename:
        return "Oasis"
    elif "ADNI" in filename and "Oasis" not in filename and "AIBL" not in filename:
        return "ADNI"
    elif "AIBL" in filename and "ADNI" not in filename and "Oasis" not in filename:
        return "AIBL"
    elif "Oasis" in filename and "ADNI" in filename and "AIBL" not in filename:
        return "ADNI-Oasis"
    elif "ADNI" in filename and "AIBL" in filename and "Oasis" not in filename:
        return "ADNI-AIBL"
    elif "Oasis" in filename and "AIBL" in filename and "ADNI" not in filename:
        return "Oasis-AIBL"
    else:
        return "Complete"

# Load in data from file
def load_data(filename):
    print(f"Loading training data from: {filename}")
    df = pd.read_csv(filename)
    data_source = get_data_name(filename)
    return df, data_source, filename

# Apply ComBat to harmonize/ reduce batch effects
# Ensures consistency between datasets
def apply_combat(features_df, batch_vector, covars_df=None):
    if not NEUROHARMONIZE_AVAILABLE:
        raise RuntimeError("neuroHarmonize not installed.")
    features = features_df.values.astype(float)
    model, _ = harmonizationLearn(features, batch_vector, covars=covars_df)
    harmonized = harmonizationApply(features, batch_vector, model)
    return pd.DataFrame(harmonized, index=features_df.index, columns=features_df.columns)

# Plot ROC for each comparison
def plot_mean_roc_curves(model_roc_data, results_path, file_suffix):
    plt.figure(figsize=(10, 8))
    base_fpr = np.linspace(0, 1, 101)
    for name, fold_data in model_roc_data.items():
        tprs = []
        aucs = []
        if not fold_data: continue
        for y_test, y_proba in fold_data:
            if len(np.unique(y_test)) < 2: continue
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc_calc(fpr, tpr)
            tpr_interp = np.interp(base_fpr, fpr, tpr);
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp);
            aucs.append(roc_auc)
        if not tprs: continue
        mean_tpr = np.mean(tprs, axis=0);
        mean_tpr[-1] = 1.0
        mean_auc = auc_calc(base_fpr, mean_tpr);
        std_auc = np.std(aucs)
        plt.plot(base_fpr, mean_tpr, label=f'{name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate')
    plt.title(f'Mean ROC Curves')
    plt.legend(loc='lower right');
    plt.grid(True)
    os.makedirs(results_path, exist_ok=True)
    plt.savefig(f"{results_path}/Mean_ROC_Curves_{file_suffix}.png", dpi=300, bbox_inches='tight');
    plt.close()

# Combine all ROCs into one plot
def plot_combined_mean_roc(all_roc_dicts, labels, results_path, file_suffix):
    plt.figure(figsize=(12, 10))
    base_fpr = np.linspace(0, 1, 101)
    for roc_dict, label in zip(all_roc_dicts, labels):
        tprs_all = [];
        aucs_all = []
        for model_name, fold_data in roc_dict.items():
            for y_test, y_proba in fold_data:
                if len(np.unique(y_test)) < 2: continue
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc_calc(fpr, tpr)
                tpr_interp = np.interp(base_fpr, fpr, tpr);
                tpr_interp[0] = 0.0
                tprs_all.append(tpr_interp);
                aucs_all.append(roc_auc)
        if not tprs_all: continue
        mean_tpr = np.mean(tprs_all, axis=0);
        mean_tpr[-1] = 1.0
        mean_auc = auc_calc(base_fpr, mean_tpr);
        std_auc = np.std(aucs_all)
        plt.plot(base_fpr, mean_tpr, label=f'{label} (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--');
    plt.xlabel('FPR');
    plt.ylabel('TPR');
    plt.title(f'Combined Mean ROC Curves')
    plt.grid(True);
    plt.legend(loc="lower right")
    os.makedirs(results_path, exist_ok=True)
    plt.savefig(f"{results_path}/Combined_ROC_All_Comparisons_{file_suffix}.png", dpi=300, bbox_inches='tight');
    plt.close()

# Create the model
def get_model_and_params():
    svm = SVC(probability=True, random_state=42, tol=1e-3, class_weight='balanced')
    # Different params work best with each comparison - grid search chooses best
    params = {
        'selector__k': [500, 914, 'all'],
        'pca__n_components': [50, 75, 100],
        'model__C': [100, 500, 1000],
        'model__gamma': [0.003, 0.005, 0.001, 'scale']
    }
    return {"SVM_RBF": (svm, params)}


from sklearn.metrics import roc_auc_score as _roc_auc

# Test each model performance
def test_models(X_sub, y_sub, groups, save_path, pca_num_default, comparison_tag):
    kf = GroupKFold(n_splits=5)
    models_and_params = get_model_and_params()
    cv_results = []
    model_roc_data = {}

    for name, (base_model, param_grid) in models_and_params.items():
        print(f"\n--- Starting Grid Search & CV for {comparison_tag} ({name}) ---")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(f_classif)),
            ('pca', PCA(random_state=42)),
            ('model', base_model)
        ])

        # 'balanced_accuracy' to prioritizes MCI detection
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='balanced_accuracy',
                                   cv=list(kf.split(X_sub, y_sub, groups)), n_jobs=-1)
        grid_search.fit(X_sub, y_sub)

        best_params = {k.replace('model__', ''): v for k, v in grid_search.best_params_.items() if
                       k.startswith('model__')}
        best_k = grid_search.best_params_.get('selector__k', 'all')
        best_n_comp = grid_search.best_params_.get('pca__n_components', 50)

        print(f"\n===== {comparison_tag} — {name} =====")
        print(f"Best parameters: {grid_search.best_params_}")

        # Make sure class_weight = 'balanced'
        model_with_best = SVC(probability=True, random_state=42, tol=1e-3, class_weight='balanced', **best_params)
        pipeline_best = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(f_classif, k=best_k)),
            ('pca', PCA(n_components=best_n_comp, random_state=42)),
            ('model', model_with_best)
        ])

        trained_model_filename = f"{save_path}/{comparison_tag}"
        if not trained_model_filename.endswith(".pkl"): trained_model_filename += ".pkl"

        fold_metrics = [];
        fold_roc = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X_sub, y_sub, groups)):
            X_train, X_test = X_sub.iloc[train_idx], X_sub.iloc[test_idx]
            y_train, y_test = y_sub.iloc[train_idx], y_sub.iloc[test_idx]

            if ENABLE_COMBAT and NEUROHARMONIZE_AVAILABLE:
                try:
                    X_train = apply_combat(X_train, batch_vector=groups.iloc[train_idx].to_numpy())
                    X_test = apply_combat(X_test, batch_vector=groups.iloc[test_idx].to_numpy())
                except Exception:
                    pass

            if ENABLE_SMOTE:
                try:
                    sm = SMOTE(k_neighbors=SMOTE_K_NEIGHBORS, random_state=42)
                    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
                except Exception:
                    X_train_res, y_train_res = X_train, y_train
            else:
                X_train_res, y_train_res = X_train, y_train

            pipeline_best.fit(X_train_res, y_train_res)

            # Save intermediate
            try:
                joblib.dump(pipeline_best, trained_model_filename)
            except Exception:
                pass

            if ENABLE_CALIBRATION:
                try:
                    # CalibratedClassifierCV trained on base model
                    calibrated = CalibratedClassifierCV(estimator=pipeline_best, method=CALIBRATION_METHOD, cv=3)
                    calibrated.fit(X_train_res, y_train_res)
                    y_proba = calibrated.predict_proba(X_test)[:, 1]
                    y_pred = calibrated.predict(X_test)
                except Exception:
                    y_proba = pipeline_best.predict_proba(X_test)[:, 1]
                    y_pred = pipeline_best.predict(X_test)
            else:
                y_proba = pipeline_best.predict_proba(X_test)[:, 1]
                y_pred = pipeline_best.predict(X_test)

            fold_roc.append((y_test.to_numpy(), y_proba))

            try:
                acc = accuracy_score(y_test, y_pred)
                auc_val = _roc_auc(y_test, y_proba)
                # Use balanced accuracy for CV reporting
                b_acc = balanced_accuracy_score(y_test, y_pred)
                sens = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
            except Exception:
                acc = auc_val = sens = spec = f1 = tn = fp = fn = tp = 0

            fold_metrics.append(
                {'acc': acc, 'auc': auc_val, 'sens': sens, 'spec': spec, 'f1': f1, 'tn': tn, 'fp': fp, 'fn': fn,
                 'tp': tp})

        # Save final version
        try:
            joblib.dump(pipeline_best, trained_model_filename)
        except Exception:
            pass

        model_roc_data[name] = fold_roc
        df_metrics = pd.DataFrame(fold_metrics)
        mean_metrics = df_metrics.mean()
        std_metrics = df_metrics.std(ddof=1)
        cv_results.append((name, mean_metrics, std_metrics))

        # Print Results
        print("\nCV Results (Mean ± Std)")
        print("-" * 70)
        print(f"{'Metric':<12} {'Mean':>12} {'Std':>12}")
        print("-" * 70)

        metrics_order = ['acc', 'auc', 'sens', 'spec', 'f1', 'tn', 'fp', 'fn', 'tp']

        for m in metrics_order:
            mean_v = mean_metrics.get(m, 0)
            std_v = std_metrics.get(m, 0)
            print(f"{m:<12} {mean_v:>12.4f} {std_v:>12.4f}")

        print("-" * 70 + "\n")

    all_roc_data.append(model_roc_data)
    return cv_results, model_roc_data

# Plot the confusion matrix
def plot_confusion_matrix(cm, class_names, results_path, test_data):
    plt.figure(figsize=(8, 6));
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix");
    plt.colorbar()
    tick_marks = np.arange(len(class_names));
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2. if cm.max() != 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True Label');
    plt.xlabel('Predicted Label');
    plt.tight_layout()
    os.makedirs(results_path, exist_ok=True)
    plt.savefig(f"{results_path}/Confusion_Matrix_{test_data}.png", dpi=300, bbox_inches='tight');
    plt.close()

# Compare each class
def binary_compare(data_row_series, comparison_tag, pca_dim, results_path):
    pipeline_filename = f"{results_path}/{comparison_tag}"
    if not pipeline_filename.endswith(".pkl"): pipeline_filename += ".pkl"
    feature_list_filename = f"{results_path}/clean_feature_names.pkl"
    try:
        best_pipeline = joblib.load(pipeline_filename)
    except FileNotFoundError:
        return "MODEL NOT FOUND", "MODEL NOT FOUND"

    if isinstance(data_row_series, pd.Series):
        X_subject = data_row_series.to_frame().T
    else:
        X_subject = data_row_series.head(1).copy()

    try:
        clean_features = joblib.load(feature_list_filename)
    except Exception:
        clean_features = None

    if clean_features is not None:
        try:
            X_subject = X_subject[clean_features]
        except Exception:
            pass

    try:
        y_pred = best_pipeline.predict(X_subject)
        y_proba = best_pipeline.predict_proba(X_subject)
        # Assuming the pipeline always predicts the minority class as 1 in its internal binary mapping
        return int(y_pred[0]), float(y_proba[0][1])
    except Exception as e:
        return "PREDICTION ERROR", "PREDICTION ERROR"


diagnosis_to_code = {"NC": 1, "MCI": 2, "AD": 3, "Prediction Failed": 0}

# Determine final image diagnosis
def final_diagnosis_probabilistic(data_row_series, pca_dim, results_path):
    class_scores = {'NC': 0.0, 'MCI': 0.0, 'AD': 0.0}
    binary_results = {}
    comp_tags = ["NCvAD", "NCvMCI", "MCIvAD"]

    for tag in comp_tags:
        predicted_class_bin, probability_class_1 = binary_compare(data_row_series, tag, pca_dim, results_path)
        if isinstance(predicted_class_bin, str):
            binary_results[f'{tag}_Prediction'] = 0
            return "Prediction Failed", binary_results

        if tag == "NCvAD":
            pos, neg = 'AD', 'NC'
        elif tag == "NCvMCI":
            pos, neg = 'MCI', 'NC'
        elif tag == "MCIvAD":
            pos, neg = 'AD', 'MCI'

        # Probabilistic Logic
        prob_pos = probability_class_1
        prob_neg = 1.0 - probability_class_1

        # Boost MCI predicition logic
        if pos == 'MCI': prob_pos *= MCI_PROB_BOOST
        if neg == 'MCI': prob_neg *= MCI_PROB_BOOST

        # Get probabilities
        class_scores[pos] += prob_pos
        class_scores[neg] += prob_neg

        binary_results[f'{tag}_Prediction'] = diagnosis_to_code.get(pos if predicted_class_bin == 1 else neg, 0)
        binary_results[f'{tag}_Probability'] = probability_class_1

    # Final decision based on maximum accumulated score
    final = max(class_scores, key=class_scores.get)

    return final, binary_results

# Check the test accuracy
def check_accuracy(test_results_path, test_data):
    print("\n" + "=" * 50);
    print("Checking Multi-Subject Prediction Accuracy");
    print("=" * 50)
    try:
        df_results = pd.read_csv(f"{test_results_path}/Test_Prediction_{test_data}.csv")
    except FileNotFoundError:
        return
    df_clean = df_results.dropna(subset=['True_Diagnosis']).copy()
    y_true = df_clean['True_Diagnosis'].astype(int)
    y_pred = df_clean['Predicted_Diagnosis'].astype(int)

    diagnosis_map = {1: "NC", 2: "MCI", 3: "AD", 0: "Failed"}
    # Ensure all labels (1, 2, 3) are included for the matrix and report
    all_possible_labels = sorted(list(set(y_pred.unique()) | set(y_true.unique()) | {1, 2, 3}))
    target_names = [diagnosis_map.get(l, str(l)) for l in all_possible_labels if l != 0]

    # Filter y_true and y_pred to only include 1, 2, or 3
    valid_mask = y_true.isin([1, 2, 3]) & y_pred.isin([1, 2, 3])
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    # Use only the valid labels (1, 2, 3) for the report and matrix
    report = classification_report(y_true_valid, y_pred_valid, labels=[1, 2, 3], target_names=target_names,
                                   zero_division=0)
    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=[1, 2, 3])

    plot_confusion_matrix(cm, target_names, test_results_path, test_data)

    print("\nConfusion Matrix:")
    # Print the confusion matrix
    cm_str = '\n'.join(['\t'.join([str(val) for val in row]) for row in cm])
    print(cm_str)

    overall_accuracy = accuracy_score(y_true_valid, y_pred_valid)
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({len(y_true_valid)} samples)")
    print("\nClassification Report:");
    print(report)


if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    try:
        try:
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
        except NameError:
            pass

        # TRAINING DATA SECTION
        pca_num = 50
        train_csv = 'Datasets/ADNI-Oasis-AIBL_dataset.csv'
        df, data, filename = load_data(train_csv)
        results_path = f"Results/{data}/PCA - {pca_num}"
        test_csv = 'Datasets/Sample_Test_Data.csv'

        if "Subject" not in df.columns: raise ValueError("Dataset must contain 'Subject' column.")
        subjects = df["Subject"]
        df = df.drop(columns=["Image_ID", "Subject"], errors="ignore")
        df["Diagnosis"] = df["Diagnosis"].replace({"NC": 1, "MCI": 2, "AD": 3})
        X = df.drop(columns=["Diagnosis"]);
        print("Label distribution:", df["Diagnosis"].value_counts().to_dict())
        y = df["Diagnosis"]
        label_map = {1: "NC", 2: "MCI", 3: "AD"}

        comparisons = [(1, 2), (1, 3), (2, 3)]
        all_results = []

        for a, b in comparisons:
            print("\n" + "=" * 70)
            print(f"Classification Task: {label_map[a]} (0) vs {label_map[b]} (1)")
            print("=" * 70)

            mask = y.isin([a, b]);
            X_sub, y_sub = X[mask], y[mask]
            groups_sub = subjects[mask].reset_index(drop=True)
            if y_sub.value_counts().min() < 5: continue

            y_sub = y_sub.replace({a: 0, b: 1}).reset_index(drop=True)
            X_sub = X_sub.reset_index(drop=True)
            vt = VarianceThreshold(threshold=0.0)
            X_sub_clean = vt.fit_transform(X_sub)
            X_sub_clean_df = pd.DataFrame(X_sub_clean, columns=X_sub.columns[vt.get_support()])

            comp_tag = os.path.splitext(comparison_to_filename(a, b))[0]
            cv_res, roc_data = test_models(X_sub_clean_df, y_sub, groups_sub, results_path, pca_num, comp_tag)

            for res in cv_res:
                name, mean, std = res
                all_results.append([f"{label_map[a]} vs {label_map[b]}", pca_num, "Before Scaling", name,
                                    mean.get('acc'), std.get('acc'), mean.get('auc'), std.get('auc'),
                                    mean.get('sens'), std.get('sens'), mean.get('spec'), std.get('spec'),
                                    mean.get('f1'), std.get('f1')])

        if all_roc_data: plot_combined_mean_roc(all_roc_data, ["NC vs MCI", "NC vs AD", "MCI vs AD"], results_path,
                                                "All_ROC")

        summary_df = pd.DataFrame(all_results,
                                  columns=["Comparison", "PCA_Num", "PCA Stage", "Model", "Acc_Mean", "Acc_Std",
                                           "AUC_Mean", "AUC_Std", "Sens_Mean", "Sens_Std", "Spec_Mean", "Spec_Std",
                                           "F1_Mean", "F1_Std"])
        os.makedirs(results_path, exist_ok=True)
        summary_df.to_csv(f"{results_path}/model_comparison_summary.csv", index=False)

        # TESTING DATA SECTION
        test_data = get_data_name(test_csv)
        try:
            test_df = pd.read_csv(test_csv)
            test_results = []
            print("\n" + "#" * 50);
            print("Running Multi-Subject Diagnosis")
            for index, row in test_df.iterrows():
                final_pred_str, binary_comp_results = final_diagnosis_probabilistic(
                    row.drop(["Image_ID", "Subject", "Diagnosis"], errors="ignore"), pca_num, results_path)
                result_entry = {'Subject': row['Subject'], 'Image_ID': row.get('Image_ID', 'N/A'),
                                'Predicted_Diagnosis': diagnosis_to_code.get(final_pred_str, 0),
                                'True_Diagnosis': row.get('Diagnosis', 'N/A')}
                result_entry.update(binary_comp_results)
                test_results.append(result_entry)
                print(f"Subject {row['Subject']}: Predicted -> {final_pred_str}")

            test_results_df = pd.DataFrame(test_results)
            test_out_path = f"Results/Test Results/{data}"
            os.makedirs(test_out_path, exist_ok=True)
            test_results_df.to_csv(f"{test_out_path}/Test_Prediction_{test_data}.csv", index=False)
            check_accuracy(test_out_path, test_data)
        except FileNotFoundError:
            print(f"Error: Test file {test_csv} not found.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback;

        traceback.print_exc()