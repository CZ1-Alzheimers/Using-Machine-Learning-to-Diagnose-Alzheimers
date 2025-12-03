# Import libraries
import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import traceback

from collections import Counter

from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, StratifiedGroupKFold, train_test_split
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, auc as auc_calc,
    confusion_matrix, classification_report, f1_score, recall_score,
    balanced_accuracy_score
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
np.random.seed(42)

# Configuration Settings
ENABLE_SMOTE = True
SMOTE_K_NEIGHBORS = 5
ENABLE_CALIBRATION = True
CALIBRATION_METHOD = "sigmoid"

# MCI boosting
MCI_FINAL_BOOST = 1.01  # Multiplies final MCI probability
MCI_DECISION_OFFSET = 0.18  # Adds raw probability to MCI in binary pairs

RESULTS_ROOT = "Results"
RANDOM_STATE = 42
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3
all_roc_data = []  # Stores ROC data for combined plot

# Global Mappings
diagnosis_to_code = {"NC": 1, "MCI": 2, "AD": 3, "Prediction Failed": 0}
code_to_diag = {v: k for k, v in diagnosis_to_code.items()}


# Safe SelectKBest
class SafeSelectKBest(BaseEstimator, TransformerMixin):
    # Selects K Best features
    def __init__(self, score_func=f_classif, k=10):
        self.score_func = score_func
        self.k = k
        self.selector = None

    def fit(self, X, y):
        k_clip = min(self.k, X.shape[1])
        self.selector = SelectKBest(self.score_func, k=k_clip)
        self.selector.fit(X, y)
        return self

    def transform(self, X):
        return self.selector.transform(X)

    def get_support(self):
        return self.selector.get_support() if self.selector is not None else None



# Plotting
# Plot ROC for each comparison
def plot_mean_roc_curves(model_roc_data, results_path, file_suffix):

    if not model_roc_data:
        return
    os.makedirs(results_path, exist_ok=True)
    base_fpr = np.linspace(0, 1, 101)  # V9 uses 101 points

    plt.figure(figsize=(10, 8))
    for name, fold_data in model_roc_data.items():
        tprs = []
        aucs = []
        if not fold_data: continue

        for y_test, y_proba in fold_data:
            if len(np.unique(y_test)) < 2: continue
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc_calc(fpr, tpr)
            tpr_interp = np.interp(base_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)
            aucs.append(roc_auc)

        if not tprs: continue
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc_calc(base_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(base_fpr, mean_tpr, label=f'{name} (AUC = {mean_auc:.3f} $\\pm$ {std_auc:.3f})')
        # Use latex for +/- symbol

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Mean ROC Curves')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Save the file
    plt.savefig(f"{results_path}/Mean_ROC_Curves_{file_suffix}.png", dpi=300, bbox_inches='tight')
    plt.close()


# Combine all ROCs into one plot
def plot_combined_mean_roc(all_roc_dicts, labels, results_path, file_suffix):
    plt.figure(figsize=(12, 10))
    base_fpr = np.linspace(0, 1, 101)  # V9 uses 101 points
    for roc_dict, label in zip(all_roc_dicts, labels):
        tprs_all = []
        aucs_all = []
        for model_name, fold_data in roc_dict.items():
            for y_test, y_proba in fold_data:
                if len(np.unique(y_test)) < 2: continue
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc_calc(fpr, tpr)
                tpr_interp = np.interp(base_fpr, fpr, tpr)
                tpr_interp[0] = 0.0
                tprs_all.append(tpr_interp)
                aucs_all.append(roc_auc)
        if not tprs_all: continue
        mean_tpr = np.mean(tprs_all, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc_calc(base_fpr, mean_tpr)
        std_auc = np.std(aucs_all)
        plt.plot(base_fpr, mean_tpr, label=f'{label} (AUC = {mean_auc:.3f} $\\pm$ {std_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'Combined Mean ROC Curves')
    plt.grid(True)
    plt.legend(loc="lower right")
    os.makedirs(results_path, exist_ok=True)
    plt.savefig(f"{results_path}/Combined_ROC_All_Comparisons_{file_suffix}.png", dpi=300, bbox_inches='tight')
    plt.close()


# Plot the confusion matrix
def plot_confusion_matrix(cm, class_names, results_path, test_data):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2. if cm.max() != 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    os.makedirs(results_path, exist_ok=True)
    # Save figure
    plt.savefig(f"{results_path}/Confusion_Matrix_{test_data}.png", dpi=300, bbox_inches='tight')
    plt.close()


# Build model and grid search
def get_model_and_params():
    # Model parameters for Grid Search
    svm = SVC(
        probability=True,
        random_state=RANDOM_STATE
    )

    params = {
        "pca__n_components": [50, 80, 150, 200, 250, 300],
        "selector__k": [30, 60, 90, 120],
        "model__C": [10, 50, 100],
        "model__gamma": [0.01, 0.001, "scale"],
        # Custom weight boosts MCI
        "model__class_weight": ["balanced", {1: 1.5, 0: 1.0}, {1: 2.0, 0: 1.0}, {1: 3.0, 0: 1.0}]
    }
    return {"SVM_RBF": (svm, params)}


def clip_components(n, max_features):
    return min(max(1, n), max_features)


# Cross validation
def test_models(X_sub, y_sub, groups, save_path, pca_num_default, comparison_tag):

    skf_outer = StratifiedGroupKFold(
        n_splits=N_OUTER_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    cv_results = []
    model_roc_data = {"SVM_RBF": []}
    best_params_list = []
    models_and_params = get_model_and_params()

    fold = 0
    for train_idx, test_idx in skf_outer.split(X_sub, y_sub, groups):
        X_train = X_sub.iloc[train_idx]
        y_train = y_sub.iloc[test_idx].to_numpy()
        X_test = X_sub.iloc[test_idx]
        y_test = y_sub.iloc[test_idx].to_numpy()

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        y_train = y_sub.iloc[train_idx].to_numpy()

        if ENABLE_SMOTE:
            try:
                sm = SMOTE(k_neighbors=SMOTE_K_NEIGHBORS, random_state=RANDOM_STATE)
                X_res, y_res = sm.fit_resample(X_train_s, y_train)
            except:
                X_res, y_res = X_train_s, y_train
        else:
            X_res, y_res = X_train_s, y_train

        nfeat = X_res.shape[1]

        for name, (base_model, param_grid) in models_and_params.items():

            adj_grid = {}
            for key, vals in param_grid.items():
                if key == "pca__n_components":
                    adj_grid[key] = sorted({
                        clip_components(v, nfeat) for v in vals
                    })
                else:
                    adj_grid[key] = vals

            pipeline = Pipeline([
                ("pca", PCA(random_state=RANDOM_STATE)),
                ("selector", SafeSelectKBest(f_classif)),
                ("model", base_model)
            ])

            inner = StratifiedKFold(
                n_splits=N_INNER_FOLDS,
                shuffle=True,
                random_state=RANDOM_STATE
            )

            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=adj_grid,
                scoring="balanced_accuracy",
                cv=inner,
                n_jobs=-1,
                refit=True
            )

            grid_search.fit(X_res, y_res)

            best_params = grid_search.best_params_
            best_params_list.append(best_params)

            # Recreate final model based on overall best parameters
            pca_n = best_params["pca__n_components"]
            sel_k = best_params["selector__k"]
            C = best_params["model__C"]
            gamma = best_params["model__gamma"]
            cw = best_params["model__class_weight"]

            # Perform PCA and Selection outside the pipeline for final SVC fit
            pca = PCA(n_components=pca_n, random_state=RANDOM_STATE)
            X_res_pca = pca.fit_transform(X_res)
            X_test_pca = pca.transform(X_test_s)

            selector = SafeSelectKBest(f_classif, k=sel_k)
            selector.fit(X_res_pca, y_res)
            X_res_sel = selector.transform(X_res_pca)
            X_test_sel = selector.transform(X_test_pca)

            model_with_best = SVC(
                C=C, gamma=gamma,
                probability=True,
                class_weight=cw,
                random_state=RANDOM_STATE
            )

            sw = None
            if cw == 'balanced':
                sw = compute_sample_weight("balanced", y_res)

            if sw is not None:
                model_with_best.fit(X_res_sel, y_res, sample_weight=sw)
            else:
                model_with_best.fit(X_res_sel, y_res)

            # Prediction with calibration
            if ENABLE_CALIBRATION:
                try:
                    calibrated = CalibratedClassifierCV(estimator=model_with_best, method=CALIBRATION_METHOD, cv=3)
                    # Train calibration on the selected/PCA features
                    calibrated.fit(X_res_sel, y_res)
                    y_proba = calibrated.predict_proba(X_test_sel)[:, 1]
                    y_pred = calibrated.predict(X_test_sel)
                except Exception:
                    y_proba = model_with_best.predict_proba(X_test_sel)[:, 1]
                    y_pred = model_with_best.predict(X_test_sel)
            else:
                y_proba = model_with_best.predict_proba(X_test_sel)[:, 1]
                y_pred = model_with_best.predict(X_test_sel)

            fold_roc = model_roc_data.get(name, [])
            fold_roc.append((y_test, y_proba))
            model_roc_data[name] = fold_roc

            # Metrics Calculation
            try:
                acc = accuracy_score(y_test, y_pred)
                auc_val = roc_auc_score(y_test, y_proba)
                b_acc = balanced_accuracy_score(y_test, y_pred)  # V9 records this internally
                sens = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
            except Exception:
                acc = auc_val = sens = spec = f1 = tn = fp = fn = tp = 0

            fold_metrics = [
                {'acc': acc, 'auc': auc_val, 'sens': sens, 'spec': spec, 'f1': f1, 'tn': tn, 'fp': fp, 'fn': fn,
                 'tp': tp}]
            df_metrics = pd.DataFrame(fold_metrics)

            cv_results.append((name, df_metrics.iloc[0]))  # Store the single row's metrics as a Series
            fold += 1  # Increment fold counter

    name = "SVM_RBF"  # Model name

    # Extract the raw metrics Series for the correct model name from all folds
    fold_metrics_list = [res[1] for res in cv_results if res[0] == name]

    # Calculate mean and std
    if fold_metrics_list:
        df_all_folds = pd.DataFrame(fold_metrics_list)
        mean_metrics = df_all_folds.mean()
        std_metrics = df_all_folds.std(ddof=1)
        results_to_return = [(name, mean_metrics, std_metrics)]
    else:
        results_to_return = []

    # Return results for final model training and ROC plotting
    return results_to_return, model_roc_data, best_params_list


# Train final model and calibrate it
def train_final_and_calibrate(X_df, y_series, best_params_list, out_dir, name_tag):
    if not best_params_list:
        raise ValueError("best_params_list empty")

    # Consensus parameter selection
    keys = set().union(*[set(d.keys()) for d in best_params_list])
    consensus = {}
    for k in keys:
        vals = [str(d[k]) for d in best_params_list if k in d]
        mc = Counter(vals).most_common(1)[0][0]
        # Find most common key
        best_val = None
        for d in best_params_list:
            if k in d and str(d[k]) == mc:
                best_val = d[k]
                break
        consensus[k] = best_val

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    if ENABLE_SMOTE:
        try:
            sm = SMOTE(k_neighbors=SMOTE_K_NEIGHBORS, random_state=RANDOM_STATE)
            X_res, y_res = sm.fit_resample(X_scaled, y_series)
        except Exception:
            X_res, y_res = X_scaled, y_series.values
    else:
        X_res, y_res = X_scaled, y_series.values

    # Extract consensus parameters
    pca_n = consensus.get('pca__n_components', min(100, X_res.shape[1]))
    sel_k = consensus.get('selector__k', min(50, X_res.shape[1]))
    C = consensus.get('model__C', 100)
    gamma = consensus.get('model__gamma', 'scale')
    cw = consensus.get('model__class_weight', 'balanced')

    # Final model training pipeline components
    pca = PCA(n_components=clip_components(pca_n, X_res.shape[1]), random_state=RANDOM_STATE)
    X_res_pca = pca.fit_transform(X_res)

    selector = SafeSelectKBest(f_classif, k=sel_k)
    selector.fit(X_res_pca, y_res)
    X_res_sel = selector.transform(X_res_pca)

    final_svc = SVC(C=C, gamma=gamma, probability=True,
                    class_weight=cw, random_state=RANDOM_STATE)

    sw = None
    if cw == 'balanced':
        sw = compute_sample_weight('balanced', y_res)

    if sw is not None:
        final_svc.fit(X_res_sel, y_res, sample_weight=sw)
    else:
        final_svc.fit(X_res_sel, y_res)

    # Calibration
    if ENABLE_CALIBRATION:
        calibrated = CalibratedClassifierCV(final_svc, method=CALIBRATION_METHOD, cv=3)
        calibrated.fit(X_res_sel, y_res)
    else:
        calibrated = final_svc  # Use base model if calibration disabled

    bundle = {
        'scaler': scaler, 'pca': pca,
        'selector': selector, 'calibrated_clf': calibrated,
        'consensus': consensus
    }


    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"{name_tag}.pkl")
    joblib.dump(bundle, model_path)

    # Return consensus for printing outside
    return bundle, model_path, consensus


# Helper function for binary prediction
def binary_predict_subject(bundle, data_row_series):
    scaler = bundle['scaler']
    pca = bundle['pca']
    selector = bundle['selector']
    clf = bundle['calibrated_clf']

    try:
        if isinstance(data_row_series, pd.Series):
            X_sub = data_row_series.to_frame().T
        else:
            X_sub = pd.DataFrame([data_row_series])

        X_scaled = scaler.transform(X_sub)
        X_pca = pca.transform(X_scaled)
        X_sel = selector.transform(X_pca)

        y_pred = int(clf.predict(X_sel)[0])
        y_proba = clf.predict_proba(X_sel)[0][1]  # Probability of class 1

        return y_pred, y_proba
    except Exception as e:
        return "PREDICTION ERROR", "PREDICTION ERROR"


# Prediction and coupling
def binary_compare(data_row_series, comparison_tag, pca_dim, results_path):
    pipeline_filename = f"{results_path}/{comparison_tag}.pkl"

    try:
        model_bundle = joblib.load(pipeline_filename)
    except FileNotFoundError:
        return "MODEL NOT FOUND", 0.5  # Return 0.5 as neutral probability
    except Exception:
        return "MODEL LOAD ERROR", 0.5

    pred_bin, prob_pos = binary_predict_subject(model_bundle, data_row_series)

    if isinstance(pred_bin, str):
        return pred_bin, 0.5  # Handle prediction error

    return pred_bin, prob_pos

# Determine final diagnosis
def final_diagnosis_probabilistic(data_row_series, pca_dim, results_root):
    classes = ['NC', 'MCI', 'AD']
    idx = {c: i for i, c in enumerate(classes)}
    binary_results = {}
    comp_tags = ['NCvMCI', 'NCvAD', 'MCIvAD']
    k = 3

    # Load all models
    loaded_bundles = {}
    for tag in comp_tags:
        global MODEL_LOAD_ROOT
        load_dir = MODEL_LOAD_ROOT if 'MODEL_LOAD_ROOT' in globals() else results_root  # Fallback
        model_path = os.path.join(load_dir, f"{tag}.pkl")
        try:
            loaded_bundles[tag] = joblib.load(model_path)
        except Exception:
            binary_results[f'{tag}_Prediction'] = 0
            binary_results[f'{tag}_Probability'] = 0.5
            return "Prediction Failed", binary_results  # Fail if any model is missing/corrupt

    # Boundary shifting
    r = np.zeros((3, 3), dtype=float)
    tag_map = {'NCvMCI': ('NC', 'MCI'), 'NCvAD': ('NC', 'AD'), 'MCIvAD': ('MCI', 'AD')}

    for tag, (neg, pos) in tag_map.items():
        bundle = loaded_bundles.get(tag)
        pred_bin, prob_pos = binary_predict_subject(bundle, data_row_series)
        if isinstance(pred_bin, str):
            binary_results[f'{tag}_Prediction'] = 0
            binary_results[f'{tag}_Probability'] = 0.5
            return "Prediction Failed", binary_results

        prob_for_r = prob_pos
        # Decision Boundary Shifting
        if tag == "NCvMCI":
            prob_for_r = prob_for_r + MCI_DECISION_OFFSET
        elif tag == "MCIvAD":
            prob_for_r = prob_for_r - MCI_DECISION_OFFSET

        prob_for_r = float(np.clip(prob_for_r, 0.001, 0.999))  # Clip to avoid division by zero in coupling

        # Fill in r matrix for coupling
        i_pos = idx[pos]
        i_neg = idx[neg]
        r[i_pos, i_neg] = prob_for_r
        r[i_neg, i_pos] = 1.0 - prob_for_r

        # Store results
        binary_results[f'{tag}_Prediction'] = diagnosis_to_code.get(pos if pred_bin == 1 else neg, 0)
        binary_results[f'{tag}_Probability'] = round(prob_pos, 4)  # Use original probability for output

    for i in range(k):
        for j in range(k):
            if i == j:
                r[i, j] = 0.0
            elif r[i, j] == 0 and r[j, i] == 0:
                r[i, j] = 0.5;
                r[j, i] = 0.5

    # Pairwise Coupling
    try:
        p = pairwise_coupling(r)
    except Exception:
        # Fallback if coupling fails
        scores = r.sum(axis=1)
        p = scores / scores.sum() if scores.sum() > 0 else np.ones(k) / k

    probs = {classes[i]: float(p[i]) for i in range(k)}

    # Final boost to MCI
    if 'MCI' in probs:
        probs['MCI'] *= MCI_FINAL_BOOST

    total = sum(probs.values())
    if total > 0:
        for c in probs: probs[c] /= total

    final_label = max(probs.items(), key=lambda kv: kv[1])[0]

    return final_label, binary_results


def pairwise_coupling(r, max_iter=100, eps=1e-6):
    r = np.array(r, dtype=float)
    k = r.shape[0]
    p = np.ones(k, dtype=float) / k

    for it in range(max_iter):
        q = np.zeros((k, k), dtype=float)
        for i in range(k):
            for j in range(k):
                if i == j: continue
                denom = p[i] + p[j]
                q[i, j] = (p[i] / denom) if denom > 0 else 0.5

        f = np.zeros(k, dtype=float)
        for i in range(k):
            f[i] = np.sum(q[i, :] - r[i, :])

        if np.max(np.abs(f)) < eps: break

        H = np.zeros((k, k), dtype=float)
        for i in range(k):
            for j in range(k):
                if i == j: continue
                denom = (p[i] + p[j]) ** 2 if (p[i] + p[j]) != 0 else 1.0
                H[i, i] += p[j] / denom
                H[i, j] -= p[i] / denom

        ridge = 1e-8
        try:
            H_reg = H.copy()
            H_reg[np.diag_indices_from(H_reg)] += ridge
            delta = np.linalg.solve(H_reg, -f)
        except np.linalg.LinAlgError:
            delta = -(0.1 / (it + 1)) * f

        p += delta
        p = np.maximum(p, 1e-12)
        p = p / p.sum()

    return p


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


# Check the test accuracy
def check_accuracy(test_results_path, test_data, source_data_tag):
    test_prediction_file = f"{test_results_path}/Test_Prediction_{test_data}.csv"

    try:
        df_results = pd.read_csv(test_prediction_file)
    except FileNotFoundError:
        print(f"Error: Test prediction file not found at {test_prediction_file}")
        return

    df_clean = df_results.dropna(subset=['True_Diagnosis', 'Predicted_Diagnosis']).copy()
    y_true = pd.to_numeric(df_clean['True_Diagnosis'], errors='coerce')
    y_pred = pd.to_numeric(df_clean['Predicted_Diagnosis'], errors='coerce')

    diagnosis_map = {1: "NC", 2: "MCI", 3: "AD", 0: "Failed"}
    target_names = ["NC", "MCI", "AD"]

    # Filter y_true and y_pred to only include 1, 2, or 3
    valid_mask = y_true.isin([1, 2, 3]) & y_pred.isin([1, 2, 3])
    y_true_valid = y_true[valid_mask].astype(int)
    y_pred_valid = y_pred[valid_mask].astype(int)

    if len(y_true_valid) == 0:
        print("No valid predictions (NC, MCI, or AD) found to calculate accuracy.")
        return

    # Generate classification report
    report = classification_report(y_true_valid, y_pred_valid, labels=[1, 2, 3], target_names=target_names,
                                   zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=[1, 2, 3])
    overall_accuracy = accuracy_score(y_true_valid, y_pred_valid)

    cm_plot_dir = os.path.join(RESULTS_ROOT, "Test Results", source_data_tag)
    plot_confusion_matrix(cm, target_names, cm_plot_dir, test_data)
    cm_plot_path = os.path.join(cm_plot_dir, f"Confusion_Matrix_{test_data}.png")

    print("\n" + "=" * 50)
    print("Checking Multi-Subject Prediction Accuracy")
    print("=" * 50)

    print(f"Saved confusion matrix plot to: {cm_plot_path}")
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({len(y_true_valid)} samples)")

    print("\nClassification Report:")

    # Header alignment
    print(" " * 12 + "precision" + " " * 3 + "recall" + " " * 3 + "f1-score" + " " * 3 + "support")
    print()

    # Data rows (NC, MCI, AD)
    for i, name in enumerate(target_names):
        metrics = report.get(name)
        if metrics:
            p = metrics['precision']
            r = metrics['recall']
            f1 = metrics['f1-score']
            s = metrics['support']
            print(f" {name:>10}\t\t{p:7.2f}\t{r:7.2f}\t{f1:7.2f}\t\t{int(s):5d}")

    print()

    # Footer
    acc = report['accuracy']
    total_support = report['macro avg']['support']

    # Accuracy row
    print(f"  {'accuracy':<10}\t\t\t\t\t\t{acc:7.2f}\t\t{total_support}")

    # Macro avg
    macro_p = report['macro avg']['precision']
    macro_r = report['macro avg']['recall']
    macro_f1 = report['macro avg']['f1-score']
    print(f"  {'macro avg':<10}\t{macro_p:7.2f}\t{macro_r:7.2f}\t{macro_f1:7.2f}\t\t{total_support}")

    # Weighted avg
    weighted_p = report['weighted avg']['precision']
    weighted_r = report['weighted avg']['recall']
    weighted_f1 = report['weighted avg']['f1-score']
    print(f"  {'weighted avg':<10}\t{weighted_p:7.2f}\t{weighted_r:7.2f}\t{weighted_f1:7.2f}\t\t{total_support}")

    # Confusion Matrix formatting
    print("\nConfusion Matrix:")

    # Header
    print("\t\tPredicted NC\tPredicted MCI\tPredicted AD")

    # Rows
    for i, label in enumerate(['True NC', 'True MCI', 'True AD']):
        # Using fixed-width alignment
        print(f"{label:<11}{cm[i, 0]:12d}{cm[i, 1]:13d}{cm[i, 2]:12d}")

if __name__ == "__main__":
    try:
        try:
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
        except NameError:
            pass

        # Training section data
        pca_num = 50
        train_csv = 'Datasets/ADNI-Oasis-AIBL_dataset.csv'
        df, data, filename = load_data(train_csv)
        results_path = f"Results/{data}/PCA - {pca_num}"

        # Global for loading models
        global MODEL_LOAD_ROOT
        MODEL_LOAD_ROOT = results_path

        if "Subject" not in df.columns:
            if 'Image_ID' in df.columns:
                df['Subject'] = df['Image_ID']
            else:
                df['Subject'] = np.arange(df.shape[0])

        subjects = df["Subject"]

        # Drop Subject and Image_ID BEFORE splitting X and y
        X = df.drop(columns=["Diagnosis", "Image_ID", "Subject"], errors="ignore")
        df["Diagnosis"] = df["Diagnosis"].replace({"NC": 1, "MCI": 2, "AD": 3})
        y = df["Diagnosis"]
        label_map = {1: "NC", 2: "MCI", 3: "AD"}
        print("Initial Label distribution:", y.value_counts().to_dict())

        # 80% traing, 20% testing split
        df_full = X.copy()
        df_full['Diagnosis'] = y
        df_full['Subject'] = subjects

        df_train, df_test = train_test_split(df_full, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

        # Extract training and testing sets
        X_train = df_train.drop(columns=["Diagnosis", "Subject"])
        y_train = df_train["Diagnosis"]
        subjects_train = df_train["Subject"]

        X_test = df_test.drop(columns=["Diagnosis", "Subject"])
        y_test = df_test["Diagnosis"]
        subjects_test = df_test["Subject"]

        X = X_train
        y = y_train
        subjects = subjects_train

        print(f"Training set size: {len(y)} samples. Label distribution: {y.value_counts().to_dict()}")
        print(f"Test set size: {len(y_test)} samples. Label distribution: {y_test.value_counts().to_dict()}")

        comparisons = [(1, 2), (1, 3), (2, 3)]
        all_results = []
        all_final_models = {}  # Store model paths for testing

        for a, b in comparisons:
            comp_name = f"{label_map[a]}v{label_map[b]}"

            print("\n" + "=" * 70)
            print(f"Classification Task: {label_map[a]} (0) vs {label_map[b]} (1)")
            print("=" * 70)

            mask = y.isin([a, b])
            X_sub = X[mask].reset_index(drop=True)
            y_sub = y[mask].replace({a: 0, b: 1}).reset_index(drop=True)

            groups_sub = subjects[mask].reset_index(drop=True)

            if y_sub.value_counts().min() < 5: continue

            vt = VarianceThreshold(threshold=0.0)
            X_sub_clean_df = pd.DataFrame(vt.fit_transform(X_sub), columns=X_sub.columns[vt.get_support()])

            comp_tag = os.path.splitext(f"{label_map[a]}v{label_map[b]}")[
                0]

            cv_res, roc_data, best_params_list = test_models(X_sub_clean_df, y_sub,
                                                             groups_sub, results_path,
                                                             pca_num, comp_tag)

            if not cv_res:
                print(f"Skipping {comp_tag} due to insufficient data or error in CV.")
                continue

            # Consensus Parameters
            keys = set().union(*[set(d.keys()) for d in best_params_list])
            consensus = {}
            for k in keys:
                vals = [str(d[k]) for d in best_params_list if k in d]
                mc = Counter(vals).most_common(1)[0][0]
                best_val = None
                for d in best_params_list:
                    if k in d and str(d[k]) == mc:
                        best_val = d[k]
                        break
                consensus[k] = best_val

            # Extract CV Results
            name = cv_res[0][0]  # "SVM_RBF"
            mean_metrics = cv_res[0][1]
            std_metrics = cv_res[0][2]

            model_out_dir = results_path
            model_bundle, model_path, _ = train_final_and_calibrate(X_sub_clean_df, y_sub, best_params_list,
                                                                    model_out_dir, comp_tag)

            # Print summary
            N_FOLDS = N_OUTER_FOLDS
            ci_factor = 1.96 / np.sqrt(N_FOLDS)

            final_params = {k.replace('model__', '').replace('selector__', '').replace('pca__', ''): v
                            for k, v in consensus.items()}
            final_params['kernel'] = 'rbf'

            print(f"\n--- Starting Grid Search & {N_FOLDS}-Fold CV for {name} ---")
            print(f"Best parameters for {name}: {final_params}")

            # Print Model Save Path
            print(f"Saved trained pipeline to: {model_path}")

            print(f"{name} {N_FOLDS}-Fold CV Results (Mean (Std) [95% CI]):")

            metrics_to_print = {
                "Accuracy": 'acc', "AUC": 'auc', "Sensitivity": 'sens',
                "Specificity": 'spec', "F1-score": 'f1'
            }

            # Find max length for alignment
            max_len = max(len(k) for k in metrics_to_print.keys())

            # Print metrics
            for label, metric_key in metrics_to_print.items():
                mean_v = mean_metrics.get(metric_key, 0.0)
                std_v = std_metrics.get(metric_key, 0.0)
                ci_err = ci_factor * std_v

                ci_low = mean_v - ci_err
                ci_high = mean_v + ci_err

                print(f"  {label:<{max_len + 1}}: {mean_v:>8.3f} ({std_v:>5.3f}) [{ci_low:5.3f} - {ci_high:5.3f}]")

            # Mean CM uses mean of TP, TN, FP, FN from cv_results
            mean_tp = mean_metrics.get('tp', 0.0)
            mean_tn = mean_metrics.get('tn', 0.0)
            mean_fp = mean_metrics.get('fp', 0.0)
            mean_fn = mean_metrics.get('fn', 0.0)

            print(
                f"  {'Mean CM':<{max_len + 1}}: TP={mean_tp:.1f}, TN={mean_tn:.1f}, FP={mean_fp:.1f}, FN={mean_fn:.1f}")

            # Print ROC plot save path
            roc_path_dir = results_path
            roc_path = os.path.join(roc_path_dir, f"Mean_ROC_Curves_{comp_tag}.png")
            print(f"Saved mean ROC plot to")
            print(f" {roc_path}")

            all_final_models[comp_tag] = model_path

            # Plots ROC per comparison
            plot_mean_roc_curves(roc_data, roc_path_dir, comp_tag)

            all_results.append([f"{label_map[a]} vs {label_map[b]}", pca_num, "Post Scaling", name,
                                mean_metrics.get('acc'), std_metrics.get('acc'), mean_metrics.get('auc'),
                                std_metrics.get('auc'),
                                mean_metrics.get('sens'), std_metrics.get('sens'), mean_metrics.get('spec'),
                                std_metrics.get('spec'),
                                mean_metrics.get('f1'), std_metrics.get('f1')])

            all_roc_data.append(roc_data)  # Store for combined plot

        if all_roc_data:
            plot_combined_mean_roc(all_roc_data, [f"{label_map[c[0]]} vs {label_map[c[1]]}" for c in comparisons],
                                   results_path, "All_ROC")

        # Save summary
        summary_df = pd.DataFrame(all_results,
                                  columns=["Comparison", "PCA_Num", "PCA Stage", "Model", "Acc_Mean", "Acc_Std",
                                           "AUC_Mean", "AUC_Std", "Sens_Mean", "Sens_Std", "Spec_Mean", "Spec_Std",
                                           "F1_Mean", "F1_Std"])
        os.makedirs(results_path, exist_ok=True)

        # Output file
        summary_df.to_csv(f"{results_path}/model_comparison_summary.csv", index=False)
        print(f"\n{results_path}/model_comparison_summary.csv")  # Display output file

        # Multi-class prediction on test set
        test_results_path = os.path.join(RESULTS_ROOT, "Test Results", data)
        os.makedirs(test_results_path, exist_ok=True)
        test_prediction_file = os.path.join(test_results_path, f"Test_Prediction_{data}.csv")

        test_data_tag = data

        print("\n" + "=" * 50)
        print(f"Starting Final Multi-Class Prediction on {test_data_tag} set...")
        print("=" * 50)

        all_predictions = []
        for index, row in X_test.iterrows():
            final_label, binary_results = final_diagnosis_probabilistic(row, pca_num, results_path)

            # Map predicted diagnosis back to number
            predicted_code = diagnosis_to_code.get(final_label, 0)
            true_code = y_test.loc[index]

            row_result = {
                'Subject_ID': subjects_test.loc[index],
                'True_Diagnosis': true_code,
                'True_Diagnosis_Label': label_map.get(true_code, 'N/A'),
                'Predicted_Diagnosis': predicted_code,
                'Predicted_Diagnosis_Label': final_label,
                **binary_results
            }
            all_predictions.append(row_result)

        df_predictions = pd.DataFrame(all_predictions)
        df_predictions.to_csv(test_prediction_file, index=False)
        print(f"Saved test predictions to: {test_prediction_file}")

        # Print the confusion matrix and overall metrics
        check_accuracy(test_results_path, test_data_tag, data)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()