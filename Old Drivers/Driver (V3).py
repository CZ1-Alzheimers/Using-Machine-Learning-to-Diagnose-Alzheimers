import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from sklearn.model_selection import KFold, GridSearchCV
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
from sklearn.model_selection import GroupKFold

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping

# errors when n_jobs > 1 (used in GridSearchCV).
matplotlib.use('Agg')

# Ensure reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Build Neural Network
def build_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=RMSprop(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Evaluation Function
def evaluate_model(y_true, y_pred, y_proba):
    y_proba_pos = y_proba.flatten()

    if len(np.unique(y_true)) < 2:
        return 0.0, 0.5, 0.0, 0.0, 0.0  # Return default safe values

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except ValueError:
        return 0.0, 0.5, 0.0, 0.0, 0.0  # Fallback for odd cases

    accuracy = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_proba_pos)
    except ValueError:
        auc = -1

    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return accuracy, auc, sensitivity, specificity, f1

# Test classical models
def test_classical_models_cv(X_sub, y_sub, groups):
    from sklearn.model_selection import StratifiedKFold
    # 5-Fold Group Cross-Validation setup to ensure both classes in each fold
    # GroupKFold keeps same-subject images together
    kf = GroupKFold(n_splits=5)

    # Models and their parameter grids
    models_and_params = {
        "KNN": (KNeighborsClassifier(),
                {'n_neighbors': [3, 5, 7, 9]}),
        "RandomForest": (RandomForestClassifier(random_state=42),
                         {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}),
        "SVM": (SVC(probability=True, random_state=42),
                {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'tol': [1e-3, 1e-4]}),
        "LogisticRegression": (LogisticRegression(max_iter=1000, random_state=42),
                               {'C': [0.1, 1, 10], 'penalty': ['l2']})
    }

    cv_results = []

    for name, (model, param_grid) in models_and_params.items():
        print(f"\n--- Starting Grid Search & 5-Fold CV for {name} ---")

        # Grid Search with 5-fold cross-validation
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=kf.split(X_sub, y_sub, groups),
            verbose=0,  # Set to 1 for more detail
            n_jobs=-1
        )

        grid_search.fit(X_sub, y_sub)
        best_model = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")

        # Evaluate the best model on all 5 folds
        all_fold_metrics = []
        for fold, (train_index, test_index) in enumerate(kf.split(X_sub, y_sub, groups)):
            X_train_fold, X_test_fold = X_sub.iloc[train_index], X_sub.iloc[test_index]
            y_train_fold, y_test_fold = y_sub.iloc[train_index], y_sub.iloc[test_index]

            best_model.fit(X_train_fold, y_train_fold)
            y_pred_fold = best_model.predict(X_test_fold)

            # Robust handling of predict_proba
            if hasattr(best_model, "predict_proba"):
                try:
                    y_proba_full = best_model.predict_proba(X_test_fold)
                    if y_proba_full.shape[1] == 2:
                        y_proba_fold = y_proba_full[:, 1]
                    else:
                        # Only one class seen during training
                        y_proba_fold = y_pred_fold.astype(float)
                except Exception:
                    y_proba_fold = y_pred_fold.astype(float)
            else:
                y_proba_fold = y_pred_fold.astype(float)

            acc, auc, sens, spec, f1 = evaluate_model(y_test_fold, y_pred_fold, y_proba_fold)
            all_fold_metrics.append({'acc': acc, 'auc': auc, 'sens': sens, 'spec': spec, 'f1': f1})

        # Calculate mean and std deviation for each metric
        df_metrics = pd.DataFrame(all_fold_metrics)
        mean_metrics = df_metrics.mean()
        std_metrics = df_metrics.std(ddof=1)

        cv_results.append((name, mean_metrics['acc'], std_metrics['acc'],
                           mean_metrics['auc'], std_metrics['auc'],
                           mean_metrics['sens'], std_metrics['sens'],
                           mean_metrics['spec'], std_metrics['spec'],
                           mean_metrics['f1'], std_metrics['f1']))

        print(f"{name} 5-Fold CV Results (Mean (Std)):")
        print(f"  Accuracy:    {mean_metrics['acc']:.3f} ({std_metrics['acc']:.3f})")
        print(f"  AUC:         {mean_metrics['auc']:.3f} ({std_metrics['auc']:.3f})")
        print(f"  Sensitivity: {mean_metrics['sens']:.3f} ({std_metrics['sens']:.3f})")
        print(f"  Specificity: {mean_metrics['spec']:.3f} ({std_metrics['spec']:.3f})")
        print(f"  F1-score:    {mean_metrics['f1']:.3f} ({std_metrics['f1']:.3f})")

    return cv_results

# PCA / Model Training
def run_pca_and_models_cv(stage_name, X_sub_clean_df, y_sub, groups, pca_num, label_a, label_b, initial_scaler=None):
    print(f"\n--- PCA ({stage_name}) ---")

    # Scaling and PCA on the entire subset
    X_to_pca = X_sub_clean_df.copy()
    if initial_scaler:
        # Fit and transform the entire subset for consistency before CV split
        X_scaled = initial_scaler.fit_transform(X_to_pca)
    else:
        X_scaled = X_to_pca.values

    max_pca = min(pca_num, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=max_pca, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Final Standard Scaling after PCA
    final_scaler = StandardScaler()
    X_final_scaled = final_scaler.fit_transform(X_pca)

    X_final_df = pd.DataFrame(X_final_scaled)  # DataFrame for use with KFold/iloc

    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"PCA reduced features to {X_final_df.shape[1]} with total variance explained = {explained_var:.3f}")

    # Compute class weights for the full dataset
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_sub), y=y_sub)
    class_weight_dict = dict(enumerate(class_weights))

    kf = GroupKFold(n_splits = 5)
    nn_fold_metrics = []

    # Neural Network with 5-Fold Stratified Cross-Validation
    print("\n--- Starting Neural Network 5-Fold CV ---")
    from sklearn.model_selection import StratifiedKFold
    kf = GroupKFold(n_splits=5)
    nn_fold_metrics = []

    # Training NN on each fold
    for fold, (train_index, test_index) in enumerate(kf.split(X_final_df, y_sub, groups)):
        X_train_fold, X_test_fold = X_final_df.iloc[train_index], X_final_df.iloc[test_index]
        y_train_fold, y_test_fold = y_sub.iloc[train_index], y_sub.iloc[test_index]

        # Build and train the model for the current fold
        model = build_model(X_train_fold.shape[1])
        early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

        history = model.fit(
            X_train_fold, y_train_fold,
            epochs=250, batch_size=16, verbose=0,
            validation_split=0.1,
            class_weight=class_weight_dict,
            callbacks=[early_stop]
        )

        y_proba = model.predict(X_test_fold, verbose=0)
        y_pred = (y_proba > 0.5).astype(int).flatten()
        acc, auc, sens, spec, f1 = evaluate_model(y_test_fold.to_numpy(), y_pred, y_proba)
        nn_fold_metrics.append({'acc': acc, 'auc': auc, 'sens': sens, 'spec': spec, 'f1': f1})

    # Calculate mean and std deviation for NN
    df_metrics = pd.DataFrame(nn_fold_metrics)
    mean_metrics = df_metrics.mean()
    std_metrics = df_metrics.std(ddof=1)

    # Print NN Results
    print(f"\nNeural Network ({stage_name}) 5-Fold CV Results (Mean (Std)):")
    print(f"  Accuracy:    {mean_metrics['acc']:.3f} ({std_metrics['acc']:.3f})")
    print(f"  AUC:         {mean_metrics['auc']:.3f} ({std_metrics['auc']:.3f})")
    print(f"  Sensitivity: {mean_metrics['sens']:.3f} ({std_metrics['sens']:.3f})")
    print(f"  Specificity: {mean_metrics['spec']:.3f} ({std_metrics['spec']:.3f})")
    print(f"  F1-score:    {mean_metrics['f1']:.3f} ({std_metrics['f1']:.3f})")

    # Plot Loss
    test_loss, _ = model.evaluate(X_test_fold, y_test_fold, verbose=0)
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.axhline(y=test_loss, color='r', linestyle='--', label=f'Last Fold Test Loss = {test_loss:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{label_a} vs {label_b} - PCA: {stage_name}')
    plt.legend()
    plt.grid(True)
    os.makedirs(results_path, exist_ok=True)
    plt.savefig(f"{results_path}/{label_a}_vs_{label_b}_{stage_name}_({pca_num}_dimensons).png", dpi=300,
                bbox_inches='tight')
    plt.close()

    # Collect NN results
    nn_results = [
        [f"{label_a} vs {label_b}", pca_num, stage_name, "NeuralNet",
         mean_metrics['acc'], std_metrics['acc'],
         mean_metrics['auc'], std_metrics['auc'],
         mean_metrics['sens'], std_metrics['sens'],
         mean_metrics['spec'], std_metrics['spec'],
         mean_metrics['f1'], std_metrics['f1']]
    ]

    # Classical ML models with Grid Search and CV
    classical_cv_results = test_classical_models_cv(X_final_df, y_sub, groups)

    # Collect all results
    all_results = nn_results
    for name, acc_m, acc_s, auc_m, auc_s, sens_m, sens_s, spec_m, spec_s, f1_m, f1_s in classical_cv_results:
        all_results.append([f"{label_a} vs {label_b}", pca_num, stage_name, name,
                            acc_m, acc_s, auc_m, auc_s, sens_m, sens_s, spec_m, spec_s, f1_m, f1_s])

    return all_results

# Main Driver
if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')

    try:
        # Change to the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)

        # Load dataset
        pca_num = 75
        filename = "Datasets/Oasis3_complete_dataset.csv"

        # Name results path
        if filename.__contains__("Oasis") and not filename.__contains__("ADNI") and not filename.__contains__("AIBL"):
           data = "Oasis"
        elif filename.__contains__("ADNI") and not filename.__contains__("Oasis") and not filename.__contains__("AIBL"):
            data = "ADNI"
        elif filename.__contains__("AIBL") and not filename.__contains__("ADNI")and not filename.__contains__("Oasis"):
            data = "AIBL"
        elif filename.__contains__("Oasis") and filename.__contains__("ADNI") and not filename.__contains__("AIBL"):
            data = "ADNI-Oasis"
        elif filename.__contains__("ADNI") and filename.__contains__("AIBL") and not filename.__contains__("Oasis"):
            data = "ADNI-AIBL"
        elif filename.__contains__("Oasis") and filename.__contains__("AIBL") and not filename.__contains__("ADNI"):
            data = "Oasis-AIBL"
        else:
            data = "Complete"

        results_path = f"Results/{data}/PCA - {pca_num}"
        df = pd.read_csv(filename)
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

            # Check class balance
            class_counts = y_sub.value_counts()
            min_class_count = class_counts.min()
            print(f"Class distribution: {label_map[a]}={class_counts.get(a, 0)}, {label_map[b]}={class_counts.get(b, 0)}")

            if min_class_count < 10:
                print(f"Warning: Minimum class has only {min_class_count} samples. This may cause issues with 5-fold CV.")
                print("Consider using fewer folds or collecting more data for balanced comparison.")

            if min_class_count < 5:
                print(f"Skipping {label_map[a]} vs {label_map[b]} comparison due to insufficient samples.")
                continue
            # Remap to 0 and 1, reset index for clean KFold/iloc access
            y_sub = y_sub.replace({a: 0, b: 1}).reset_index(drop=True)
            X_sub = X_sub.reset_index(drop=True)

            # Remove constant features
            vt = VarianceThreshold(threshold=0.0)
            X_sub_clean = vt.fit_transform(X_sub)
            X_sub_clean_df = pd.DataFrame(X_sub_clean)

            # PCA Before Scaling
            results_before = run_pca_and_models_cv(
                "Before Scaling", X_sub_clean_df, y_sub, groups_sub,
                pca_num, label_map[a], label_map[b], initial_scaler=None
            )

            # PCA After Scaling
            results_after = run_pca_and_models_cv(
                "After Scaling", X_sub_clean_df, y_sub, groups_sub,
                pca_num, label_map[a], label_map[b], initial_scaler=initial_scaler
            )

            all_results.extend(results_before + results_after)

        # Combine and save one summary CSV
        summary_df = pd.DataFrame(all_results, columns=[
            "Comparison", "PCA_Num","PCA Stage", "Model",
            "Accuracy_Mean", "Accuracy_Std",
            "AUC_Mean", "AUC_Std",
            "Sensitivity_Mean", "Sensitivity_Std",
            "Specificity_Mean", "Specificity_Std",
            "F1-score_Mean", "F1-score_Std"
        ])
        os.makedirs("Results", exist_ok=True)
        summary_df.to_csv(f"{results_path}/model_comparison_summary_{pca_num}_dimensions.csv", index=False)
        print(f"\nSaved all results into: {results_path}/model_comparison_summary_{pca_num}_dimensions.csv")

    except FileNotFoundError as file:
        print(f"Error: Could not find {file}. Please check your file path.")