import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_curve
)
from sklearn.model_selection import LeaveOneOut

# -----------------------------------------------------------------------------
# Core metric functions
# -----------------------------------------------------------------------------

def compute_cllr_from_llrs(llrs: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute the C_llr cost using log-likelihood ratios (LLRs).

    Parameters:
    - llrs: array of log10-likelihood ratios for each sample
    - y_true: boolean array, True for positive class, False for negative class

    Returns:
    - C_llr: float, the average proper cost of log-likelihood ratios
    """
    # Convert log10-LR to linear LR
    LR = np.power(10, llrs)

    # Separate positive (target=1) and negative trials
    pos_mask = y_true.astype(bool)
    neg_mask = ~pos_mask
    n_pos = np.sum(pos_mask)
    n_neg = np.sum(neg_mask)

    # Term for positive trials: log2(1 + 1/LR)
    term_pos = np.sum(np.log2(1 + 1 / LR[pos_mask])) / (2 * n_pos)
    # Term for negative trials: log2(1 + LR)
    term_neg = np.sum(np.log2(1 + LR[neg_mask])) / (2 * n_neg)

    return term_pos + term_neg


def compute_cllr(pred_probs: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute C_llr using predicted posterior probabilities.
    Internally converts probs to log-likelihood ratios (LLRs) and calls compute_cllr_from_llrs.
    """
    eps = 1e-15  # avoid division by zero
    # Convert to LLRs: log10(p / (1-p))
    llrs = np.log10(pred_probs / (1 - pred_probs + eps))
    return compute_cllr_from_llrs(llrs, y_true)


def compute_cllr_min(
    pred_probs: np.ndarray,
    y_true: np.ndarray,
    offset_range: tuple = (-10, 10),
    num_offsets: int = 1001
) -> float:
    """
    Compute the minimum possible C_llr by applying an additive offset to the LLRs.

    Searches `num_offsets` offsets uniformly in log10-space between offset_range.
    """
    eps = 1e-15
    # Base LLRs
    llrs = np.log10(pred_probs / (1 - pred_probs + eps))
    # Grid of additive offsets
    offsets = np.linspace(offset_range[0], offset_range[1], num_offsets)
    # Evaluate C_llr for each offset and take the minimum
    cllrs = [compute_cllr_from_llrs(llrs + off, y_true) for off in offsets]
    return float(np.min(cllrs))


def compute_eer(pred_probs: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute the Equal Error Rate (EER) from predicted probabilities.

    Finds point where False Positive Rate (FPR) equals False Negative Rate (FNR) on ROC curve.
    """
    # ROC curve: false positive rate, true positive rate, thresholds
    fpr, tpr, thresholds = roc_curve(y_true, pred_probs)
    fnr = 1 - tpr
    # EER is where FPR and FNR are closest
    idx = np.nanargmin(np.absolute(fnr - fpr))
    return float(fpr[idx])


# -----------------------------------------------------------------------------
# Main performance function
# -----------------------------------------------------------------------------

def performance(
    df_train: pd.DataFrame,
    score_col: str,
    target_col: str,
    df_test: pd.DataFrame = None,
    additional_metadata: dict = None
) -> pd.DataFrame:
    """
    Evaluate various speaker-verification metrics on a dataset.

    If df_test is None, uses Leave-One-Out cross-validation on df_train.
    Otherwise, trains on df_train and evaluates on df_test.

    Parameters:
    - df_train: DataFrame containing scores and true labels for training/calibration
    - score_col: column name of system scores (posterior probabilities)
    - target_col: column name of boolean/integer true labels (1=target, 0=non-target)
    - df_test: optional DataFrame for evaluation (same columns as df_train)
    - additional_metadata: optional dict to prepend to output (e.g. corpus/model info)

    Returns:
    - DataFrame with one row of metrics and metadata
    """
    # Decide between LOO CV or single train/test evaluation
    if df_test is None:
        loo = LeaveOneOut()
        probs, llrs, truths = [], [], []

        # Iterate each sample as the validation fold
        for train_idx, test_idx in loo.split(df_train):
            train_df = df_train.iloc[train_idx]
            test_row = df_train.iloc[test_idx]

            # Fit logistic regression on score → posterior probability
            model = LogisticRegression(solver='lbfgs')
            model.fit(train_df[[score_col]], train_df[target_col])

            # Predict probability of target class for held-out sample
            p = model.predict_proba(test_row[[score_col]])[:, 1][0]
            probs.append(p)

            # Convert to log10 LLR and collect true label
            llrs.append(np.log10(p / (1 - p)))
            truths.append(bool(test_row[target_col].iloc[0]))

        # Assemble arrays for metric calculations
        pred_probs = np.array(probs)
        pred_llrs = np.array(llrs)
        y_true = np.array(truths)
    else:
        # Train on entire training set
        model = LogisticRegression(solver='lbfgs')
        model.fit(df_train[[score_col]], df_train[target_col])
        # Predict on held-back test set
        pred_probs = model.predict_proba(df_test[[score_col]])[:, 1]
        pred_llrs = np.log10(pred_probs / (1 - pred_probs))
        y_true = df_test[target_col].to_numpy()

    # Core metrics
    cllr = compute_cllr(pred_probs, y_true)
    cllr_min = compute_cllr_min(pred_probs, y_true)
    eer = compute_eer(pred_probs, y_true)
    auc_val = float(roc_auc_score(y_true, pred_probs))

    # Threshold at LLR=0 (equal support) for classification metrics
    y_pred = pred_llrs > 0
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred))
    recall = float(recall_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Mean log10 LLR for true and false trials
    mean_true_llr = float(np.mean(pred_llrs[y_true]))
    mean_false_llr = float(np.mean(pred_llrs[~y_true]))

    # Count trials
    true_trials = int(np.sum(y_true))
    false_trials = int(len(y_true) - true_trials)

    # Compile results dict
    results = {
        'Cllr': cllr,
        'Cllr_min': cllr_min,
        'EER': eer,
        'AUC': auc_val,
        'Balanced_Accuracy': bal_acc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'TP': int(tp),
        'FP': int(fp),
        'FN': int(fn),
        'TN': int(tn),
        'Mean_TRUE_LLR': mean_true_llr,
        'Mean_FALSE_LLR': mean_false_llr,
        'TRUE_trials': true_trials,
        'FALSE_trials': false_trials
    }

    # Prepend any additional metadata fields
    if additional_metadata:
        results = {**additional_metadata, **results}

    # Return as single-row DataFrame
    return pd.DataFrame([results])