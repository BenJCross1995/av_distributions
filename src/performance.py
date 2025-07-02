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
    additional_metadata: dict = None,
    keep_cols: list = None
) -> pd.DataFrame:
    """
    Evaluate speaker-verification metrics and optionally carry through metadata columns.

    Parameters:
    - df_train: DataFrame for training or cross-validation
    - score_col: column name for system scores (posterior probabilities)
    - target_col: column name for true labels (1=target, 0=non-target)
    - df_test: optional DataFrame for evaluation; if None, uses Leave-One-Out CV
    - additional_metadata: dict of static metadata fields and values
    - keep_cols: list of column names to extract from df_train as metadata;
      each must be present and have a single unique value in df_train

    Returns:
    - DataFrame with one row of metrics plus metadata
    """
    # Prepare metadata dict
    metadata = {} if additional_metadata is None else dict(additional_metadata)
    if keep_cols:
        for col in keep_cols:
            if col not in df_train.columns:
                raise KeyError(f"Column '{col}' not found in df_train")
            unique_vals = df_train[col].unique()
            if len(unique_vals) != 1:
                raise ValueError(
                    f"Column '{col}' must have a single unique value to use as metadata; found: {unique_vals}"
                )
            metadata[col] = unique_vals[0]

    # Decide between LOO CV or train/test
    if df_test is None:
        loo = LeaveOneOut()
        probs, truths = [], []
        for train_idx, test_idx in loo.split(df_train):
            train_df = df_train.iloc[train_idx]
            test_row = df_train.iloc[test_idx]
            model = LogisticRegression(solver='lbfgs')
            model.fit(train_df[[score_col]], train_df[target_col])
            p = model.predict_proba(test_row[[score_col]])[:, 1][0]
            probs.append(p)
            truths.append(bool(test_row[target_col].iloc[0]))
        pred_probs = np.array(probs)
        pred_llrs = np.log10(pred_probs / (1 - pred_probs))
        y_true = np.array(truths)
    else:
        model = LogisticRegression(solver='lbfgs')
        model.fit(df_train[[score_col]], df_train[target_col])
        pred_probs = model.predict_proba(df_test[[score_col]])[:, 1]
        pred_llrs = np.log10(pred_probs / (1 - pred_probs))
        y_true = df_test[target_col].to_numpy()

    # Compute core metrics
    cllr = compute_cllr(pred_probs, y_true)
    cllr_min = compute_cllr_min(pred_probs, y_true)
    eer = compute_eer(pred_probs, y_true)
    auc_val = float(roc_auc_score(y_true, pred_probs))

    # Classification at LLR=0 threshold
    y_pred = pred_llrs > 0
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred))
    recall = float(recall_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Log-likelihood stats
    mean_true_llr = float(np.mean(pred_llrs[y_true]))
    mean_false_llr = float(np.mean(pred_llrs[~y_true]))
    true_trials = int(np.sum(y_true))
    false_trials = int(len(y_true) - true_trials)

    # Compile results and include metadata
    metrics = {
        'Cllr': cllr,
        'Cllr_min': cllr_min,
        'EER': eer,
        'Mean_TRUE_LLR': mean_true_llr,
        'Mean_FALSE_LLR': mean_false_llr,
        'TRUE_trials': true_trials,
        'FALSE_trials': false_trials,
        'AUC': auc_val,
        'Balanced_Accuracy': bal_acc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn,
    }
    results = {**metadata, **metrics}
    
    return pd.DataFrame([results])