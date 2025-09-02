import glob
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler # NEW
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score
)
from sklearn.model_selection import LeaveOneOut

# My modules
from read_and_write_docs import read_jsonl

# -----------------------------------------------------------------------------
# Performance Utility Functions
# -----------------------------------------------------------------------------

def aggregate_results_data(df, group_cols = ['problem', 'target']):
    """Aggregate results data based on grouping columns"""
    
    grouped_df = (
        df
        .groupby(group_cols, as_index=False)
        ['score']
        .mean()
    )

    return grouped_df

def combine_and_aggregate_results_data(file_loc, method='paraphrase', group_cols=['problem', 'target']):
    """Combine results data from a directory and aggreagate in one go"""
    
    jsonl_files = glob.glob(f"{file_loc}/*.jsonl")

    combined_results = []
    for filename in jsonl_files:
        data = read_jsonl(filename)
        grouped_data = aggregate_results_data(data, group_cols)
        grouped_data.insert(0, 'method', method)
    
        combined_results.append(grouped_data)

    results = pd.concat(combined_results, ignore_index=True)

    return results

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
    keep_cols: list = None,
    group_cols: list | str = None,
) -> pd.DataFrame:
    """
    Evaluate speaker‑verification metrics and optionally carry through metadata columns,
    either once or per group, with group_cols up front in the output.
    """
    # Normalize group_cols to list
    if group_cols is not None and not isinstance(group_cols, (list, tuple)):
        group_cols = [group_cols]

    def _single_perf(sub_train, sub_test, group_vals=None):
        # start row with group identifiers (if any)
        row = {}
        if group_vals is not None:
            if isinstance(group_vals, tuple):
                for col, val in zip(group_cols, group_vals):
                    row[col] = val
            else:
                row[group_cols[0]] = group_vals

        # metadata
        meta = {} if additional_metadata is None else dict(additional_metadata)
        if keep_cols:
            for c in keep_cols:
                if c not in sub_train:
                    raise KeyError(f"Column '{c}' not in df_train")
                vals = sub_train[c].unique()
                if len(vals) != 1:
                    raise ValueError(f"Column '{c}' must be single-valued per group; got {vals}")
                meta[c] = vals[0]
        row.update(meta)

        # train/test or LOO
        if sub_test is None:
            loo = LeaveOneOut()
            probs, truths = [], []
            for tr_idx, te_idx in loo.split(sub_train):
                tr = sub_train.iloc[tr_idx]
                te = sub_train.iloc[te_idx]
                m = LogisticRegression(solver='lbfgs')
                m.fit(tr[[score_col]], tr[target_col])
                p = m.predict_proba(te[[score_col]])[:,1][0]
                probs.append(p)
                truths.append(bool(te[target_col].iloc[0]))
            y_true    = np.array(truths)
            pred_probs = np.array(probs)
        else:
            m = LogisticRegression(solver='lbfgs')
            m.fit(sub_train[[score_col]], sub_train[target_col])
            pred_probs = m.predict_proba(sub_test[[score_col]])[:,1]
            y_true     = sub_test[target_col].to_numpy()

        llr = np.log10(pred_probs / (1 - pred_probs))

        # core metrics (import your own compute_cllr etc.)
        stats = {
            'Cllr': compute_cllr(pred_probs, y_true),
            'Cllr_min': compute_cllr_min(pred_probs, y_true),
            'EER': compute_eer(pred_probs, y_true),
            'Mean_TRUE_LLR': float(np.mean(llr[y_true])),
            'Mean_FALSE_LLR': float(np.mean(llr[~y_true])),
            'TRUE_trials': int(np.sum(y_true)),
            'FALSE_trials': int(len(y_true) - np.sum(y_true)),
            'AUC': float(roc_auc_score(y_true, pred_probs)),
        }
        y_pred = llr > 0
        stats.update({
            'Balanced_Accuracy': float(balanced_accuracy_score(y_true, y_pred)),
            'Precision': float(precision_score(y_true, y_pred)),
            'Recall': float(recall_score(y_true, y_pred)),
            'F1': float(f1_score(y_true, y_pred)),
        })
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        stats.update({'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn)})

        row.update(stats)
        return row

    # no grouping: single row
    if group_cols is None:
        result = _single_perf(df_train, df_test)
        return pd.DataFrame([result])

    # per‑group
    rows = []
    for grp_vals, grp_df in df_train.groupby(group_cols):
        # filter test to matching group if needed
        sub_test = None
        if df_test is not None:
            mask = pd.Series(True, index=df_test.index)
            vals = grp_vals if isinstance(grp_vals, tuple) else (grp_vals,)
            for col, val in zip(group_cols, vals):
                mask &= df_test[col] == val
            sub_test = df_test[mask]

        rows.append(_single_perf(grp_df, sub_test, group_vals=grp_vals))

    return pd.DataFrame(rows)

def performance_paraphrase(
    df_train: pd.DataFrame,
    score_col: str,
    target_col: str,
    df_test: pd.DataFrame = None,
    additional_metadata: dict = None,
    keep_cols: list = None
) -> pd.DataFrame:
    """
    Variant of `performance` that standardizes raw scores before logistic calibration.

    Parameters:
    - df_train: training DataFrame
    - score_col: name of column containing raw system scores (e.g., log-likelihood ratios)
    - target_col: column name for binary labels (1=target, 0=non-target)
    - df_test: optional test DataFrame; if None, uses Leave-One-Out CV on df_train
    - additional_metadata: dict of static metadata fields to include
    - keep_cols: list of DataFrame columns to carry through as metadata

    Returns:
    - DataFrame with one row of metrics plus metadata
    """
    # Prepare metadata
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

    # Extract and standardize scores
    scaler = StandardScaler()
    scores_train = df_train[[score_col]].values.astype(float)
    scores_train_std = scaler.fit_transform(scores_train)

    # Decide between LOO CV or train/test
    if df_test is None:
        loo = LeaveOneOut()
        probs, truths = [], []
        for train_idx, test_idx in loo.split(df_train):
            X_tr = scores_train_std[train_idx]
            y_tr = df_train[target_col].iloc[train_idx]
            X_te = scores_train_std[test_idx]

            # Fit unregularized logistic
            model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000)
            model.fit(X_tr, y_tr)

            # Predict probability for the single test sample
            p = model.predict_proba(X_te)[:, 1][0]
            probs.append(p)

            # Extract the true label as a scalar
            true_label = df_train[target_col].iloc[test_idx[0]]
            truths.append(bool(true_label))

        pred_probs = np.array(probs)
        y_true = np.array(truths)
        pred_llrs = np.log10(pred_probs / (1 - pred_probs))
    else:
        # Standardize both train and test using train statistics
        scores_test = df_test[[score_col]].values.astype(float)
        scores_test_std = scaler.transform(scores_test)

        # Fit unregularized logistic
        model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000)
        model.fit(scores_train_std, df_train[target_col])

        # Predict on test set
        pred_probs = model.predict_proba(scores_test_std)[:, 1]
        y_true = df_test[target_col].to_numpy()
        pred_llrs = np.log10(pred_probs / (1 - pred_probs))

    # Compute metrics (reuse your existing functions)
    cllr = compute_cllr(pred_probs, y_true)
    cllr_min = compute_cllr_min(pred_probs, y_true)
    eer = compute_eer(pred_probs, y_true)
    auc_val = float(roc_auc_score(y_true, pred_probs))

    # Classify at LLR=0 threshold
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

    # Compile results
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

# -----------------------------------------------------------------------------
# Plotting functions
# -----------------------------------------------------------------------------

def distribution_plot(df,
                      score_col='score',
                      target_col='target',
                      xlabel='Log Likelihood Ratio Score',
                      ylabel='Count',
                      title='Score Distribution by True/False Author Match',
                      group_col=None):
    """
    Plots score distributions as histograms, optionally facetted by a grouping column.

    Parameters:
    -----------
    df : pandas.DataFrame
        Must contain a boolean (or binary) target column and a score column.
    score_col : str
        Column name for the scores to histogram.
    target_col : str
        Column name for binary labels (True/1 = same author, False/0 = different authors).
    xlabel, ylabel, title : str
        Plot labels and overall title.
    group_col : str or None
        If provided, column name to group by; creates one subplot per unique value.
    """
    def _plot_panel(ax, subdf):
        # Same‐author histogram:
        subdf.loc[subdf[target_col], score_col].hist(
            ax=ax, alpha=0.7, label='Same Author'
        )
        # Different‐author histogram (invert the mask with ~):
        subdf.loc[~subdf[target_col], score_col].hist(
            ax=ax, alpha=0.7, label='Different Authors'
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc='best')
        ax.grid(False)

    if group_col is None:
        # Single panel
        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_panel(ax, df)
        ax.set_title(title)
        fig.tight_layout()
        plt.show()

    else:
        groups = df[group_col].dropna().unique()
        n = len(groups)
        ncols = int(math.ceil(math.sqrt(n)))
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), squeeze=False)

        for idx, grp in enumerate(groups):
            ax = axes.flat[idx]
            subdf = df[df[group_col] == grp]
            _plot_panel(ax, subdf)
            ax.set_title(f"{group_col} = {grp}")

        # Hide unused axes
        for idx in range(n, nrows * ncols):
            axes.flat[idx].set_visible(False)

        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

def roc_plot(df,
             target_col='target',
             score_col='score',
             title='ROC Curve',
             group_col=None):
    """
    Plots ROC curve(s) and prints summary metrics & confusion matrix,
    optionally facetted by a grouping column.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain the target and score columns.
    target_col : str
        Name of the binary ground‑truth column (1 = positive, 0 = negative).
    score_col : str
        Name of the continuous score/probability column.
    title : str
        Overall title for the ROC plot(s).
    group_col : str or None
        If provided, creates one ROC subplot per unique value in this column
        and prints metrics per group. Otherwise behaves as single‑panel.
    """
    def _compute_metrics(subdf):
        y_true = subdf[target_col]
        y_score = subdf[score_col]
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        best_thresh = thresholds[best_idx]
        preds = y_score >= best_thresh
        f1 = f1_score(y_true, preds)
        acc = accuracy_score(y_true, preds)
        cm = confusion_matrix(y_true, preds)
        cm_df = pd.DataFrame(cm,
                             index=['Actual 0', 'Actual 1'],
                             columns=['Pred 0', 'Pred 1'])
        return {
            'fpr': fpr,
            'tpr': tpr,
            'best_idx': best_idx,
            'best_thresh': best_thresh,
            'roc_auc': roc_auc,
            'f1': f1,
            'accuracy': acc,
            'cm_df': cm_df
        }

    if group_col is None:
        # Single-panel ROC
        metrics = _compute_metrics(df)
        plt.figure(figsize=(8, 6))
        plt.plot(metrics['fpr'], metrics['tpr'],
                 label=f'ROC (AUC = {metrics["roc_auc"]:.2f})', linewidth=2)
        bi = metrics['best_idx']
        plt.scatter(metrics['fpr'][bi], metrics['tpr'][bi],
                    zorder=5,
                    label=f'Best Thr = {metrics["best_thresh"]:.2f}')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title(title)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Print metrics
        print("Summary Metrics\n")
        print(f"Optimal Threshold (Youden's J): {metrics['best_thresh']:.3f}")
        print(f"AUC: {metrics['roc_auc']:.3f}")
        print(f"F1 Score: {metrics['f1']:.3f}")
        print(f"Accuracy: {metrics['accuracy']:.3f}\n")
        print("Confusion Matrix:")
        print(metrics['cm_df'])

    else:
        # Facetted ROC by group
        groups = df[group_col].dropna().unique()
        n = len(groups)
        ncols = int(math.ceil(math.sqrt(n)))
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4*ncols, 3*nrows),
                                 squeeze=False)
        summary = []

        for idx, grp in enumerate(groups):
            ax = axes.flat[idx]
            subdf = df[df[group_col] == grp]
            m = _compute_metrics(subdf)
            # Plot ROC
            ax.plot(m['fpr'], m['tpr'],
                    label=f'AUC = {m["roc_auc"]:.2f}',
                    linewidth=1.5)
            bi = m['best_idx']
            ax.scatter(m['fpr'][bi], m['tpr'][bi],
                       s=20, zorder=5,
                       label=f'Thr ≈ {m["best_thresh"]:.2f}')
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
            ax.set_title(f"{group_col} = {grp}")
            ax.set_xlabel('FPR')
            ax.set_ylabel('TPR')
            ax.grid(alpha=0.3)
            if idx == 0:
                ax.legend(loc='lower right', fontsize='small')

            # Collect summary
            summary.append({
                group_col: grp,
                'AUC': m['roc_auc'],
                'Best Threshold': m['best_thresh'],
                'F1 Score': m['f1'],
                'Accuracy': m['accuracy']
            })

        # Turn off any unused axes
        for idx in range(n, nrows*ncols):
            axes.flat[idx].set_visible(False)

        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

        # Display summary table
        summary_df = pd.DataFrame(summary).set_index(group_col)
        print("Per‑Group Summary Metrics:\n")
        print(summary_df)

        # Optionally, to print each confusion matrix:
        for grp, m in zip(groups, [ _compute_metrics(df[df[group_col]==g]) for g in groups ]):
            print(f"\nConfusion Matrix for {group_col} = {grp}:")
            print(m['cm_df'])

def tippett_plot(df,
                 target_col='target',
                 score_col='score',
                 xlabel='Likelihood‑ratio threshold',
                 ylabel='Cumulative proportion',
                 title='Tippett Plot',
                 group_col=None):
    """
    Draws a Tippett plot from a DataFrame, optionally facetted by a grouping column.

    Parameters:
    -----------
    df : pandas.DataFrame
        Must contain a binary column (H1 vs H2 labels) and a score column (LR outputs).
    target_col : str
        Column name for binary labels (1 = target/H1, 0 = non-target/H2).
    score_col : str
        Column name for the likelihood‑ratio scores.
    xlabel, ylabel, title : str
        Plot labels and title.
    group_col : str or None
        If provided, column name to group by; creates one subplot per unique value.
    """
    def _compute_rates(subdf):
        scores_h1 = subdf.loc[subdf[target_col] == 1, score_col].values
        scores_h2 = subdf.loc[subdf[target_col] == 0, score_col].values
        thresh = np.sort(np.unique(np.concatenate((scores_h1, scores_h2))))
        tpr = np.array([np.mean(scores_h1 >= t) for t in thresh])
        tnr = np.array([np.mean(scores_h2 <= t) for t in thresh])
        return thresh, tpr, tnr

    if group_col is None:
        # Single panel
        thresh, tpr, tnr = _compute_rates(df)
        plt.figure(figsize=(8, 5))
        plt.semilogx(thresh, tpr, label='TPR (H₁)', linewidth=2)
        plt.semilogx(thresh, tnr, label='TNR (H₂)', linewidth=2)
        plt.axvline(1, color='gray', linestyle='--', label='LR = 1')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True, which='both', ls=':')
        plt.tight_layout()
        plt.show()

    else:
        groups = df[group_col].dropna().unique()
        n = len(groups)
        # choose a roughly square layout
        ncols = int(math.ceil(math.sqrt(n)))
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), squeeze=False)
        for idx, grp in enumerate(groups):
            ax = axes.flat[idx]
            subdf = df[df[group_col] == grp]
            thresh, tpr, tnr = _compute_rates(subdf)
            ax.semilogx(thresh, tpr, label='TPR (H₁)', linewidth=1.5)
            ax.semilogx(thresh, tnr, label='TNR (H₂)', linewidth=1.5)
            ax.axvline(1, color='gray', linestyle='--', label='LR = 1')
            ax.set_title(f"{grp}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, which='both', ls=':')
            if idx == 0:
                ax.legend(loc='best', fontsize='small')

        # turn off any unused axes
        for idx in range(n, nrows*ncols):
            axes.flat[idx].set_visible(False)

        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
