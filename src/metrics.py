import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def calculate_cllr(y_true, y_scores):
    """
    Calculate the Cllr and Cllr_min.
    """
    y_scores = np.array(y_scores)
    y_true = np.array(y_true)
    
    logit = lambda x: np.log(x / (1 - x))
    inv_logit = lambda x: 1 / (1 + np.exp(-x))
    
    log_lr = logit(y_scores)
    log_lr_0 = log_lr[y_true == 0]
    log_lr_1 = log_lr[y_true == 1]
    
    cllr = -0.5 * (np.mean(np.log(1 - inv_logit(log_lr_0))) + np.mean(np.log(inv_logit(log_lr_1))))
    cllr_min = -0.5 * (np.log(1 - inv_logit(np.mean(log_lr_0))) + np.log(inv_logit(np.mean(log_lr_1))))
    
    return cllr, cllr_min

def calculate_eer(y_true, y_scores):
    """
    Calculate the Equal Error Rate (EER).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

def performance(df, score_col, target_col, result_name, result_description, corpus, data_type, model):
    """
    Calculate various performance metrics for binary classification and return them in a DataFrame.
    
    Parameters:
    - df: pandas DataFrame containing the data.
    - score_col: str, the name of the column containing the scores.
    - target_col: str, the name of the column containing the true labels (0 or 1).
    - result_name: str, the name of the result.
    - result_description: str, a description of the result.
    - corpus: str, the name of the corpus used.
    - data_type: str, the type of data (e.g., 'train', 'test', 'validation').
    - model: str, the name of the model used.
    
    Returns:
    - pandas DataFrame: A DataFrame containing the performance metrics and additional metadata.
    """
    scores = df[score_col].values
    targets = df[target_col].values
    
    # Calculate metrics
    auc_roc = roc_auc_score(targets, scores)
    balanced_acc = balanced_accuracy_score(targets, (scores > 0.5).astype(int))
    precision = precision_score(targets, (scores > 0.5).astype(int))
    recall = recall_score(targets, (scores > 0.5).astype(int))
    f1 = f1_score(targets, (scores > 0.5).astype(int))
    cllr, cllr_min = calculate_cllr(targets, scores)
    eer = calculate_eer(targets, scores)
    
    cm = confusion_matrix(targets, (scores > 0.5).astype(int))
    tp = cm[1, 1]
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    
    mean_true_llr = np.mean(np.log(scores[targets == 1] / (1 - scores[targets == 1])))
    mean_false_llr = np.mean(np.log((1 - scores[targets == 0]) / scores[targets == 0]))
    
    true_trials = np.sum(targets == 1)
    false_trials = np.sum(targets == 0)
    
    metrics = {
        'Result_Name': result_name,
        'Result_Description': result_description,
        'Corpus': corpus,
        'Data_Type': data_type,
        'Model': model,
        'AUC_ROC': auc_roc,
        'Balanced_Accuracy': balanced_acc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Cllr': cllr,
        'Cllr_min': cllr_min,
        'EER': eer,
        'Mean_TRUE_LLR': mean_true_llr,
        'Mean_FALSE_LLR': mean_false_llr,
        'TRUE_Trials': true_trials,
        'FALSE_Trials': false_trials,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn
    }
    
    # Convert the dictionary to a DataFrame
    metrics_df = pd.DataFrame([metrics])
    
    return metrics_df