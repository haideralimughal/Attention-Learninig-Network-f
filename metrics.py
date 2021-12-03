import numpy as np

def iou(y_true, y_pred, threshold=0.5):
    y_pred = threshold_binarize(y_pred, threshold)
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection)

def dice_coef(y_true, y_pred, threshold=0.5):
    y_pred = threshold_binarize(y_pred, threshold)
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (
                np.sum(y_true_f) + np.sum(y_pred_f))

def Jaccard_coef_V2(y_true, y_pred, threshold=0.5):
    y_pred = threshold_binarize(y_pred, threshold)
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum( y_true_f * y_pred_f)
    return (intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection)

def recall(y_true, y_pred, threshold=0.5):
    y_pred = threshold_binarize(y_pred, threshold)
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(np.multiply(y_true_f, y_pred_f))
    re = (intersection) / (np.sum(y_true_f) + 1)
    return re

def precision(y_true, y_pred, threshold=0.5):
    y_pred = threshold_binarize(y_pred, threshold)
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(np.multiply(y_true_f, y_pred_f))
    pr = (intersection) / (np.sum(y_pred_f) + 1)
    return pr

def Specificiy(y_true, y_pred, threshold=0.5):
    y_pred = threshold_binarize(y_pred, threshold)
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(np.multiply((1 - y_true_f), (1-y_pred_f)))
    sp = (intersection) / (np.sum((1 - y_pred_f)) + 1)
    return sp

def f1(y_true, y_pred, threshold=0.5):
    y_pred = threshold_binarize(y_pred, threshold)
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    prec = precision(y_true_f, y_pred_f)
    rec = recall(y_true_f, y_pred_f)

    F1 = 2 * prec * rec / (prec + rec + 1e-10)

    return F1

# def f1(y_true, y_pred, threshold=0.5):
#     y_pred = threshold_binarize(y_pred, threshold)
#     y_true_f = y_true.flatten()
#     y_pred_f = y_pred.flatten()
#     tp = np.sum(y_true_f * y_pred_f) #true postives
#     tn = np.sum((1 - y_true_f) * (1 - y_pred_f)) #true negatives
#     fp = np.sum((y_true) * (1 - y_pred_f)) # false positives
#     fn = np.sum((1 - y_true_f) * (y_pred_f))
#
#     # Pre = tp / (tp + fp + 1e-10)
#     # Rec = tp / (tp + fn + 1e-10)
#     # F1 = 2 * (Pre * Rec) / (Pre + Rec + 1e-10)
#     F1 = 2 * tp / (2 * tp + fp + fn + 1e-10)
#     return F1

def threshold_binarize(x, threshold=0.5):
    return (x > threshold).astype(np.float32)

