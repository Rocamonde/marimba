import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import auc, roc_curve
from marimba.utils import plasticc_class_map

import marimba as mbm

LINE_STYLES = ['dashed', 'dashdot', 'dotted']
NUM_STYLES = len(LINE_STYLES)


def plot_roc(y_test, y_pred_proba, figsize=(8, 8), save_path=None):
    fpr_grid = np.linspace(0, 1, 100)
    plt.figure(figsize=figsize)
    roc_results = []
    for i in range(y_pred_proba.shape[1]):
        y_test_binary = np.where(y_test == i, 1, 0)
        fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        roc_results.append(
            (i, np.interp(fpr_grid, fpr, tpr), fpr, tpr, roc_auc))

    roc_results = sorted(roc_results, key=lambda x: x[4], reverse=True)
    roc_results = np.array(roc_results)
    # mean TPR
    mean_tpr = np.mean(roc_results[:, 1], axis=0)
    mean_roc = np.mean(roc_results[:, 4], axis=0)

    for i, _, fpr, tpr, roc_auc in roc_results:
        curve_name = list(plasticc_class_map.values())[i]
        lines = plt.plot(fpr, tpr, label='{0} (area = {1:0.2f})'
                         ''.format(curve_name, roc_auc))
        lines[0].set_linestyle(LINE_STYLES[i % NUM_STYLES])

    # plot mean TPR
    lines = plt.plot(fpr_grid, mean_tpr, 'w', label='Mean ROC curve (area = {0:0.2f})'
                     ''.format(mean_roc))
    lines[0].set_linestyle('solid')

    plt.plot([0, 1], [0, 1], 'b--')

    plt.legend()
    plt.title('Receiver Operating Characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    if save_path is not None:
        plt.savefig(save_path+'_roc.pdf', bbox_inches='tight')


def get_confusion_matrix(predicted_categories, true_categories):
    confusion_matrix = np.array(tf.math.confusion_matrix(
        true_categories, predicted_categories))
    # normalize the confusion matrix by dividing every row by its sum
    confusion_matrix = confusion_matrix / \
        confusion_matrix.sum(axis=1)[:, np.newaxis]
    # change the sign of off-diagonal elements to make it more intuitive
    # confusion_matrix = np.negative(confusion_matrix, where=(np.ones(confusion_matrix.shape) - np.eye(confusion_matrix.shape[0])).astype(bool)) + np.diag(confusion_matrix) * np.eye(confusion_matrix.shape[0])
    return confusion_matrix


def plot_confusion_matrix(cm, figsize, save_path=None):
    vmin = np.min(cm)
    vmax = np.max(cm)
    off_diag_mask = np.eye(*cm.shape, dtype=bool)

    fig = plt.figure(figsize=figsize)

    gs0 = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[20, 2], hspace=0.05)
    gs00 = matplotlib.gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs0[1], hspace=0)

    ax = fig.add_subplot(gs0[0])
    cax1 = fig.add_subplot(gs00[0])
    cax2 = fig.add_subplot(gs00[1])

    sns.heatmap(cm, annot=True, mask=~off_diag_mask, cmap='Blues', vmin=vmin, vmax=vmax, ax=ax,
                cbar_ax=cax2, fmt='.2f', xticklabels=list(plasticc_class_map.values()), yticklabels=list(plasticc_class_map.values()))  # type: ignore
    sns.heatmap(cm, annot=True, mask=off_diag_mask, cmap='OrRd', vmin=vmin, vmax=vmax, ax=ax, cbar_ax=cax1, cbar_kws=dict(
        ticks=[]), fmt='.2f', xticklabels=list(plasticc_class_map.values()), yticklabels=list(plasticc_class_map.values()))  # type: ignore

    if save_path is not None:
        fig.savefig(save_path+'_cm.pdf', bbox_inches='tight')


def predict_plot_cm_roc(model, X, y_true, figsize=(8, 8), save_path=None, one_hot=True):
    if one_hot:
        y_pred = model.predict(X)
    else:
        y_pred = model.predict_proba(X)
    y_pred_categories = np.argmax(y_pred, axis=1)
    if one_hot:
        y_true_categories = np.argmax(y_true, axis=1)
    else:
        y_true_categories = y_true

    cm = get_confusion_matrix(y_pred_categories, y_true_categories)
    plot_confusion_matrix(cm, figsize, save_path=save_path)
    plot_roc(y_true_categories, y_pred, figsize=figsize, save_path=save_path)
