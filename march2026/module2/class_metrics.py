import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
    log_loss,
    confusion_matrix
)

from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize


sns.set_style("whitegrid")


# --------------------------------------------------
# GRID UTILITY
# --------------------------------------------------

def compute_grid(n_plots, base_rows, base_cols):

    capacity = base_rows * base_cols

    if n_plots <= capacity:
        return base_rows, base_cols

    n = math.ceil(math.sqrt(n_plots))
    return n, n


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

def plot_confusion_matrix(ax, cm):

    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    total = cm.sum()

    cm_ext = np.zeros((cm.shape[0] + 1, cm.shape[1] + 1))

    cm_ext[:-1, :-1] = cm
    cm_ext[:-1, -1] = row_sums
    cm_ext[-1, :-1] = col_sums
    cm_ext[-1, -1] = total

    labels = [f"C{i}" for i in range(cm.shape[0])] + ["Total"]

    df = pd.DataFrame(cm_ext, index=labels, columns=labels)

    sns.heatmap(
        df,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        cbar=False,
        linewidths=0.5,
        ax=ax
    )

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.tick_params(axis='both', which='both', length=0)


# --------------------------------------------------
# ROC CURVE
# --------------------------------------------------

def plot_roc(ax, fpr, tpr, roc_auc):

    ax.plot(fpr, tpr, linewidth=2, label=f"Model (AUC={roc_auc:.3f})")

    ax.plot(
        [0,1],
        [0,1],
        linestyle="--",
        color="gray",
        label="Random Guess"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    ax.set_title("ROC Curve")

    ax.legend()


# --------------------------------------------------
# PRECISION RECALL
# --------------------------------------------------

def plot_pr(ax, recall, precision, pr_auc, baseline):

    ax.plot(recall, precision, linewidth=2,
            label=f"Model (AUC={pr_auc:.3f})")

    ax.axhline(
        baseline,
        linestyle="--",
        color="gray",
        label="Random Guess"
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    ax.set_title("Precision–Recall Curve")

    ax.legend()


# --------------------------------------------------
# PROBABILITY DISTRIBUTION
# --------------------------------------------------

def plot_probability_distribution(ax, y_true, y_prob):

    classes = np.unique(y_true)

    palette = sns.color_palette("deep", len(classes))

    for i, c in enumerate(classes):

        sns.histplot(
            y_prob[y_true == c],
            bins=25,
            stat="density",
            kde=True,
            ax=ax,
            color=palette[i],
            label=f"True Class {c}",
            alpha=0.6
        )

    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")

    ax.set_title("Probability Distribution by True Class")

    ax.legend()


# --------------------------------------------------
# CALIBRATION CURVE
# --------------------------------------------------

def plot_calibration(ax, prob_true, prob_pred):

    ax.plot(prob_pred, prob_true, marker="o", label="Model")

    ax.plot(
        [0,1],
        [0,1],
        linestyle="--",
        color="gray",
        label="Perfect Calibration"
    )

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Frequency")

    ax.set_title("Calibration Curve")

    ax.legend()


# --------------------------------------------------
# THRESHOLD METRICS
# --------------------------------------------------

def plot_threshold_metrics(ax, thresholds, p, r, f1, s):

    ax.plot(thresholds, p, label="Precision")
    ax.plot(thresholds, r, label="Recall")
    ax.plot(thresholds, f1, label="F1 Score")
    ax.plot(thresholds, s, label="Specificity")

    ax.set_xlabel("Probability Threshold")
    ax.set_ylabel("Metric Value")

    ax.set_title("Metrics vs Threshold")

    ax.legend()


# --------------------------------------------------
# CUMULATIVE GAIN
# --------------------------------------------------

def plot_gain(ax, perc, gain):

    ax.plot(perc, gain, linewidth=2, label="Model")

    ax.plot(
        [0,1],
        [0,1],
        linestyle="--",
        color="gray",
        label="Random Guess"
    )

    ax.set_xlabel("Fraction of Population")
    ax.set_ylabel("Fraction of Positives Captured")

    ax.set_title("Cumulative Gain Curve")

    ax.legend()


# --------------------------------------------------
# MULTICLASS ROC
# --------------------------------------------------

def plot_multiclass_roc(ax, y_bin, y_prob, classes):

    for i in range(len(classes)):

        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)

        ax.plot(
            fpr,
            tpr,
            label=f"Class {classes[i]} (AUC={roc_auc:.2f})"
        )

    ax.plot([0,1],[0,1],'--',color="gray",label="Random Guess")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    ax.set_title("Multiclass ROC")

    ax.legend()


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def classification_report_with_plots(
        y_true,
        y_prob,
        prob_threshold=0.5,
        grid=(2,2)
):

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    n_classes = len(np.unique(y_true))

    plots = []
    metrics = {}

    # --------------------------------------------------
    # BINARY
    # --------------------------------------------------

    if n_classes == 2 and y_prob.ndim == 1:

        y_pred = (y_prob >= prob_threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        specificity = tn / (tn + fp)

        roc_auc = roc_auc_score(y_true, y_prob)

        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall_curve, precision_curve)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "specificity": specificity,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "log_loss": log_loss(y_true, y_prob)
        }

        fpr, tpr, _ = roc_curve(y_true, y_prob)

        plots.append(("roc", (fpr, tpr, roc_auc)))
        plots.append(("pr", (recall_curve, precision_curve, pr_auc, np.mean(y_true))))
        plots.append(("cm", confusion_matrix(y_true, y_pred)))
        plots.append(("prob", (y_true, y_prob)))

        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        plots.append(("cal", (prob_true, prob_pred)))

        thresholds = np.linspace(0,1,100)

        p,r,f1,s = [],[],[],[]

        for t in thresholds:

            pred = (y_prob >= t).astype(int)

            tn,fp,fn,tp = confusion_matrix(y_true,pred).ravel()

            p.append(precision_score(y_true,pred,zero_division=0))
            r.append(recall_score(y_true,pred,zero_division=0))
            f1.append(f1_score(y_true,pred,zero_division=0))
            s.append(tn/(tn+fp) if (tn+fp)>0 else 0)

        plots.append(("th",(thresholds,p,r,f1,s)))

        order = np.argsort(-y_prob)
        y_sorted = y_true[order]

        cumulative = np.cumsum(y_sorted)

        perc = np.arange(1,len(y_true)+1)/len(y_true)
        gain = cumulative/np.sum(y_true)

        plots.append(("gain",(perc,gain)))

    # --------------------------------------------------
    # MULTICLASS
    # --------------------------------------------------

    else:

        y_pred = np.argmax(y_prob, axis=1)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro"),
            "recall_macro": recall_score(y_true, y_pred, average="macro"),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "log_loss": log_loss(y_true, y_prob)
        }

        plots.append(("cm", confusion_matrix(y_true, y_pred)))

        classes = np.unique(y_true)
        y_bin = label_binarize(y_true, classes=classes)

        plots.append(("mroc",(y_bin,y_prob,classes)))

        plots.append(("mprob",(y_true,y_prob)))

    # --------------------------------------------------
    # SUBPLOTS
    # --------------------------------------------------

    rows, cols = compute_grid(len(plots), grid[0], grid[1])

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols,5*rows))
    axes = axes.flatten()

    for i,(ptype,data) in enumerate(plots):

        ax = axes[i]

        if ptype=="roc":
            plot_roc(ax,*data)

        elif ptype=="pr":
            plot_pr(ax,*data)

        elif ptype=="cm":
            plot_confusion_matrix(ax,data)

        elif ptype=="prob":
            plot_probability_distribution(ax,*data)

        elif ptype=="cal":
            plot_calibration(ax,*data)

        elif ptype=="th":
            plot_threshold_metrics(ax,*data)

        elif ptype=="gain":
            plot_gain(ax,*data)

        elif ptype=="mroc":
            plot_multiclass_roc(ax,*data)

        elif ptype=="mprob":

            y_true,y_prob=data

            probs=y_prob[np.arange(len(y_true)),y_true]

            sns.histplot(probs,bins=25,kde=True,ax=ax)

            ax.set_title("True-Class Probability Distribution")
            ax.set_xlabel("Probability Assigned to True Class")
            ax.set_ylabel("Density")

    for j in range(len(plots),len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    return pd.DataFrame(
        list(metrics.items()),
        columns=["metric","value"]
    )


# --------------------------------------------------
# EXAMPLE
# --------------------------------------------------

if __name__ == "__main__":

    y_true = [0,1,1,0,1,0,1,0]
    y_prob = [0.1,0.9,0.8,0.3,0.7,0.2,0.6,0.4]

    report = classification_report_with_plots(
        y_true,
        y_prob,
        prob_threshold=0.5,
        grid=(2,2)
    )

    print(report)