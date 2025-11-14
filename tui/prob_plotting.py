import matplotlib.pyplot as plt


def plot_polygon(x_vals: list[float], y_vals: list[float], file_name: str):
    if len(x_vals) != len(y_vals):
        print("ERROR: x_vals: list[float] != y_vals: list[float]")
        return

    # Построение полигона частот
    plt.plot(x_vals, y_vals, marker="o", color="blue", label="Полигон")

    plt.legend()
    plt.grid(True)
    save_fig_to_pdf_and_png(file_name)
    plt.close()


def save_fig_to_pdf_and_png(file_path: str) -> None:
    plt.savefig(file_path)
    if file_path.endswith(".png"):
        file_path = file_path[: -len(".png")] + ".pdf"
    else:
        file_path = file_path[: -len(".pdf")] + ".png"
    plt.savefig(file_path)


def plot_polygon_from_data(data: list[float], file_name: str):
    """Plot frequency polygon from raw data."""
    from collections import Counter

    counts = Counter(data)
    x_vals = sorted(counts.keys())
    y_vals = [float(counts[x]) for x in x_vals]
    plot_polygon(x_vals, y_vals, file_name)


def plot_empirical_cdf(data: list[float], file_path: str):
    """Plot empirical cumulative distribution function."""
    sorted_data = sorted(data)
    n = len(sorted_data)

    def empirical_cdf(x):
        return sum(1 for val in sorted_data if val <= x) / n

    # To make a smooth step plot, use all unique points + boundaries
    x_vals = [min(sorted_data) - 0.5] + sorted_data + [max(sorted_data) + 0.5]
    y_vals = [empirical_cdf(x) for x in x_vals]

    plt.step(x_vals, y_vals, where="post")
    plt.ylabel("F*(x)")
    plt.title("Эмпирическая функция распределения")
    plt.grid(True)
    save_fig_to_pdf_and_png(file_path)
    plt.close()


def boxplot_single(data: list[float], file_path: str):
    """Plot a single boxplot."""
    plt.figure(figsize=(6, 5))
    plt.boxplot(data, vert=True, patch_artist=True)
    plt.title("Boxplot")
    plt.ylabel("Значения")
    plt.grid(True)
    save_fig_to_pdf_and_png(file_path)
    plt.close()


import matplotlib.pyplot as plt
import numpy as np


def calculate_roc_points(
    y_true: list[float], y_scores: list[float], num_thresholds: int = 100
):
    if len(y_true) != len(y_scores):
        print("ERROR: y_true and y_scores must have the same length")
        return [], [], []

    # Create thresholds from min to max score
    thresholds = np.linspace(min(y_scores), max(y_scores), num_thresholds)
    fpr = []
    tpr = []

    for threshold in thresholds:
        tp = 0  # True positives
        fp = 0  # False positives
        tn = 0  # True negatives
        fn = 0  # False negatives

        for i in range(len(y_true)):
            if y_scores[i] >= threshold:  # Predicted positive
                if y_true[i] == 1:  # Actually positive
                    tp += 1
                else:  # Actually negative
                    fp += 1
            else:  # Predicted negative
                if y_true[i] == 1:  # Actually positive
                    fn += 1
                else:  # Actually negative
                    tn += 1

        # Calculate TPR and FPR
        current_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        current_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tpr.append(current_tpr)
        fpr.append(current_fpr)

    return fpr, tpr, thresholds.tolist()


def calculate_auc(x: list[float], y: list[float]) -> float:
    """
    Calculate Area Under Curve using trapezoidal rule

    Parameters:
    x: X-coordinates (e.g., FPR for ROC)
    y: Y-coordinates (e.g., TPR for ROC)

    Returns:
    auc_value: Area under the curve
    """
    auc_value = 0.0
    for i in range(1, len(x)):
        # Area of trapezoid = (x2-x1) * (y1+y2)/2
        width = x[i] - x[i - 1]
        height_avg = (y[i] + y[i - 1]) / 2
        auc_value += abs(width) * height_avg

    return auc_value


def plot_roc_curve(y_true: list[float], y_scores: list[float], file_name: str):
    """
    Plot ROC curve and calculate AUC manually without sklearn

    Parameters:
    y_true: True binary labels
    y_scores: Target scores (probability of positive class)
    file_name: Output file name for saving the plot
    """
    if len(y_true) != len(y_scores):
        print("ERROR: y_true and y_scores must have the same length")
        return

    # Calculate ROC points
    fpr, tpr, thresholds = calculate_roc_points(y_true, y_scores)

    if not fpr or not tpr:
        print("ERROR: Failed to calculate ROC points")
        return

    # Calculate AUC
    roc_auc = calculate_auc(fpr, tpr)

    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.3f})"
    )
    plt.plot(
        [0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random classifier"
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True)

    save_fig_to_pdf_and_png(file_name)
    plt.close()


def calculate_pr_points(
    y_true: list[float], y_scores: list[float], num_thresholds: int = 100
):
    if len(y_true) != len(y_scores):
        print("ERROR: y_true and y_scores must have the same length")
        return [], [], []

    # Create thresholds from min to max score
    thresholds = np.linspace(min(y_scores), max(y_scores), num_thresholds)
    precision = []
    recall = []

    for threshold in thresholds:
        tp = 0  # True positives
        fp = 0  # False positives
        fn = 0  # False negatives

        for i in range(len(y_true)):
            if y_scores[i] >= threshold:  # Predicted positive
                if y_true[i] == 1:  # Actually positive
                    tp += 1
                else:  # Actually negative
                    fp += 1
            else:  # Predicted negative
                if y_true[i] == 1:  # Actually positive
                    fn += 1

        # Calculate precision and recall
        current_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        current_recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        precision.append(current_precision)
        recall.append(current_recall)

    return precision, recall, thresholds.tolist()


def plot_pr_curve(y_true: list[float], y_scores: list[float], file_name: str):
    if len(y_true) != len(y_scores):
        print("ERROR: y_true and y_scores must have the same length")
        return

    # Calculate PR points
    precision, recall, thresholds = calculate_pr_points(y_true, y_scores)

    if not precision or not recall:
        print("ERROR: Failed to calculate PR points")
        return

    # Calculate AUC for PR curve
    pr_auc = calculate_auc(
        recall[::-1], precision[::-1]
    )  # Reverse for proper integration

    # Calculate the no-skill line (baseline)
    no_skill = sum(1 for y in y_true if y == 1) / len(y_true)

    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(
        recall, precision, color="green", lw=2, label=f"PR curve (area = {pr_auc:.3f})"
    )

    plt.axhline(
        y=no_skill,
        color="red",
        linestyle="--",
        label=f"No Skill (Precision = {no_skill:.3f})",
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True)

    save_fig_to_pdf_and_png(file_name)
    plt.close()


def plot_multiple_roc_curves(
    curves_data: list[tuple[list[float], list[float], str]], file_name: str
):
    plt.figure(figsize=(8, 6))

    colors = [
        "darkorange",
        "blue",
        "green",
        "red",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]

    for i, (y_true, y_scores, model_name) in enumerate(curves_data):
        fpr, tpr, _ = calculate_roc_points(y_true, y_scores)
        if not fpr or not tpr:
            continue

        roc_auc = calculate_auc(fpr, tpr)

        color = colors[i % len(colors)]
        plt.plot(
            fpr, tpr, color=color, lw=2, label=f"{model_name} (AUC = {roc_auc:.3f})"
        )

    plt.plot(
        [0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random classifier"
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("Receiver Operating Characteristic (ROC) Curves")
    plt.legend(loc="lower right")
    plt.grid(True)

    save_fig_to_pdf_and_png(file_name)
    plt.close()
