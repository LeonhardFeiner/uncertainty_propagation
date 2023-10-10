import itertools
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import roc_curve, RocCurveDisplay, auc
from sklearn.calibration import calibration_curve
import numpy as np
import jax.numpy as jnp
import jax

matplotlib.use("agg")

# https://arxiv.org/abs/2107.02270

c6 = np.array(
    [
        [87, 144, 252],
        [248, 156, 32],
        [228, 37, 54],
        [150, 74, 139],
        [156, 156, 161],
        [122, 33, 221],
    ],
    dtype=np.uint8,
)
c8 = np.array(
    [
        [24, 69, 251],
        [255, 94, 2],
        [201, 31, 22],
        [200, 73, 169],
        [173, 173, 125],
        [134, 200, 221],
        [87, 141, 255],
        [101, 99, 100],
    ],
    dtype=np.uint8,
)
c10 = np.array(
    [
        [63, 144, 218],
        [255, 169, 14],
        [189, 31, 1],
        [148, 164, 162],
        [131, 45, 182],
        [169, 107, 89],
        [231, 99, 0],
        [185, 172, 112],
        [113, 117, 129],
        [146, 218, 221],
    ],
    dtype=np.uint8,
)


def get_color_array(num: int):
    if num <= 6:
        return c6[:num]
    elif num <= 8:
        return c8[:num]
    elif num <= 10:
        return c10[:num]
    else:
        raise ValueError("colormap not available")


def get_color_map(num: int):
    array = get_color_array(num) / 255
    return ListedColormap(array, f"ColorBlind{num}", num)


def plot_confusion_matrix(
    target,
    prediction,
    labels,
    normalize_digits=False,
):
    """ """
    shape = (len(labels),) * 2
    cm = jnp.zeros(shape, jnp.int32).at[target, prediction].add(1)

    if normalize_digits:  # .astype("float") * 10
        cm = cm / jnp.sum(cm, axis=1, keepdims=True)
        cm = jnp.nan_to_num(cm, copy=True)
        cm = (cm * (10**normalize_digits)).astype("int")

    cm = np.array(cm)

    fig = plt.figure(figsize=(8, 8), facecolor="w", edgecolor="k")  # dpi=320,
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap="Oranges")

    tick_marks = np.arange(len(labels))

    ax.set_xlabel("Predicted", fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(labels, fontsize=4, rotation=-90, ha="center")
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    ax.set_ylabel("True Label", fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=4, va="center")
    ax.yaxis.set_label_position("left")
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(
            j,
            i,
            format(cm[i, j], "d") if cm[i, j] != 0 else ".",
            horizontalalignment="center",
            fontsize=8,
            verticalalignment="center",
            color="black",
        )
    fig.set_tight_layout(True)
    return fig


def plot_2d_scatter(
    target,
    prediction,
    color=None,
    first_max=None,
    second_max=None,
    color_max=10,
    first_name=None,
    second_name=None,
    color_name=None,
    add_line=True,  # normalize_digits=False,
):
    """ """
    fig = plt.figure(figsize=(8, 8), facecolor="w", edgecolor="k")  # dpi=320,
    ax = fig.add_subplot(1, 1, 1)

    if color is None:
        scatter = ax.scatter(np.array(target), np.array(prediction))
    else:
        scatter = ax.scatter(
            np.array(target),
            np.array(prediction),
            c=color,
            cmap=get_color_map(color_max),
        )
        legend = ax.legend(*scatter.legend_elements(), title=color_name)

    if first_max is not None:
        ax.set_xlim(0, first_max)
    if second_max is not None:
        ax.set_ylim(0, second_max)

    if first_name is not None:
        ax.set_xlabel(first_name)

    if second_name is not None:
        ax.set_ylabel(second_name)

    if add_line:
        all_values = jnp.stack((target, prediction))
        line_coords = np.stack([jnp.min(all_values), jnp.max(all_values)])
        ax.plot(line_coords, line_coords)

    return fig


def plot_bar(
    target,
    prediction,
    first_name=None,
    second_name=None,
    add_numbers=False,
    formater=".02f",
):
    """ """
    fig = plt.figure(figsize=(8, 4), facecolor="w", edgecolor="k")  # dpi=320,
    ax = fig.add_subplot(1, 1, 1)

    scatter = ax.bar(np.array(target), np.array(prediction))

    if add_numbers:
        rel_offset = prediction.max() * 0.05
        x_shift = 1 / (1 + len(prediction))
        for i, v in enumerate(prediction):
            ax.text(i - x_shift, v + rel_offset, f"{v:{formater}}")

    if first_name is not None:
        ax.set_xlabel(first_name)

    if second_name is not None:
        ax.set_ylabel(second_name)

    return fig


def plot_violin(
    uncertainty,
    first_name=None,
    second_name=None,
):
    """ """
    fig = plt.figure(figsize=(8, 8), facecolor="w", edgecolor="k")  # dpi=320,
    ax = fig.add_subplot(1, 1, 1)

    ax.violinplot(uncertainty)

    # if first_max is not None:
    #     ax.set_xlim(0, first_max)
    # if second_max is not None:
    #     ax.set_ylim(0, second_max)

    if first_name is not None:
        ax.set_xlabel(first_name)

    if second_name is not None:
        ax.set_ylabel(second_name)

    return fig


def plot_2d_hist(
    target,
    prediction,
    color=None,
    first_bins=None,
    second_bins=None,
    first_name=None,
    second_name=None,
    color_name=None,
    add_line=True,  # normalize_digits=False,
):
    """ """
    fig = plt.figure(figsize=(8, 8), facecolor="w", edgecolor="k")  # dpi=320,
    ax = fig.add_subplot(1, 1, 1)

    if color is None:
        scatter = ax.scatter(np.array(target), np.array(prediction))
    else:
        scatter = ax.scatter(
            np.array(target),
            np.array(prediction),
            c=color,
            cmap=get_color_map(color_max),
        )
        legend = ax.legend(*scatter.legend_elements(), title=color_name)

    if first_max is not None:
        ax.set_xlim(0, first_max)
    if second_max is not None:
        ax.set_ylim(0, second_max)

    if first_name is not None:
        ax.set_xlabel(first_name)

    if second_name is not None:
        ax.set_ylabel(second_name)

    if add_line:
        all_values = jnp.stack((target, prediction))
        line_coords = np.stack([jnp.min(all_values), jnp.max(all_values)])
        ax.plot(line_coords, line_coords)

    return fig


def get_multi_roc(ground_truth, class_confidences, class_labels):
    fig, ax = plt.subplots()

    for class_index, (class_label, class_confidence) in enumerate(
        zip(class_labels, class_confidences.T)
    ):
        class_target = (ground_truth == class_index).astype(int)

        fpr, tpr, thresholds = roc_curve(class_target, class_confidence)

        roc_auc = auc(fpr, tpr)
        display = RocCurveDisplay(
            fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=class_label
        )

        display.plot(ax)

    return fig


def plot_lines(
    target,
    prediction,
    line_names=None,
    line_legend_name=None,
    first_name=None,
    second_name=None,
    add_line=True,
    horizontal=False,
):
    fig = plt.figure(figsize=(8, 8), facecolor="w", edgecolor="k")  # dpi=320,
    ax = fig.add_subplot(1, 1, 1)

    scatter = ax.plot(np.transpose(target), np.transpose(prediction))
    legend = ax.legend(line_names, title=line_legend_name)

    if first_name is not None:
        ax.set_xlabel(first_name)

    if second_name is not None:
        ax.set_ylabel(second_name)

    if add_line:
        if horizontal:
            line_coords = jnp.stack((jnp.min(target), jnp.max(target)))
            ax.plot(line_coords, jnp.zeros_like(line_coords))
        else:
            all_values = jnp.stack((target, prediction))
            line_coords = np.stack([jnp.min(all_values), jnp.max(all_values)])
            ax.plot(line_coords, line_coords)

    return fig
