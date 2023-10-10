from typing import Dict, List, Tuple, Union
import numpy as np
from numpy.typing import NDArray
import jax.numpy as jnp
import jax
from einops import rearrange, reduce, repeat
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib as mpl

mpl.use("agg")

AnyArray = Union[NDArray, jax.Array]
AnyFloatingArray = Union[NDArray[np.floating], jax.Array]


def get_image_collection(
    comparision: AnyFloatingArray,
    scale: int = 3,
    bounded_greyscale=False,
    clipped_grayscale=False,
    # rescale_row=None,
    segmentation_num_classes=None,
) -> AnyFloatingArray:
    if segmentation_num_classes is not None:
        comparision = cm.get_cmap("Set1")(jnp.squeeze(comparision, -1) / 9)
    # else:
    # if rescale_row is not None:
    #     min_val = jnp.min(comparision[rescale_row], keepdims=True)
    #     max_val = jnp.max(comparision[rescale_row], keepdims=True)

    #     comparision = (comparision - min_val) / (max_val - min_val)

    elif clipped_grayscale:
        comparision = jnp.clip(comparision, 0, 1)

    greyscale_img = repeat(
        comparision,
        "source digit h w c -> c (source h tile_h) (digit w tile_w)",
        tile_h=scale,
        tile_w=scale,
    )

    assert len(greyscale_img) in {1, 3, 4}

    if len(greyscale_img) == 4:
        assert not bounded_greyscale
        return greyscale_img
    elif len(greyscale_img) == 3:
        assert not bounded_greyscale
        return jnp.concatenate((greyscale_img, jnp.ones_like(greyscale_img[:1])))
    elif bounded_greyscale:
        color_img = rearrange(
            cm.get_cmap("Greys").with_extremes(under="blue", over="red")(greyscale_img),
            "1 h w c -> c h w",
            c=4,
        )
        return color_img
    else:
        color_img = rearrange(
            cm.get_cmap("viridis")(greyscale_img), "1 h w c -> c h w", c=4
        )

        return color_img


def get_image(
    image_stack,
    scale: int = 3,
    bounded_greyscale=False,
    clipped_grayscale=False,
    difference_color=False,
    segmentation_num_classes=None,
) -> jax.Array:
    if image_stack.shape[-1] != 1:
        if image_stack.shape[-1] == 3:
            image_stack = jnp.concatenate(
                (image_stack, jnp.ones_like(image_stack[..., :1])), -1
            )
        assert image_stack.shape[-1] == 4
    else:
        channel_less_image_stack = rearrange(image_stack, "... 1 -> ...")
        if segmentation_num_classes is not None:
            image_stack = cm.get_cmap("Set1")(channel_less_image_stack / 9)
        elif difference_color:
            image_stack = cm.get_cmap("coolwarm")(channel_less_image_stack / 2 + 0.5)
        elif clipped_grayscale:
            image_stack = cm.get_cmap("gray")(
                jnp.clip(channel_less_image_stack, 0, 1 - 1e7)
            )
        elif bounded_greyscale:
            image_stack = cm.get_cmap("gray").with_extremes(under="blue", over="red")(
                channel_less_image_stack
            )
        else:
            image_stack = cm.get_cmap("viridis")(channel_less_image_stack)

    image_stack = repeat(
        image_stack,
        "... h w c -> ... (h tile_h) (w tile_w) c",
        tile_h=scale,
        tile_w=scale,
    )

    return image_stack


def combine_image_collection(collection) -> AnyFloatingArray:
    return rearrange(collection, "source digit h w c -> (source h) (digit w) c")


def get_image_stack(
    raw_stack: AnyFloatingArray,  # ..., h w c
    bounded_greyscale=False,
    clipped_grayscale=False,
    segmentation_num_classes=None,
) -> AnyFloatingArray:
    raw_stack = rearrange(raw_stack, "... h w c -> ... h w c")

    if segmentation_num_classes is not None:
        raw_stack = cm.get_cmap("Set1")(jnp.squeeze(raw_stack, -1) / 9)

    elif clipped_grayscale:
        raw_stack = jnp.clip(raw_stack, 0, 1)

    assert raw_stack.shape[-1] in {1, 3, 4}

    if raw_stack.shape[-1] == 4:
        assert not bounded_greyscale
        return raw_stack
    elif raw_stack.shape[-1] == 3:
        assert not bounded_greyscale
        return jnp.concatenate((raw_stack, jnp.ones_like(raw_stack[..., :1])), -1)
    elif bounded_greyscale:
        color_img = (
            cm.get_cmap("Greys").with_extremes(under="blue", over="red")(raw_stack),
        )
        return color_img
    else:
        color_img = cm.get_cmap("viridis")(raw_stack)
        return color_img


def get_barplots(
    labels, values, dpi=128, plot_res_factor=2, formater="=8.02f", add_numbers=True
):
    assert values.shape[1] == len(labels)
    num_images, num_labels = values.shape
    fig, axes = plt.subplots(
        1,
        num_images,
        figsize=(num_images * plot_res_factor, plot_res_factor),
        sharey=True,
        dpi=dpi,
        constrained_layout=True,
        # tight_layout=True,
    )
    if num_images == 1:
        axes = [axes]
    # fig.ylim = (0, 1)
    # fig.title = title

    rel_offset = values.max() * 0.05
    x_shift = 1 / (1 + num_labels)
    for ax, value_list in zip(axes, values):
        ax.get_yaxis().set_visible(False)
        ax.xaxis.set_ticks_position("none")
        # ax.get_xaxis().set_visible(False)
        if add_numbers:
            for i, v in enumerate(value_list):
                ax.text(i - x_shift, v + rel_offset, f"{v:{formater}}")
        # ax.ylim = (0, 100)
        # ax.axis.set_visible(False)
        ax.bar(jnp.arange(len(labels)), value_list, tick_label=labels)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    return fig


def get_text_plot(text_list, dpi=128, plot_res_factor=2, col_divisor=4):
    fig, axes = plt.subplots(
        len(text_list),
        1,
        figsize=(plot_res_factor / col_divisor, plot_res_factor * len(text_list)),
        dpi=dpi,
        constrained_layout=True,
    )

    if len(text_list) == 1:
        axes = [axes]

    for ax, text in zip(axes, text_list):
        ax.text(
            0.5,
            0.5,
            text,
            rotation=90,
            verticalalignment="center",
            horizontalalignment="center",
        )

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    return fig


def convert_fix_to_array(fig, image_count, plot_res_factor=2):
    image = convert_fix_to_array_raw(fig, plot_res_factor=plot_res_factor)

    if image_count is None:
        return image
    else:
        return repeat(
            image,
            "h (d w) c -> d h w c",
            d=image_count,
            c=4,
        )


def convert_fix_to_array_raw(fig, plot_res_factor=2):
    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = np.rint(fig.get_size_inches() * fig.get_dpi()).astype(int)
    no_alpha_img = (
        np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
        / 255
    )

    image = np.concatenate((no_alpha_img, np.ones_like(no_alpha_img[..., :1])), -1)

    return repeat(
        image,
        "h w c -> (h a) (w b) c",
        a=plot_res_factor,
        b=plot_res_factor,
        c=4,
    )


def get_barplots_img(labels, values, dpi=128, image_res_factor=1, **kwargs):
    fig = get_barplots(labels, values, dpi, **kwargs)
    array = convert_fix_to_array(fig, len(values), image_res_factor)
    plt.close(fig)
    return array


def get_text_img(
    text_list, dpi=128, image_res_factor=1, plot_res_factor=2, col_divisor=4
):
    fig = get_text_plot(text_list, dpi, plot_res_factor, col_divisor)
    array = convert_fix_to_array_raw(fig, 1)
    plt.close(fig)
    return array


def get_labelside_fig(labels, gts, dpi=128, plot_res_factor=1):
    num_images = len(gts)
    num_labels = len(labels)

    plot_width = dpi * plot_res_factor
    boarders = np.linspace(0, plot_width, num_labels + 1, endpoint=True, dtype=int)
    starts = boarders[:-1]
    ends = boarders[1:]
    plot_height = dpi * plot_res_factor
    img = np.zeros((plot_width * num_images, plot_height))

    vertical_start = plot_height // 4
    vertical_end = plot_height - vertical_start

    for i, gt in enumerate(gts):
        img[
            i * plot_width + starts[gt] : i * plot_width + ends[gt],
            vertical_start:vertical_end,
        ] = 1

    fig = plt.figure(figsize=(num_images, 1), dpi=dpi)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.imshow(img.T)
    ax.set_axis_off()
    fig.add_axes(ax)

    return fig


def get_labelside_img(labels, gts, dpi=128, image_res_factor=2, plot_res_factor=1):
    fig = get_labelside_fig(labels, gts, dpi, plot_res_factor)
    array = convert_fix_to_array(fig, len(gts), image_res_factor)
    plt.close(fig)
    return array
