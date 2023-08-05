import os
import matplotlib.pyplot as plt
import numpy as np


def plot_distribution(data: list[int | float], title: str = '', xlabel: str = 'Data', ylabel: str = 'Frequency', bins: int = 30) -> plt.Figure:
    """
    Plot the distribution of data.

    Args:
        data (array-like): The data to be plotted.
        title (str): The title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        bins (int, optional): Number of bins for the histogram. Defaults to 30.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data, bins=bins, alpha=0.7, color='b')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    return fig


def plot_image_with_scatter(valence: np.ndarray, arousal: np.ndarray, img: str | np.ndarray, sigma: float = 0.25) -> plt.Figure:
    """
    Plot an image with scatter points.

    Args:
        valence (np.ndarray): Array of valence values.
        arousal (np.ndarray): Array of arousal values.
        img (Union[str, np.ndarray]): Path to an image file or image as a numpy array.
        sigma (float, optional): A value for transparency calculation. 1 - no transperency, 0 - completely transparent. Defaults to 0.25.

    Returns:
        plt.Figure: The matplotlib Figure object.

    Raises:
        IsADirectoryError: If img is a directory.
        FileNotFoundError: If img file is not found.
        ValueError: If img is not a string or a numpy array.
    """

    if isinstance(img, str):
        if os.path.exists(img):
            if os.path.isfile(img):
                name = os.path.basename(img)
                img = plt.imread(img)
            else:
                raise IsADirectoryError(f'{img} is a directory')
        else:
            raise FileNotFoundError(f'File {img} not found')
    elif isinstance(img, np.ndarray):
        name = 'image'
    else:
        raise ValueError(f'path must be a string or a numpy array, got {type(img)}')

    mean_valence = valence.mean()
    mean_arousal = arousal.mean()
    alpha_linear = 1 - np.abs(np.sqrt((valence - mean_valence) ** 2 + (arousal - mean_arousal) ** 2)) / 8

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(img)
    ax1.set_title(name)
    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)

    ax2.grid()
    ax2.scatter(mean_valence, mean_arousal, color='tab:red', zorder=2)
    ax2.scatter(
        valence,
        arousal,
        color='tab:blue',
        alpha=np.exp(-(1 - alpha_linear) ** 2 / sigma ** 2),
        zorder=2
    )
    ax2.axhline(4, color='black', zorder=1, alpha=0.5)
    ax2.axvline(4, color='black', zorder=1, alpha=0.5)
    ax2.set_xlim(1, 7)
    ax2.set_ylim(1, 7)
    ax2.set_ylabel('Arousal')
    ax2.set_xlabel('Valence')
    ax2.legend(['mean', 'individual'], loc='upper right')

    return fig
