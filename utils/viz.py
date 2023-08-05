import matplotlib.pyplot as plt

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