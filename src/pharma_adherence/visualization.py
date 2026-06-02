import matplotlib.pyplot as plt
import pandas as pd


def plot_hist(df: pd.DataFrame, column: str, label_rotation = "horizontal") -> plt:
    """Plot a histogram showing the distribution of a single numeric column."""
    #TODO: Plot a histogram for a numeric column.
    data = df[column].dropna()
    plt.hist(data, bins=30, color = 'skyblue', edgecolor='black')

    plt.xlabel(column)
    plt.ylabel("Count")
    plt.xticks(rotation=label_rotation)
    plt.title(f"Distribution of {column}")
    return plt

def plot_bar(df: pd.DataFrame, categories: str, values: float, label_rotation = "horizontal") -> plt:
    """Plot a bar chart of mean values grouped by a categorical column."""
    #TODO: Plot a bar plot for numeric categories and values.
    #      For multiple values per category, plot the average value per category
    new_df = df.groupby(categories, dropna=False)[values].mean().sort_values()
    plt.bar(new_df.index, new_df, color = 'skyblue')

    plt.xlabel(categories)
    plt.ylabel(values)
    plt.xticks(rotation=label_rotation)
    plt.title(f"Bar Chart of {values} of {categories}")
    return plt

def plot_scatter(df: pd.DataFrame, x: float, y: float) -> plt:
    """Plot a scatter plot to visualize the relationship between two numeric columns."""
    #TODO: Plot a scatter plot for numeric x and y values.
    plt.scatter(df[x], df[y])

    plt.title(f"{y} vs {x}")
    plt.xlabel(x)
    plt.ylabel(y)
    return plt