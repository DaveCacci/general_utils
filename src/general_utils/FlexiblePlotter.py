from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import os
import logging
import mplcursors  # Ensure mplcursors is installed for hover functionality of labels
from collections import defaultdict

def list_to_dataframe(data: List[np.ndarray], column_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convert a list of arrays to a DataFrame.
    Args:
        data: List of arrays.
        column_names: List of column names. If not provided, default names will be used.
    Returns:
        DataFrame containing the data.
    """
    df = pd.DataFrame(data).T
    if column_names:
        df.columns = column_names
    else:
        df.columns = [f'Var{i}' for i in range(df.shape[1])]
    return df

@dataclass
class FlexiblePlotter:
    """
    A flexible plotter that supports datetime and float x-axes, with customizable options.
    All methods show the figure if `show_plot` is True. Externally if False, it is possible to save the plot/s;
    except for the 'generate_plots_with_incremental_variables' method, which saves the plots incrementally.
    To handle:
    - (i) different variables of a dataframe/dict on different plots
    - (ii) different varibles of a dataframedict on same plot
    - (iii) different variables across different dataframes in one plot
    - (iv) different variables across different dataframes on different plots
    
    Note: this class at the moment is not able to plot confidence intervals of variables with 'fill_between'!
    """
    default_figsize: Tuple[int, int] = (8, 6)
    default_color: str = "blue"
    default_alpha: float = 0.8
    default_linestyle: str = "-"
    default_marker: Optional[str] = None
    logger: logging.Logger = logging.getLogger(__name__)

    def __post_init__(self):
        self.logger.info('-----------------------------------------------\n')
        self.logger.info('Instantiting FlexiblePlotter class\n')

    def _apply_legend(self, ax, show_legend: bool = True, legend_param: Optional[Dict] = None):
        """Apply legend if requested, keeping default behavior intact."""
        if not show_legend:
            return
        if legend_param is None:
            ax.legend(loc="best")
        else:
            ax.legend(**legend_param)

    def plot(
        self,
        data: Union[pd.DataFrame, Dict[str, List[float]]],
        variables: Optional[List[str]] = None,
        y_labels: Optional[List[str]] = None,
        y_limits: Optional[List[Optional[tuple]]] = None,
        y_ticks: Optional[List[Optional[int]]] = None,
        x_label: Optional[str] = None,
        x_axis_format: Optional[str] = "float",
        x_axis_ticks_format: Optional[str] = None,
        tick_rotation: int = 45,
        interval: int = 24,
        x_limits: Optional[List[Optional[tuple]]] = None,
        colors: Optional[List[str]] = None,
        alphas: Optional[List[float]] = None,
        linestyles: Optional[List[str]] = None,
        markers: Optional[List[Optional[str]]] = None,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        grid: bool = False,  # New parameter for grid,
        show_plot: bool = True, # New parameter for showing the plot or not (for saving purposes)
        show_legend: bool = True,
        legend_param: Optional[Dict] = None,  # New parameter for legend customization
        layout_param: Optional[Dict] = None  # New parameter for layout customization
    ):
        """
        Plot data with flexibility for datetime or float x-axis.

        Args:
            data: DataFrame or dictionary containing the data.
            variables: List of variables to plot. If None, all columns are plotted.
            y_labels: List of y-axis labels.
            y_limits: List of y-axis limits (tuple) for each variable.
            y_ticks: List of numbers of ticks for the y-axis for each variable.
            x_label: Label for the x-axis.
            x_axis_format: Format of the x-axis ("datetime" or "float").
            tick_rotation: Rotation angle for x-axis ticks.
            interval: Interval for datetime tick marks.
            x_limits: List of x-axis limits (tuple) for each variable.
            colors: List of colors for each variable.
            alphas: List of alpha values for each variable.
            linestyles: List of linestyles for each variable.
            markers: List of markers for each variable.
            title: Title of the plot.
            figsize: Size of the figure (width, height).
            grid: Enable or disable the grid for all subplots.
        """
        figsize = figsize or self.default_figsize

        if isinstance(data, pd.DataFrame) and 'Timestamp' in data.columns:
            x = pd.to_datetime(data['Timestamp'])
            data = data.drop(columns='Timestamp')
        elif isinstance(data, pd.DataFrame) and 'Timestamp' not in data.columns:
            x = data.index
        elif isinstance(data, dict) and 'Timestamp' in data.keys():
            x = pd.to_datetime(data['Timestamp'])
            del data['Timestamp']
            data = pd.DataFrame(data)
        elif isinstance(data, dict) and 'Timestamp' not in data.keys():
            x = np.arange(0,max([len(data[key]) for key in data]))
            data = pd.DataFrame(data)
        else:
            raise TypeError("Data must be a DataFrame or a dictionary.")

        if variables is None:
            variables = data.columns.tolist()

        num_vars = len(variables)
        y_labels = y_labels or variables
        y_limits = y_limits or [None] * num_vars
        y_ticks = y_ticks or [None] * num_vars
        x_limits = x_limits or [None] * num_vars
        colors = colors or plt.get_cmap('tab10').colors[:num_vars]
        alphas = alphas or [self.default_alpha] * num_vars
        linestyles = linestyles or [self.default_linestyle] * num_vars
        markers = markers or [self.default_marker] * num_vars

        fig, axes = plt.subplots(num_vars, 1, figsize=figsize, sharex=True)

        if num_vars == 1:
            axes = [axes]  # Ensure axes is always iterable

        for i, var in enumerate(variables):
            ax = axes[i]
            ax.plot(
                x,
                data[var],
                color=colors[i],
                alpha=alphas[i],
                linestyle=linestyles[i],
                marker=markers[i],
                label=var,
            )
            ax.set_ylabel(y_labels[i])
            if y_limits[i]:
                ax.set_ylim(y_limits[i])
            if x_limits[i]:
                if x_axis_format == "datetime":
                    start_time, end_time = x_limits[i]
                    ax.set_xlim(mdates.date2num(start_time), mdates.date2num(end_time))
                else:
                    ax.set_xlim(x_limits[i])
            if y_ticks[i]:
                ax.yaxis.set_major_locator(plt.MaxNLocator(y_ticks[i]))
            if grid:
                ax.grid(True, alpha=0.5)  # Enable grid if specified
            self._apply_legend(ax, show_legend, legend_param)

        if x_axis_format == "datetime":
            for ax in axes:
                maj_formatter = x_axis_ticks_format or "%Y-%m-%d %H:%M"
                ax.xaxis.set_major_formatter(mdates.DateFormatter(maj_formatter))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
                ax.tick_params(axis="x", rotation=tick_rotation)
            x_label = x_label or "DateTime"

        x_label = x_label or "X-axis"
        axes[-1].set_xlabel(x_label)

        if title:
            fig.suptitle(title, fontsize=16)
        if layout_param is None:
            plt.tight_layout() # Adjust layout to make space for the legend
        else:
            plt.tight_layout(**layout_param)

        if show_plot:
            plt.show()

    def plot_single(
        self,
        data: Union[pd.DataFrame, Dict[str, List[float]]],
        variables: Optional[List[str]] = None,
        y_label: Optional[str] = None,
        y_limit: Optional[tuple] = None,
        y_ticks: Optional[int] = None,
        x_label: Optional[str] = None,
        x_axis_format: Optional[str] = "float",
        x_axis_ticks_format: Optional[str] = None,
        tick_rotation: int = 45,
        interval: int = 24,
        x_limit: Optional[tuple] = None,
        colors: Optional[List[str]] = None,
        alphas: Optional[List[float]] = None,
        linestyles: Optional[List[str]] = None,
        markers: Optional[List[Optional[str]]] = None,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        grid: bool = False,  # New parameter for grid
        show_plot: bool = True, # New parameter for showing the plot or not (for saving purposes)
        show_legend: bool = True,
        legend_param: Optional[Dict] = None,  # New parameter for legend customization
        layout_param: Optional[Dict] = None  # New parameter for layout customization
    ):
        """
        Plot multiple variables on a single plot.

        Args:
            data: DataFrame or dictionary containing the data.
            variables: List of variables to plot. If None, all columns are plotted.
            y_label: Label for the y-axis.
            y_limit: Limits for the y-axis (tuple).
            y_ticks: Number of ticks for the y-axis.
            x_label: Label for the x-axis.
            x_axis_format: Format of the x-axis ("datetime" or "float").
            tick_rotation: Rotation angle for x-axis ticks.
            interval: Interval for datetime tick marks.
            x_limit: Limits for the x-axis (tuple).
            colors: List of colors for each variable.
            alphas: List of alpha values for each variable.
            linestyles: List of linestyles for each variable.
            markers: List of markers for each variable.
            title: Title of the plot.
            figsize: Size of the figure (width, height).
            grid: Enable or disable the grid.
        """
        figsize = figsize or self.default_figsize

        if isinstance(data, pd.DataFrame) and 'Timestamp' in data.columns:
            x = pd.to_datetime(data['Timestamp'])
            data = data.drop(columns='Timestamp')
        elif isinstance(data, pd.DataFrame) and 'Timestamp' not in data.columns:
            x = data.index
        elif isinstance(data, dict) and 'Timestamp' in data.keys():
            x = pd.to_datetime(data['Timestamp'])
            del data['Timestamp']
            data = pd.DataFrame(data)
        elif isinstance(data, dict) and 'Timestamp' not in data.keys():
            x = np.arange(0,max([len(data[key]) for key in data]))
            data = pd.DataFrame(data)
        else:
            raise TypeError("Data must be a DataFrame or a dictionary.")

        if variables is None:
            variables = data.columns.tolist()

        colors = colors or plt.get_cmap('tab10').colors[:len(variables)]
        alphas = alphas or [self.default_alpha] * len(variables)
        linestyles = linestyles or [self.default_linestyle] * len(variables)
        markers = markers or [self.default_marker] * len(variables)

        fig, ax = plt.subplots(figsize=figsize)

        for i, var in enumerate(variables):
            ax.plot(
                x,
                data[var],
                color=colors[i],
                alpha=alphas[i],
                linestyle=linestyles[i],
                marker=markers[i],
                label=var,
            )

        ax.set_ylabel(y_label or "Y-axis")
        if y_limit:
            ax.set_ylim(y_limit)
        if x_limit:
            if x_axis_format == "datetime":
                start_time, end_time = x_limit
                ax.set_xlim(mdates.date2num(start_time), mdates.date2num(end_time))
            else:
                ax.set_xlim(x_limit)
        if y_ticks:
            ax.yaxis.set_major_locator(plt.MaxNLocator(y_ticks))
        if grid:
            ax.grid(True, alpha=0.5)  # Enable grid if specified
        self._apply_legend(ax, show_legend, legend_param)

        if x_axis_format == "datetime":
            maj_formatter = x_axis_ticks_format or "%Y-%m-%d %H:%M"
            ax.xaxis.set_major_formatter(mdates.DateFormatter(maj_formatter))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
            ax.tick_params(axis="x", rotation=tick_rotation)
            x_label = x_label or "DateTime"

        ax.set_xlabel(x_label or "X-axis")

        if title:
            ax.set_title(title, fontsize=16)
        if layout_param is None:
            plt.tight_layout() # Adjust layout to make space for the legend
        else:
            plt.tight_layout(**layout_param)

        if show_plot:
            plt.show()

    def plot_variable_across_dfs(
            self,
            data: Dict[str, pd.DataFrame],
            variable: str,
            timestamp_col: str = "Timestamp",
            y_label: Optional[str] = None,
            y_limit: Optional[tuple] = None,
            y_ticks: Optional[int] = None,
            x_label: Optional[str] = None,
            x_axis_format: Optional[str] = "datetime",
            x_axis_ticks_format: Optional[str] = None,
            tick_rotation: int = 45,
            interval: int = 24,
            x_limit: Optional[tuple] = None,
            colors: Optional[List[str]] = None,
            alphas: Optional[List[float]] = None,
            linestyles: Optional[List[str]] = None,
            markers: Optional[List[Optional[str]]] = None,
            title: Optional[str] = None,
            figsize: Optional[Tuple[int, int]] = None,
            grid: bool = False,  # New parameter for grid
            show_plot: bool = True, # New parameter for showing the plot or not (for saving purposes)
            show_legend: bool = True,
            legend_param: Optional[Dict] = None,  # New parameter for legend customization
            layout_param: Optional[Dict] = None  # New parameter for layout customization
        ):
            """
            Plot a specific variable across multiple DataFrames with different time spans.

            Args:
                data: Dictionary containing the DataFrames. Each key is a label for the DataFrame.
                variable: The variable to plot.
                timestamp_col: The name of the timestamp column in the DataFrames.
                y_label: Label for the y-axis.
                y_limit: Limits for the y-axis (tuple).
                y_ticks: Number of ticks for the y-axis.
                x_label: Label for the x-axis.
                x_axis_format: Format of the x-axis ("datetime" or "float").
                tick_rotation: Rotation angle for x-axis ticks.
                interval: Interval for datetime tick marks.
                x_limit: Limits for the x-axis (tuple).
                colors: List of colors for each DataFrame.
                alphas: List of alpha values for each DataFrame.
                linestyles: List of linestyles for each DataFrame.
                markers: List of markers for each DataFrame.
                title: Title of the plot.
                figsize: Size of the figure (width, height).
                grid: Enable or disable the grid.
            """
            figsize = figsize or self.default_figsize

            colors = colors or plt.get_cmap('tab10').colors[:len(data)]
            alphas = alphas or [self.default_alpha] * len(data)
            linestyles = linestyles or [self.default_linestyle] * len(data)
            markers = markers or [self.default_marker] * len(data)

            fig, ax = plt.subplots(figsize=figsize)

            for i, (label, df) in enumerate(data.items()):
                if timestamp_col not in df.columns or variable not in df.columns:
                    raise ValueError(f"DataFrame {label} must contain columns '{timestamp_col}' and '{variable}'")

                ax.plot(
                    df[timestamp_col],
                    df[variable],
                    color=colors[i],
                    alpha=alphas[i],
                    linestyle=linestyles[i],
                    marker=markers[i],
                    label=label,
                )

            ax.set_ylabel(y_label or variable)
            if y_limit:
                ax.set_ylim(y_limit)
            if x_limit:
                if x_axis_format == "datetime":
                    start_time, end_time = x_limit
                    ax.set_xlim(mdates.date2num(start_time), mdates.date2num(end_time))
                else:
                    ax.set_xlim(x_limit)
            if y_ticks:
                ax.yaxis.set_major_locator(plt.MaxNLocator(y_ticks))
            if grid:
                ax.grid(True, alpha=0.5)  # Enable grid if specified
            self._apply_legend(ax, show_legend, legend_param)

            if x_axis_format == "datetime":
                maj_formatter = x_axis_ticks_format or "%Y-%m-%d %H:%M"
                ax.xaxis.set_major_formatter(mdates.DateFormatter(maj_formatter))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
                ax.tick_params(axis="x", rotation=tick_rotation)
                x_label = x_label or "DateTime"

            ax.set_xlabel(x_label or "X-axis")

            if title:
                ax.set_title(title, fontsize=16)
            if layout_param is None:
                plt.tight_layout() # Adjust layout to make space for the legend
            else:
                plt.tight_layout(**layout_param)

            if show_plot:
                plt.show()

    def plot_multiple_variables_across_dfs(
        self,
        data: Dict[str, pd.DataFrame],
        variables: Dict[str, List[str]],
        timestamp_col: str = "Timestamp",
        y_label: Optional[str] = None,
        y_limit: Optional[tuple] = None,
        y_ticks: Optional[int] = None,
        x_label: Optional[str] = None,
        x_axis_format: Optional[str] = "datetime",
        x_axis_ticks_format: Optional[str] = None,
        tick_rotation: int = 45,
        interval: int = 24,
        x_limit: Optional[tuple] = None,
        colors: Optional[Dict[str, List[str]]] = None,
        alphas: Optional[Dict[str, List[float]]] = None,
        linestyles: Optional[Dict[str, List[str]]] = None,
        markers: Optional[Dict[str, List[Optional[str]]]] = None,
        plot_types: Optional[Dict[str, List[str]]] = None,  # New parameter for plot types
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        grid: bool = False,  # New parameter for grid
        show_plot: bool = True, # New parameter for showing the plot or not (for saving purposes)
        show_legend: bool = True,
        legend_param: Optional[Dict] = None,  # New parameter for legend customization
        layout_param: Optional[Dict] = None  # New parameter for layout customization
    ):
        """
        Plot different variables across multiple DataFrames with different time spans.

        Args:
            data: Dictionary containing the DataFrames. Each key is a label for the DataFrame.
            variables: Dictionary specifying the variables to plot for each DataFrame.
            timestamp_col: The name of the timestamp column in the DataFrames.
            y_label: Label for the y-axis.
            y_limit: Limits for the y-axis (tuple).
            y_ticks: Number of ticks for the y-axis.
            x_label: Label for the x-axis.
            x_axis_format: Format of the x-axis ("datetime" or "float").
            tick_rotation: Rotation angle for x-axis ticks.
            interval: Interval for datetime tick marks.
            x_limit: Limits for the x-axis (tuple).
            colors: Dictionary of colors for each DataFrame's variables.
            alphas: Dictionary of alpha values for each DataFrame's variables.
            linestyles: Dictionary of linestyles for each DataFrame's variables.
            markers: Dictionary of markers for each DataFrame's variables.
            plot_types: Dictionary of plot types ('line' or 'scatter') for each DataFrame's variables.
            title: Title of the plot.
            figsize: Size of the figure (width, height).
            grid: Enable or disable the grid.
        """
        figsize = figsize or self.default_figsize

        fig, ax = plt.subplots(figsize=figsize)

        for i, (label, df) in enumerate(data.items()):
            if timestamp_col not in df.columns:
                raise ValueError(f"DataFrame {label} must contain the column '{timestamp_col}'")
            for j, variable in enumerate(variables.get(label, [])):
                if variable not in df.columns:
                    self.logger.info(f"Warning: Variable '{variable}' not found in DataFrame '{label}'")
                    continue

                color = colors[label][j] if colors and label in colors and j < len(colors[label]) else plt.get_cmap('tab10')(j % 10)
                alpha = alphas[label][j] if alphas and label in alphas and j < len(alphas[label]) else self.default_alpha
                linestyle = linestyles[label][j] if linestyles and label in linestyles and j < len(linestyles[label]) else self.default_linestyle
                marker = markers[label][j] if markers and label in markers and j < len(markers[label]) else self.default_marker
                plot_type = plot_types[label][j] if plot_types and label in plot_types and j < len(plot_types[label]) else 'line'

                if plot_type == 'scatter':
                    ax.scatter(
                        df[timestamp_col],
                        df[variable],
                        color=color,
                        alpha=alpha,
                        marker=marker,
                        label=f"{label} - {variable}",
                    )
                else:
                    ax.plot(
                        df[timestamp_col],
                        df[variable],
                        color=color,
                        alpha=alpha,
                        linestyle=linestyle,
                        marker=marker,
                        label=f"{label} - {variable}",
                    )

        ax.set_ylabel(y_label or "Value")
        if y_limit:
            ax.set_ylim(y_limit)
        if x_limit:
            if x_axis_format == "datetime":
                start_time, end_time = x_limit
                ax.set_xlim(mdates.date2num(start_time), mdates.date2num(end_time))
            else:
                ax.set_xlim(x_limit)
        if y_ticks:
            ax.yaxis.set_major_locator(plt.MaxNLocator(y_ticks))
        if grid:
            ax.grid(True, alpha=0.5)  # Enable grid if specified
        self._apply_legend(ax, show_legend, legend_param)

        if x_axis_format == "datetime":
            maj_formatter = x_axis_ticks_format or "%Y-%m-%d %H:%M"
            ax.xaxis.set_major_formatter(mdates.DateFormatter(maj_formatter))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
            ax.tick_params(axis="x", rotation=tick_rotation)
            x_label = x_label or "DateTime"

        ax.set_xlabel(x_label or "X-axis")

        if title:
            ax.set_title(title, fontsize=16)
        if layout_param is None:
            plt.tight_layout() # Adjust layout to make space for the legend
        else:
            plt.tight_layout(**layout_param)

        if show_plot:
            plt.show()

    def generate_plots_with_incremental_variables(
        self,
        data: Dict[str, pd.DataFrame],
        variables: List[List[Tuple[str, str]]],  # List of lists of (label, variable) tuples
        output_folder: str,
        run_index: float,
        timestamp_col: str = "Timestamp",
        y_label: Optional[str] = None,
        y_limit: Optional[tuple] = None,
        y_ticks: Optional[int] = None,
        x_label: Optional[str] = None,
        x_axis_format: Optional[str] = "datetime",
        x_axis_ticks_format: Optional[str] = None,
        tick_rotation: int = 45,
        interval: int = 24,
        x_limit: Optional[tuple] = None,
        vertical_lines: Optional[List[Union[float, pd.Timestamp]]] = None,  # <-- changed
        horizontal_lines: Optional[List[float]] = None,  # <-- new parameter    
        colors: Optional[Dict[str, List[str]]] = None,
        alphas: Optional[Dict[str, List[float]]] = None,
        linestyles: Optional[Dict[str, List[str]]] = None,
        markers: Optional[Dict[str, List[Optional[str]]]] = None,
        plot_types: Optional[Dict[str, List[str]]] = None,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        grid: bool = False,
        show_legend: bool = True,
        legend_param: Optional[Dict] = None,  # New parameter for legend customization
        layout_param: Optional[Dict] = None  # New parameter for layout customization
    ):
        figsize = figsize or self.default_figsize

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        fig, ax = plt.subplots(figsize=figsize)

        # Track the global index for each label
        label_counters = defaultdict(int)
        # Generate plots incrementally
        #for i in range(1, len(variables) + 1):
        for i in [len(variables)]:  # Only the last plot
            ax.clear()
            for variable_group in variables[:i]:
                for label, var in variable_group:
                    df = data[label]
                    if timestamp_col not in df.columns or var not in df.columns:
                        self.logger.info(f"Warning: Variable '{var}' not found in DataFrame '{label}'")
                        continue

                    idx = label_counters[label]
                    color = colors[label][idx] if colors and label in colors and idx < len(colors[label]) else plt.get_cmap('tab10')(idx % 10)
                    alpha = alphas[label][idx] if alphas and label in alphas and idx < len(alphas[label]) else self.default_alpha
                    linestyle = linestyles[label][idx] if linestyles and label in linestyles and idx < len(linestyles[label]) else self.default_linestyle
                    marker = markers[label][idx] if markers and label in markers and idx < len(markers[label]) else self.default_marker
                    plot_type = plot_types[label][idx] if plot_types and label in plot_types and idx < len(plot_types[label]) else 'line'

                    if plot_type == 'scatter':
                        ax.scatter(
                            df[timestamp_col],
                            df[var],
                            color=color,
                            alpha=alpha,
                            marker=marker,
                            label=f"{label} - {var}",
                        )
                    elif plot_type == 'step':
                        ax.step(
                            df[timestamp_col],
                            df[var],
                            color=color,
                            alpha=alpha,
                            linestyle=linestyle,
                            where='post',
                            label=f"{label} - {var}",
                        )
                    else:
                        ax.plot(
                            df[timestamp_col],
                            df[var],
                            color=color,
                            alpha=alpha,
                            linestyle=linestyle,
                            marker=marker,
                            label=f"{label} - {var}",
                        )
                    label_counters[label] += 1  # Increment the counter for this label

            ax.set_ylabel(y_label or "Value")
            if y_limit:
                ax.set_ylim(y_limit)
            if x_limit:
                if x_axis_format == "datetime":
                    start_time, end_time = x_limit
                    ax.set_xlim(mdates.date2num(start_time), mdates.date2num(end_time))
                else:
                    ax.set_xlim(x_limit)
            if y_ticks:
                ax.yaxis.set_major_locator(plt.MaxNLocator(y_ticks))
            if grid:
                ax.grid(True, alpha=0.5)
            self._apply_legend(ax, show_legend, legend_param)

            if x_axis_format == "datetime":
                maj_formatter = x_axis_ticks_format or "%Y-%m-%d %H:%M"
                ax.xaxis.set_major_formatter(mdates.DateFormatter(maj_formatter))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
                ax.tick_params(axis="x", rotation=tick_rotation)
                x_label = x_label or "DateTime"

            ax.set_xlabel(x_label or "X-axis")

            if vertical_lines is not None:
                for vline in vertical_lines:
                    if x_axis_format == "datetime" and isinstance(vline, datetime):
                        ax.axvline(mdates.date2num(vline), color='grey', linestyle='--')
                    elif x_axis_format == "float" and isinstance(vline, (int, float)):
                        ax.axvline(vline, color='grey', linestyle='--')
            if horizontal_lines is not None:
                for hline in horizontal_lines:
                    ax.axhline(hline, color='grey', linestyle='dotted')

            if title:
                ax.set_title(title, fontsize=16)
            if layout_param is None:
                plt.tight_layout() # Adjust layout to make space for the legend
            else:
                plt.tight_layout(**layout_param)

            plt.savefig(os.path.join(output_folder, f"plot_{title}_run{run_index}_{i}.png"), bbox_inches='tight')

        plt.close(fig)

    def plot_grouped_variables_across_dfs(
        self,
        data: Dict[str, pd.DataFrame],
        variable_groups: List[List[Tuple[str, str]]],
        timestamp_col: str = "Timestamp",
        y_labels: Optional[List[str]] = None,
        y_limits: Optional[List[Optional[tuple]]] = None,
        y_ticks: Optional[List[Optional[int]]] = None,
        x_label: Optional[str] = None,
        x_axis_format: Optional[str] = "datetime",
        x_axis_ticks_format: Optional[str] = None,
        tick_rotation: int = 45,
        interval: int = 24,
        x_limits: Optional[List[Optional[tuple]]] = None,
        colors: Optional[Dict[str, List[str]]] = None,
        alphas: Optional[Dict[str, List[float]]] = None,
        linestyles: Optional[Dict[str, List[str]]] = None,
        markers: Optional[Dict[str, List[Optional[str]]]] = None,
        plot_types: Optional[List[List[str]]] = None,  # List of plot types for each variable in each subplot
        figsize: Optional[Tuple[int, int]] = None,
        grid: bool = False,
        title: Optional[str] = None,
        show_plot: bool = True,
        show_legend: bool = True,
        legend_param: Optional[Dict] = None,  # New parameter for legend customization
        layout_param: Optional[Dict] = None,  # New parameter for layout customization
        vertical_lines: Optional[List[Optional[List[Union[float, pd.Timestamp]]]]] = None,  # NEW
        horizontal_lines: Optional[List[Optional[List[float]]]] = None  # NEW
    ):
        """
        Plot grouped variables from different DataFrames on separate subplots.

        Args:
            data: Dictionary containing the DataFrames. Each key is a label for the DataFrame.
            variable_groups: List of groups, where each group is a list of (label, variable) tuples.
            timestamp_col: The name of the timestamp column in the DataFrames.
            y_labels: List of y-axis labels for each subplot.
            y_limits: List of y-axis limits (tuple) for each subplot.
            y_ticks: List of numbers of ticks for the y-axis for each subplot.
            x_label: Label for the x-axis.
            x_axis_format: Format of the x-axis ("datetime" or "float").
            tick_rotation: Rotation angle for x-axis ticks.
            interval: Interval for datetime tick marks.
            x_limits: List of x-axis limits (tuple) for each subplot.
            colors: Dictionary of colors for each DataFrame's variables.
            alphas: Dictionary of alpha values for each DataFrame's variables.
            linestyles: Dictionary of linestyles for each DataFrame's variables.
            markers: Dictionary of markers for each DataFrame's variables.
            plot_types: List of plot types ('line' or 'scatter') for each variable in each subplot.
            figsize: Size of the figure (width, height).
            grid: Enable or disable the grid for all subplots.
            title: Title of the plot.
            show_plot: Whether to display the plot.
        """
        figsize = figsize or self.default_figsize
        num_subplots = len(variable_groups)
        y_labels = y_labels or [f"Group {i+1}" for i in range(num_subplots)]
        y_limits = y_limits or [None] * num_subplots
        y_ticks = y_ticks or [None] * num_subplots
        x_limits = x_limits or [None] * num_subplots
        plot_types = plot_types or [["line"] * len(group) for group in variable_groups]
        vertical_lines = vertical_lines or [None] * num_subplots
        horizontal_lines = horizontal_lines or [None] * num_subplots

        fig, axes = plt.subplots(num_subplots, 1, figsize=figsize, sharex=True)

        if num_subplots == 1:
            axes = [axes]  # Ensure axes is always iterable

        for i, group in enumerate(variable_groups):
            ax = axes[i]
            for j, (label, variable) in enumerate(group):
                if label not in data or variable not in data[label].columns:
                    raise ValueError(f"DataFrame '{label}' must contain the variable '{variable}'")

                df = data[label]
                if timestamp_col not in df.columns:
                    raise ValueError(f"DataFrame '{label}' must contain the column '{timestamp_col}'")

                color = colors[label][j] if colors and label in colors and j < len(colors[label]) else plt.get_cmap('tab10')(j % 10)
                alpha = alphas[label][j] if alphas and label in alphas and j < len(alphas[label]) else self.default_alpha
                linestyle = linestyles[label][j] if linestyles and label in linestyles and j < len(linestyles[label]) else self.default_linestyle
                marker = markers[label][j] if markers and label in markers and j < len(markers[label]) else self.default_marker
                plot_type = plot_types[i][j] if i < len(plot_types) and j < len(plot_types[i]) else 'line'

                if plot_type == 'scatter':
                    ax.scatter(
                        df[timestamp_col],
                        df[variable],
                        color=color,
                        alpha=alpha,
                        marker=marker,
                        label=f"{label} - {variable}"
                    )
                elif plot_type == 'step':
                    ax.step(
                        df[timestamp_col],
                        df[variable],
                        color=color,
                        alpha=alpha,
                        linestyle=linestyle,
                        where='post',
                        label=f"{label} - {variable}",
                    )
                else:
                    ax.plot(
                        df[timestamp_col],
                        df[variable],
                        color=color,
                        alpha=alpha,
                        linestyle=linestyle,
                        marker=marker,
                        label=f"{label} - {variable}"
                    )

            ax.set_ylabel(y_labels[i])
            if y_limits[i]:
                ax.set_ylim(y_limits[i])
            if y_ticks[i]:
                ax.yaxis.set_major_locator(plt.MaxNLocator(y_ticks[i]))
            if x_limits[i]:
                if x_axis_format == "datetime":
                    start_time, end_time = x_limits[i]
                    ax.set_xlim(mdates.date2num(start_time), mdates.date2num(end_time))
                else:
                    ax.set_xlim(x_limits[i])
            if grid:
                ax.grid(True, alpha=0.5)
            self._apply_legend(ax, show_legend, legend_param)

            # --- Add vertical lines for this subplot ---
            if vertical_lines[i]:
                for vline in vertical_lines[i]:
                    if x_axis_format == "datetime" and isinstance(vline, datetime):
                        ax.axvline(mdates.date2num(vline), color='grey', linestyle='--')
                    elif x_axis_format == "float" and isinstance(vline, (int, float)):
                        ax.axvline(vline, color='grey', linestyle='--')
            # --- Add horizontal lines for this subplot ---
            if horizontal_lines[i]:
                for hline in horizontal_lines[i]:
                    ax.axhline(hline, color='grey', linestyle='dotted')

        if x_axis_format == "datetime":
            for ax in axes:
                maj_formatter = x_axis_ticks_format or "%Y-%m-%d %H:%M"
                ax.xaxis.set_major_formatter(mdates.DateFormatter(maj_formatter))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
                ax.tick_params(axis="x", rotation=tick_rotation)
            x_label = x_label or "DateTime"

        axes[-1].set_xlabel(x_label or "X-axis")

        if title:
            fig.suptitle(title, fontsize=16)
        if layout_param is None:
            plt.tight_layout() # Adjust layout to make space for the legend
        else:
            plt.tight_layout(**layout_param)

        if show_plot:
            plt.show()

    def plot_multiple_variables_across_dfs_with_labels(
        self,
        data: Dict[str, pd.DataFrame],
        variables: Dict[str, List[str]],
        timestamp_col: str = "Timestamp",
        y_label: Optional[str] = None,
        y_limit: Optional[tuple] = None,
        y_ticks: Optional[int] = None,
        x_label: Optional[str] = None,
        x_axis_format: Optional[str] = "datetime",
        x_axis_ticks_format: Optional[str] = None,
        tick_rotation: int = 45,
        interval: int = 24,
        x_limit: Optional[tuple] = None,
        vertical_lines: Optional[List[Union[float, pd.Timestamp]]] = None,  # <-- changed
        horizontal_lines: Optional[List[float]] = None,  # <-- new parameter    
        colors: Optional[Dict[str, List[str]]] = None,
        alphas: Optional[Dict[str, List[float]]] = None,
        linestyles: Optional[Dict[str, List[str]]] = None,
        markers: Optional[Dict[str, List[Optional[str]]]] = None,
        plot_types: Optional[Dict[str, List[str]]] = None,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        grid: bool = False,
        show_plot: bool = True,
        show_legend: bool = True,
        legend_param: Optional[Dict] = None,
        layout_param: Optional[Dict] = None
    ):
        """
        Plot different variables across multiple DataFrames with interactive labels using mplcursors.
        """
        figsize = figsize or self.default_figsize

        fig, ax = plt.subplots(figsize=figsize)

        for i, (label, df) in enumerate(data.items()):
            if timestamp_col not in df.columns:
                raise ValueError(f"DataFrame {label} must contain the column '{timestamp_col}'")
            for j, variable in enumerate(variables.get(label, [])):
                if variable not in df.columns:
                    self.logger.info(f"Warning: Variable '{variable}' not found in DataFrame '{label}'")
                    continue

                color = colors[label][j] if colors and label in colors and j < len(colors[label]) else plt.get_cmap('tab10')(j % 10)
                alpha = alphas[label][j] if alphas and label in alphas and j < len(alphas[label]) else self.default_alpha
                linestyle = linestyles[label][j] if linestyles and label in linestyles and j < len(linestyles[label]) else self.default_linestyle
                marker = markers[label][j] if markers and label in markers and j < len(markers[label]) else self.default_marker
                plot_type = plot_types[label][j] if plot_types and label in plot_types and j < len(plot_types[label]) else 'line'

                if plot_type == 'scatter':
                    ax.scatter(
                        df[timestamp_col],
                        df[variable],
                        color=color,
                        alpha=alpha,
                        marker=marker,
                        label=f"{label} - {variable}",
                    )
                elif plot_type == 'step':
                    ax.step(
                        df[timestamp_col],
                        df[variable],
                        color=color,
                        alpha=alpha,
                        linestyle=linestyle,
                        marker=marker,
                        where='post',
                        label=f"{label} - {variable}",
                    )
                else:
                    ax.plot(
                        df[timestamp_col],
                        df[variable],
                        color=color,
                        alpha=alpha,
                        linestyle=linestyle,
                        marker=marker,
                        label=f"{label} - {variable}",
                    )

        ax.set_ylabel(y_label or "Value")
        if y_limit:
            ax.set_ylim(y_limit)
        if x_limit:
            if x_axis_format == "datetime":
                start_time, end_time = x_limit
                ax.set_xlim(mdates.date2num(start_time), mdates.date2num(end_time))
            else:
                ax.set_xlim(x_limit)
        if y_ticks:
            ax.yaxis.set_major_locator(plt.MaxNLocator(y_ticks))
        if grid:
            ax.grid(True, alpha=0.5)
        self._apply_legend(ax, show_legend, legend_param)

        if x_axis_format == "datetime":
            maj_formatter = x_axis_ticks_format or "%Y-%m-%d %H:%M"
            ax.xaxis.set_major_formatter(mdates.DateFormatter(maj_formatter))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
            ax.tick_params(axis="x", rotation=tick_rotation)
            x_label = x_label or "DateTime"

        ax.set_xlabel(x_label or "X-axis")

        if vertical_lines is not None:
            for vline in vertical_lines:
                if x_axis_format == "datetime" and isinstance(vline, datetime):
                    ax.axvline(mdates.date2num(vline), color='grey', linestyle='--')
                elif x_axis_format == "float" and isinstance(vline, (int, float)):
                    ax.axvline(vline, color='grey', linestyle='--')
        if horizontal_lines is not None:
            for hline in horizontal_lines:
                ax.axhline(hline, color='grey', linestyle='dotted')

        if title:
            ax.set_title(title, fontsize=16)
        if layout_param is None:
            plt.tight_layout()
        else:
            plt.tight_layout(**layout_param)

        # Add interactive labels with mplcursors
        cursor = mplcursors.cursor(ax.lines, hover=True)

        # Hide annotation when not hovering over a line
        cursor.connect("add", lambda sel: sel.annotation.set_visible(True))
        cursor.connect("remove", lambda sel: sel.annotation.set_visible(False))

        if show_plot:
            plt.show()



