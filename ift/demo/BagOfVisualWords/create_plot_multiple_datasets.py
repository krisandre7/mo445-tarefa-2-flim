import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas
import subprocess
import sys

def exec_subplot_iteration(plt_idx, num_plts, cell, sep_char, columns_plot, first_row_names,
                            log_scale, individual_subplot_titles, minor_ticks, axes_ticks_rotation,
                            x_vals_as_ticks, y_vals_as_ticks, x_range, y_range, x_grid, y_grid,
                            no_error_bars, line_format, line_format_list):
    if plt_idx < num_plts:
        # read the data from the csvs
        dataset = input_csv[0][plt_idx]

        if first_row_names:
            data = pandas.read_csv(dataset, sep=sep_char)
        else:
            data = pandas.read_csv(dataset, sep=sep_char, header=None)

        # setup the log scale
        if log_scale == 'x-axis':
            cell.set_xscale('log')
        elif log_scale == 'y-axis':
            cell.set_yscale('log')
        elif log_scale == 'x-y-axis':
            cell.set_xscale('log')
            cell.set_yscale('log')

        # verify columns to plot
        if columns_plot == 'all':
            columns_plot_list = range(1, len(data.columns))
        else:
            columns_plot_list = eval(columns_plot)

        # minor ticks in axes
        if minor_ticks:
            cell.minorticks_on()
        else:
            cell.minorticks_off()

        # set x and y limit values
        if x_range != []:
            cell.set_xlim(x_range[0],x_range[1])
        if y_range != []:
            cell.set_ylim(y_range[0],y_range[1])

        # set grid lines
        if x_grid:
            cell.xaxis.grid(True)
        if y_grid:
            cell.yaxis.grid(True)

        # set the format for the lines
        if line_format_list != []:
            if len(line_format_list) != len(columns_plot_list):
                sys.exit('Error: The list with line formats must have the same length as the list of columns to plot (it is different for)')
        else:
            line_format_list = [line_format for i in range(len(columns_plot_list))]

        # create the data series
        x = data[data.columns[0]].to_numpy('float')

        # use x-values as ticks
        if x_vals_as_ticks:
            cell.set_xticks(x)
            cell.set_xticklabels(x, rotation=axes_ticks_rotation)

        # set the individual title for the subplot
        if len(individual_subplot_titles) > 0:
            cell.set_title(individual_subplot_titles[plt_idx])

        # set other params for the subplot
        cell.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

        for i in range(len(columns_plot_list)):
            col = data[data.columns[columns_plot_list[i]]]
            
            # verify the use of error bars
            if type(col[0]) == str and col[0].find("+-") != -1:
                y = [float(val.split('+-')[0]) for val in col]
                e = [float(val.split('+-')[1]) for val in col]
                if no_error_bars:
                    cell.plot(x, y, line_format_list[i])
                else:
                    cell.errorbar(x, y, e, fmt=line_format_list[i], capsize=4)
            else:
                y = col.to_numpy('float')
                cell.plot(x, y, line_format_list[i])

            # use y-values as ticks
            if y_vals_as_ticks:
                cell.yticks(y, rotation=axes_ticks_rotation)
                y_vals_as_ticks = False

    return columns_plot_list, data

#===============#
# main function
#===============#
if __name__ == "__main__":
    # get the size of the terminal window
    terminal_rows, terminal_cols = subprocess.check_output(['stty', 'size']).decode().split()
    terminal_cols = int(terminal_cols)

    # verify input parameters
    parser = argparse.ArgumentParser('Creates a plot with several subplots with multiple series each (from a CSV file for each dataset)')
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-i', '--input_data', type=str, help='CSV (semi-colon) file containing the paths of the CSV files that contain the \
                                x and y values (first column is the x-axis). If the values contain "+-" standard deviation will be considered in the plot', required=True)
    required_named.add_argument("--subplots_matrix", type=str, help="Matrix configuration for the subplots (a two-element list passed as text)")
    required_named.add_argument('-o', '--output_img', type=str, help='Output image (.png)', required=True)
    parser.add_argument("--fig_size", type=str, default="(10,10)", help="Size for the matrix figure that contains the subplots (a 2-tuple passed as text)")
    parser.add_argument("--sep_char", type=str, default=";", help="Separator character for the CSV file")
    parser.add_argument("-c", "--columns_plot", type=str, default="all", help="The columns to be plotted (a list passed as text) starting from 1, \
                        0 is the x-axis")
    parser.add_argument("-r", "--first_row_names", action="store_true", default=False, help="Use the first row as the series names")
    parser.add_argument("-l", "--log_scale", type=str, default="no", choices=['x-axis','y-axis','x-y-axis'], help="Whether or not to use log \
                        scale in the axes")
    parser.add_argument("--common_plot_title", type=str, default="", help="Common title for the entire plot")
    parser.add_argument("--individual_subplot_titles", type=str, default="[]", help="Individual title for each subplot (a list passed as text)")
    parser.add_argument("--minor_ticks", action="store_true", default=False, help="Show minor ticks in the axes")
    parser.add_argument("--axes_ticks_rotation", type=float, default=0, help="Change axes ticks rotation")
    parser.add_argument("--x_vals_as_ticks", action="store_true", default=False, help="Use x-values as x-ticks")
    parser.add_argument("--y_vals_as_ticks", action="store_true", default=False, help="Use y-values as y-ticks (takes the first y-series in the data)")
    parser.add_argument("--x_label", type=str, help="Label for the x-axis")
    parser.add_argument("--y_label", type=str, help="Label for the y-axis")
    parser.add_argument("--x_range", type=str, default="[]", help="Range for the x-values (a two-element list passed as text)")
    parser.add_argument("--y_range", type=str, default="[]", help="Range for the y-values (a two-element list passed as text)")
    parser.add_argument("--x_grid", action="store_true", default=False, help="Show x-axis grid lines")
    parser.add_argument("--y_grid", action="store_true", default=False, help="Show y-axis grid lines")
    parser.add_argument("--no_error_bars", action="store_true", default=False, help="Do not plot error bars")
    parser.add_argument("--line_format", type=str, default="o-", help="Format to be used in the plot lines: a single format for all the series. It must \
                        be a valid matplotlib style/marker/color combination")
    parser.add_argument("--line_format_list", type=str, default="[]", help="Format to be used in the plot lines: a list specifying the format for each \
                        series (a list passed as text). It must be a valid matplotlib style/marker/color combination")
    parser.add_argument("--legend_loc", type=str, default="none", choices=['none','best','upper right','upper left','lower left','lower right','right',
                        'center left','center right','lower center','upper center','center'], help="Legend location inside the plot")
    parser.add_argument("--legend_loc_tuple", type=str, default=None, help="Legend location inside the plot (a 2-tuple passed as text)")
    parser.add_argument("--img_dpi", type=int, default=300, help="DPI resolution for the output image")
    args = parser.parse_args()

    # read input parameters
    input_data = args.input_data
    subplots_matrix = eval(args.subplots_matrix)
    output_img = args.output_img
    fig_size = eval(args.fig_size)
    sep_char = args.sep_char
    columns_plot = args.columns_plot
    first_row_names = args.first_row_names
    log_scale = args.log_scale
    common_plot_title = args.common_plot_title
    individual_subplot_titles = eval(args.individual_subplot_titles)
    minor_ticks = args.minor_ticks
    axes_ticks_rotation = args.axes_ticks_rotation
    x_vals_as_ticks = args.x_vals_as_ticks
    y_vals_as_ticks = args.y_vals_as_ticks
    x_label = args.x_label
    y_label = args.y_label
    x_range = eval(args.x_range)
    y_range = eval(args.y_range)
    x_grid = args.x_grid
    y_grid = args.y_grid
    no_error_bars = args.no_error_bars
    line_format = args.line_format
    line_format_list = eval(args.line_format_list)
    legend_loc = args.legend_loc
    legend_loc_tuple = eval(args.legend_loc_tuple) if args.legend_loc_tuple != None else args.legend_loc_tuple
    img_dpi = args.img_dpi

    # read the input file
    input_csv = pandas.read_csv(input_data, header=None)

    num_plt_rows = subplots_matrix[0]
    num_plt_cols = subplots_matrix[1]
    num_plts = len(input_csv[0])

    if len(individual_subplot_titles)>0 and len(individual_subplot_titles) != num_plts:
        sys.exit("The number of individual subplot titles must be equal to the number of plots in the input CSV")

    # create the plot    
    #fig, _ = plt.subplots()
    fig, axes2d = plt.subplots(num_plt_rows, num_plt_cols, sharex=True, sharey=True, figsize=fig_size)

    # read the csv for each dataset
    plt_idx = 0
    if num_plt_rows == 1 or num_plt_cols == 1:
        for ii, cell in enumerate(axes2d):
            columns_plot_list, data = exec_subplot_iteration(plt_idx, num_plts, cell, sep_char, columns_plot, first_row_names,
                            log_scale, individual_subplot_titles, minor_ticks, axes_ticks_rotation,
                            x_vals_as_ticks, y_vals_as_ticks, x_range, y_range, x_grid, y_grid,
                            no_error_bars, line_format, line_format_list)
            plt_idx += 1

    else:
        for ii, row in enumerate(axes2d):
            for jj, cell in enumerate(row):
                columns_plot_list, data = exec_subplot_iteration(plt_idx, num_plts, cell, sep_char, columns_plot, first_row_names,
                            log_scale, individual_subplot_titles, minor_ticks, axes_ticks_rotation,
                            x_vals_as_ticks, y_vals_as_ticks, x_range, y_range, x_grid, y_grid,
                            no_error_bars, line_format, line_format_list)
                plt_idx += 1

    # set the common title for all the plots
    if len(common_plot_title) > 0:
        fig.suptitle(common_plot_title)

    # create the common x and y labels
    fig.add_subplot(111, frameon=False) # add a big axis, hide frame
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False) # hide tick and tick label of the big axis
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()

    # delete the unused subplots
    for i in range(1, num_plt_rows*num_plt_cols-num_plts+1):
        axes2d.flat[-i].set_visible(False)

    # create the legend
    if legend_loc != 'none' or legend_loc_tuple != None:
        if legend_loc_tuple != None:
            fig.legend(list(data.columns[columns_plot_list]), loc=legend_loc_tuple)
        else:
            fig.legend(list(data.columns[columns_plot_list]), loc=legend_loc)

    plt.savefig(output_img, dpi=img_dpi)