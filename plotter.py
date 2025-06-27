### Plotting Functions

import matplotlib.pyplot as plt
import correlation as corr
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from datetime import datetime
import os



def plot_waterfall(input_cdf_df, input_categorical_var, input_title, fig_width=650, fig_height=500, max_pct=0.95): 
    top_pct = input_cdf_df[input_cdf_df['cum_pct']<=max_pct]

    measure = ['relative'] *len(top_pct)
    measure.append("total")

    x = list(top_pct[input_categorical_var])
    x.append("TOTAL")

    text = list(top_pct['cum_pct'].diff())
    text[0] = top_pct['cum_pct'].iloc[0]
    text.append(top_pct['cum_pct'].iloc[-1])
    text = [str(np.round(x,3)) for x in text]

    y = list(top_pct['cum_pct'].diff())
    y[0] = top_pct['cum_pct'].iloc[0]
    y.append(0)


    fig = go.Figure(go.Waterfall(
    name = "", orientation = "v",
    measure = measure, 
    x = x,
    textposition = "outside",
    text =  text,
    y = y,
    connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))

    fig.update_layout(
        width=fig_width,
        height=fig_height,
        title = input_title ,
        showlegend = True
    )
    fig.update_xaxes(title_text='{}'.format(input_categorical_var))
    fig.update_yaxes(title_text='Pct over total')


    fig.show()


def plot_waterfall_breakdown(input_categorical_var, input_action,input_sql_table, title, fig_width=650, fig_height=500, max_pct=0.95): 
    
    input_categorical_var_counts = query_to_df(f"select {input_categorical_var}, sum(sum_{input_action}) as num_instances from {input_sql_table} group by 1 ")
    input_categorical_var_counts.sort_values(by='num_instances', ascending=False, inplace=True)
    input_categorical_var_counts['pct_over_total'] = input_categorical_var_counts['num_instances']/input_categorical_var_counts['num_instances'].sum()
    input_categorical_var_counts['cum_pct'] = input_categorical_var_counts['pct_over_total'].cumsum()
    top_pct = input_categorical_var_counts[input_categorical_var_counts['cum_pct']<=max_pct]

    measure = ['relative'] *len(top_pct)
    measure.append("total")
    
    x = list(top_pct[input_categorical_var])
    x.append("TOTAL")

    text = list(top_pct['cum_pct'].diff())
    text[0] = top_pct['cum_pct'].iloc[0]
    text.append(top_pct['cum_pct'].iloc[-1])
    text = [str(np.round(x,3)) for x in text]

    y = list(top_pct['cum_pct'].diff())
    y[0] = top_pct['cum_pct'].iloc[0]
    y.append(0)


    fig = go.Figure(go.Waterfall(
        name = "", orientation = "v",
        measure = measure, 
        x = x,
        textposition = "outside",
        text =  text,
        y = y,
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))

    fig.update_layout(
            width=fig_width,
            height=fig_height,
            title = title ,
            showlegend = True
    )
    fig.update_xaxes(title_text='{}'.format(input_categorical_var))
    fig.update_yaxes(title_text='Pct over total')


    fig.show()
    
    return input_categorical_var_counts


def plot_lineplot(x, y, title='', xlabel='', ylabel=''):
    plt.figure(figsize=(20, 10))
    plt.plot(x, y)
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)

    if not os.path.exists('plots'):
        os.makedirs('plots')

    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y_%H:%M:%S.%f)")
    plt.savefig("plots/{}_{}.png".format(title, timestampStr))

    # plt.show()

def plot_scatterplot(x, y, title='', xlabel='', ylabel=''):
    plt.figure(figsize=(20, 10))
    plt.scatter(x, y)
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)

    if not os.path.exists('plots'):
        os.makedirs('plots')

    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y_%H:%M:%S.%f)")
    plt.savefig("plots/{}_{}.png".format(title, timestampStr))

    plt.show()

    
def plot_plotly_scatterplot(x_list, y_list, fig_width, fig_height, fig_title='', x_axis_text='', y_axis_text='', xtick_angle=90): 
    fig = px.scatter(x=x_list, y=y_list)
    fig.update_layout(
        width=fig_width,
        height=fig_height,
        title = fig_title ,
        showlegend = True
    )
    fig.update_xaxes(title_text=x_axis_text, tickangle=90)
    fig.update_yaxes(title_text=y_axis_text)
    fig.show()
    
def plot_correlation(df, var_name, target_var='fraud', min_correlation=0.05):
    """
    Generates a plot between the values of a column in a dataframe and the target variable.
    It filters out irrelevant values to only keep those above a min_correlation (absolute value).
    """
    title = f"Correlation between {var_name} and {target_var}"
    df3 = corr.get_correlation_df(df, var_name, target_var, min_correlation)

    df3.plot.bar(
        figsize=(20, 10), title="Correlation with {}".format(var_name), fontsize=15, grid=True)
    plt.title(title, fontsize=20)
    plt.xlabel(var_name, fontsize=20)
    plt.ylabel("Correlation with {}".format(target_var), fontsize=20)

    if not os.path.exists('plots'):
        os.makedirs('plots')

    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y_%H:%M:%S.%f)")
    plt.savefig("plots/{}_{}.png".format(title, timestampStr))

    plt.show()
    
    

def plotly_multiline_plot(x_vals, y_vals_dict, title, fig_width=800, fig_height=600, x_label='', y_label=''):
    """
    Creates a multiline plot in Plotly.

    Args:
        x_vals: A list or array of x-axis values.
        y_vals_dict: A dictionary containing one or more sets of y-axis values,
            with each key representing a different line on the plot.
        title: A string representing the title of the plot.

    Returns:
        None.
    """
    # Create a new figure
    fig = go.Figure()

    # Add each line to the figure
    for line_name, y_vals in y_vals_dict.items():
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, name=line_name))

    # Set the title of the plot
    fig.update_layout(title=title, 
                      width=fig_width,
                        height=fig_height,
                     showlegend=True)
    
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    

    # Show the plot
    fig.show()
    
    

def plot_markowitz_bullet(x_vals, y_vals, text_vals): 
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers',
            marker=dict(color='blue', size=10),
            text=text_vals,
            name='Data'
        )
    )

    # Add annotations to the plot
    annotations = []
    for i in range(len(x_vals)):
        annotations.append(
            go.layout.Annotation(
                x=x_vals[i],
                y=y_vals[i],
                text=text_vals[i],
                showarrow=False,
                font=dict(size=12, color='black')
            )
        )

    fig.update_layout(
        title='Annotated Scatter Plot Example',
        xaxis_title='X Axis Title',
        yaxis_title='Y Axis Title',
        annotations=annotations
    )

    # Show the plot
    fig.show()
    


    