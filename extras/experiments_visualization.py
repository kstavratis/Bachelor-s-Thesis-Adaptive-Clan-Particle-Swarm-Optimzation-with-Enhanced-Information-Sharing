# Needed imports
import os

import pandas as pd

import webbrowser

from dash import Dash, html, dcc, callback, Output, Input, ctx

import tkinter as tk
from tkinter import filedialog

import plotly.graph_objs as go

# Empty the content of the global variables when the application restarts.
data_dfs = []
content_of_html_table = []

def plot_figure(minx: int, maxx: int, lb: float=0, ub : float=10**10):

    fig = go.Figure()

    for df in data_dfs:
        filtered_df = df[(df['0'] >= lb) & (df['0'] <= ub)].loc[minx:maxx]

        fig.add_trace(
            go.Scatter(x=filtered_df['iteration'],
                       y=filtered_df['0'],
                       mode='lines'
            )
        )

    return fig


def generate_csv_files_table(csv_filenames: list):
    return html.Table([

        html.Thead(html.Th('CSV files to be processed')),

        html.Div([
        html.Tbody([
                html.Tr([ html.Td(filename) ])
            for filename in csv_filenames]
        )
        ], style={'height' : '100px', 'overflow' : 'scroll'} )
    ])

@callback(
    Output('csv-filenames-table', 'children'),
    Input('keyword-input', 'value'),
    Input('rootdir-lbl', 'children'),
    prevent_initial_call=True
)
def generate_html_table_of_csv_files(keyword, root_directory):
    csv_files = []
    print(f'root_directory = {root_directory}')
    for dirpath, dirnames, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename.endswith(".csv") and keyword in filename:
                full_filename = os.path.join(dirpath, filename)
                csv_files.append(full_filename)


    global content_of_html_table; content_of_html_table = csv_files

    return generate_csv_files_table(csv_files)


@callback(
    Output('rootdir-lbl', 'children'),
    Input('select-rootdir-button', 'n_clicks'),
    prevent_initial_call=True
)
def select_csv_files_root_dir(dummy_input):

    root = tk.Tk()
    # root.withdraw() # Do not show that tkinter is being used to user.
    pathname = filedialog.askdirectory()
    root.destroy()
    # If it not destroyed manually,
    # it seems like there are conflicts taking place when a new root is to be drawn.
    # See https://stackoverflow.com/questions/71163507/python-dash-tkinter-openfile-dialog-works-fine-only-the-first-time-but-fails-a

    return pathname

def add_dataset(filenames: list[str]):

    dfs = []

    for filename in filenames:
        # The 0-th column is the 'iteration' id column
        df = pd.read_csv(filename, index_col=[0])
        dfs.append(df)
    
    experiments_df = pd.concat(dfs)
    # Acquire the wanted result! Mean of the value per iteration.
    experiments_df = experiments_df.groupby(level=0).mean()
    experiments_df.reset_index(inplace=True)

    data_dfs.append(experiments_df)


@callback(
    Output('experiments-fig', 'figure'),
    Input('csv-filenames-table', 'children'),
    Input('iterations-range-slider', 'value'),
    Input('obj-val-lb', 'value'),
    Input('obj-val-ub', 'value'),
    prevent_initial_call=True,
)
def update_figure(table, slider_configuration, lb, ub):

    minx, maxx = slider_configuration

    # Identify the source which triggered the callback
    # (i.e. which component was manipulated)
    triggered_id = ctx.triggered_id

    # Update figure because a new dataset has been added.
    if triggered_id == 'csv-filenames-table':
        add_dataset(content_of_html_table)

    # Data is filtered whenever the sliders are manipulated.
    figure = plot_figure(minx, maxx, lb, ub)


    return figure


app = Dash(__name__)


app.layout = html.Center([
    html.H1(children='pymixinswarms Experiment Plotter', style={'textAlign':'center'}),

    # ==================== User input START ====================
    html.Div([
        # Keyword
        html.Div([
            html.Label('keyword input', style={'margin-right': '20px'}),
            dcc.Input(
                placeholder='Names of .csv files to be collected".',
                type='text',
                value='',
                id='keyword-input'
            )
        ], style = {'margin' : '25px'}),
    

        # Button & show path
        html.Div([
            html.Button('Select Directory', id='select-rootdir-button', style={'margin-right': '20px'}),
            html.Label(id='rootdir-lbl')
        ]),
    
        # Resulting table
        html.Table('', id='csv-filenames-table')
    ]),
    # ==================== User input FINISH ====================

    # ==================== Output (graphs) START ====================

    html.Div([
        dcc.Loading(
            id='fig-load',
            children=dcc.Graph(id='experiments-fig'),
            type='default'
        ),

        html.Div([
            'Objective value range ',
            dcc.Input(id='obj-val-lb', type='number', placeholder='Objective value lower bound', value=0),
            ' - ',
            dcc.Input(id='obj-val-ub', type='number', placeholder='Objective value upper bound', value=10**10)
        ]
        ),

        html.Div([
            'Iterations',
            dcc.RangeSlider(id='iterations-range-slider', className='',
                min=0, max=2000,
                step=1,
                marks=None,
                value=[0, 2000],
                allowCross=False,
                tooltip={'placement' : 'bottom', 'always_visible': True}
            ),
        ])
    ])
    # ==================== Output (graphs) FINISH ====================
])

if __name__ == '__main__':
    BASE_URL = 'http://localhost:8050'
    app.server.base_url = BASE_URL
    webbrowser.open(app.server.base_url)
    app.run(debug=False)