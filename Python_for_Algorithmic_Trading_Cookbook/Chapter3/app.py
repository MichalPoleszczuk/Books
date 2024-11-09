import datetime
import numpy as np
import pandas as pd
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.io as pio
from sklearn.decomposition import PCA
from openbb import obb

obb.user.preferences.output_type = "dataframe"

# setting the default styling, chart templates, and initialize the app
pio.templates.default = "plotly"
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# constructing the components of the user interface starting with the text field to enter the list of ticker symbols, the dropdown, the date picker and submit button to somehow run our app
ticker_field = [
    html.Label("Enter Ticker Symbols: "),
    dcc.Input(
        id="ticker-input",
        type="text",
        placeholder="Enter tickers separated by commas (e.g. AAPL or MSFT)",
        style={"width": "50%"}
    ),
]

components_field = [
    html.Label("Select Number of Components:"),
    dcc.Dropdown(
        id="component-dropdown",
        options=[{"label": i, "value": i} for i in range(1, 6)],
        value=3,
        style={"width": "50%"}
    ),
]

date_picker_field = [
    html.Label("Select Date Range:"),  # Label for date picker
    dcc.DatePickerRange(
        id="date-picker",
        start_date=datetime.datetime.now() - datetime.timedelta(365 * 3),
        end_date=datetime.datetime.now(),  # Default to today's date
        display_format="YYYY-MM-DD",
    ),
]
submit = [
    html.Button("Submit", id="submit-button"),
]

# Combine the form elements and placeholders for visualizations to form the app layout
app.layout = dbc.Container(
    [
        html.H1("PCA on Stock Returns"),
        dbc.Row([dbc.Col(ticker_field)]),
        dbc.Row([dbc.Col(components_field)]),
        dbc.Row([dbc.Col(date_picker_field)]),
        dbc.Row([dbc.Col(submit)]),
        dbc.Row(
            [
                dbc.Col([dcc.Graph(id="bar-chart")], width=4),
                dbc.Col([dcc.Graph(id="line-chart")], width=4),
                dbc.Col([dcc.Graph(id="scatter-plot")], width=4),
            ]
        ),
    ]
)

# Now we need to implement a func that will update the charts upon user input
@app.callback(
    [
        Output("bar-chart", "figure"),
        Output("line-chart", "figure"),
        Output("scatter-plot", "figure"),
    ],
    [Input("submit-button", "n_clicks")],
    [
        dash.dependencies.State("ticker-input", "value"),
        dash.dependencies.State("component-dropdown", "value"),
        dash.dependencies.State("date-picker", "start_date"),
        dash.dependencies.State("date-picker", "end_date"),
    ],
)
def update_graphs(n_clicks, tickers, n_components, start_date, end_date):
    if not tickers:
        return {}, {}, {}

    # Parsing input from the users
    tickers = tickers.split(",")
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%S.%f").date()
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%S.%f").date()
    
    # Download the stock data
    data = obb.equity.price.historical(
        tickers,
        start_date=start_date,
        end_date=end_date,
        provider="yfinance"
    ).pivot(columns="symbol", values="close")
    daily_returns = data.pct_change().dropna()

    # Fit the principal component model
    pca = PCA(n_components=n_components)
    pca.fit(daily_returns)
    explained_var_ratio = pca.explained_variance_ratio_

    # Generate the bar chart for individual explained variance
    bar_chart = go.Figure(
        data=[
            go.Bar(
                x=["PC" + str(i + 1) for i in range(n_components)],
                y=explained_var_ratio,
            )
        ],
        layout=go.Layout(
            title="Explained Variance by Component",
            xaxis=dict(title="Principal Component"),
            yaxis=dict(title="Explained Variance"),
        ),
    )

    # Generate the line chart for cumulative explained variance
    cumulative_var_ratio = np.cumsum(explained_var_ratio)
    line_chart = go.Figure(
        data=[
            go.Scatter(
                x=["PC" + str(i + 1) for i in range(n_components)],
                y=cumulative_var_ratio,
                mode="lines+markers",
            )
        ],
        layout=go.Layout(
            title="Cumulative Explained Variance",
            xaxis=dict(title="Principal Component"),
            yaxis=dict(title="Cumulative Explained Variance"),
        ),
    )

    # Compute factor exposures
    X = np.asarray(daily_returns)
    factor_returns = pd.DataFrame(
        columns=["f" + str(i + 1) for i in range(n_components)],
        index=daily_returns.index,
        data=X.dot(pca.components_.T),
    )
    factor_exposures = pd.DataFrame(
        index=["f" + str(i + 1) for i in range(n_components)],
        columns=daily_returns.columns,
        data=pca.components_,
    ).T
    labels = factor_exposures.index
    data = factor_exposures.values

    # Generate the chart for factor exposures
    scatter_plot = go.Figure(
        data=[
            go.Scatter(
                x=factor_exposures["f1"],
                y=factor_exposures["f2"],
                mode="markers+text",
                text=labels,
                textposition="top center",
            )
        ],
        layout=go.Layout(
            title="Scatter Plot of First Two Factors",
            xaxis=dict(title="Factor 1"),
            yaxis=dict(title="Factor 2"),
        ),
    )

    return bar_chart, line_chart, scatter_plot



