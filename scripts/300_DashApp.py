# Make sure dash is installed: pip install dash

# Import required libraries
import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output

# Initialize the Dash app
app = dash.Dash(__name__)

# Load and prepare data
try:
    df_valid_pred = pd.read_csv('zorgdata/df_valid_pred.csv', index_col=False)
    df_valid_pred['datum'] = pd.to_datetime(df_valid_pred['datum'], format='%Y-%m-%d')
except Exception as e:
    print("Error loading data:", e)
    # Handle the error appropriately (e.g., load default data or exit)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Onrustscore"),
    dcc.Dropdown(
        id='client-selector',
        options=[{'label': i, 'value': i} for i in df_valid_pred['ct_id'].unique()],
        value='kamer01'
    ),
    dcc.Graph(id='prediction-plot')
])

# Define callback to update graph based on selected client
@app.callback(
    Output('prediction-plot', 'figure'),
    [Input('client-selector', 'value')]
)
def update_graph(selected_client):
    df_client = df_valid_pred[df_valid_pred['ct_id'] == selected_client].copy()
    if df_client.empty:
        return px.line(title=f"No data for Client {selected_client}")

    # Create the plot using Plotly
    fig = px.line(df_client, x='datum', y='pred_bert_prob', 
                  title=f'Onrustscore van cliënt {selected_client}')
    fig.update_xaxes(tickangle=45)

    return fig

# Run the app
if __name__ == '__main__':
    try:
        app.run_server(debug=True)
    except Exception as e:
        print("Error running the server:", e)
