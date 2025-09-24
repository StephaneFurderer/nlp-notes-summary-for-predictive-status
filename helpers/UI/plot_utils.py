import plotly.graph_objects as go



def plot_single_claim_lifetime(df, selected_claim, x_axis:str,y_axis:list[str]=None):
    data = df[df['clmNum'] == selected_claim]
    data = data.sort_values(by=x_axis)

    if y_axis is None:
        metrics_to_plot = ['reserve_cumsum', 'paid_cumsum', 'incurred_cumsum', 'expense_cumsum']
    fig = go.Figure()
    for metric in y_axis:
        fig.add_trace(go.Scatter(x=df[x_axis], y=df[metric], mode='lines+markers', name=metric))
    fig.update_layout(title=f"Claim {selected_claim} Development Pattern", xaxis_title=x_axis, yaxis_title=y_axis)
    return fig