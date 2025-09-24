import plotly.graph_objects as go



def plot_single_claim_lifetime(df, selected_claim, x_axis: str, y_axis: list[str] = None):
    data = df[df['clmNum'] == selected_claim]
    data = data.sort_values(by=x_axis)

    if y_axis is None:
        y_axis = ['reserve_cumsum', 'paid_cumsum', 'incurred_cumsum', 'expense_cumsum']

    fig = go.Figure()
    for metric in y_axis:
        fig.add_trace(go.Scatter(x=data[x_axis], y=data[metric], mode='lines+markers', name=metric))
    fig.update_layout(
        title_text=f"Claim {selected_claim} Development Pattern",
        xaxis_title=x_axis,
        yaxis_title="amount in $"
    )
    return fig