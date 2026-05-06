import plotly.express as px
import plotly.graph_objects as go

def plot_revenue_trend(df):
    daily = df.groupby('OrderDate')['Revenue'].sum().reset_index()
    return px.line(daily, x='OrderDate', y='Revenue', title="Xu hướng doanh thu", markers=True, template="plotly_white")

def plot_top_products(df):
    df = df.sort_values('Revenue', ascending=True)
    fig = px.bar(df, y='Product', x='Revenue', orientation='h', color='Revenue', 
                 color_continuous_scale='Sunsetdark', text_auto='.2s', template="plotly_white")
    fig.update_layout(margin=dict(l=200), showlegend=False)
    return fig

def plot_forecast(actual_df, forecast_df):
    """Hàm vẽ biểu đồ dự báo AI"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_df['OrderDate'], y=actual_df['Revenue'], name='Thực tế', line=dict(color='#0ea5e9')))
    fig.add_trace(go.Scatter(x=forecast_df['OrderDate'], y=forecast_df['PredictedRevenue'], 
                             name='Dự báo AI', line=dict(color='#ef4444', dash='dash')))
    fig.update_layout(title="Dự báo xu hướng bằng Linear Regression", template="plotly_white")
    return fig

def plot_cluster_distribution(df):
    return px.pie(df, names='ClusterName', values='TotalRevenue', hole=0.4, title="Tỷ trọng nhóm khách hàng")

def plot_top_customers(df):
    return px.bar(df, x='CustomerID', y='Revenue', color='Revenue', title="Chi tiêu theo khách hàng")