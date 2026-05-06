import pandas as pd

def revenue_by_date(df):
    return df.groupby('OrderDate')['Revenue'].sum().reset_index()

def revenue_by_month(df):
    df_m = df.copy()
    df_m['Month'] = df_m['OrderDate'].dt.to_period('M').astype(str)
    return df_m.groupby('Month')['Revenue'].sum().reset_index()

def top_products(df, n=10):
    return df.groupby('Product')['Revenue'].sum().reset_index().sort_values('Revenue', ascending=False).head(n)

def revenue_by_region(df):
    return df.groupby('Region')['Revenue'].sum().reset_index().sort_values('Revenue', ascending=False)

def top_customers(df, n=10):
    return df.groupby('CustomerID')['Revenue'].sum().reset_index().sort_values('Revenue', ascending=False).head(n)

def average_revenue_per_day(df):
    daily = revenue_by_date(df)
    return daily['Revenue'].mean() if not daily.empty else 0