import numpy as np
from sklearn.linear_model import LinearRegression

def forecast_revenue(df):
    revenue_df = df.groupby(df["OrderDate"].dt.date)["Revenue"].sum().reset_index()
    revenue_df.columns = ["OrderDate", "Revenue"]

    revenue_df["DayIndex"] = np.arange(len(revenue_df))

    X = revenue_df[["DayIndex"]]
    y = revenue_df["Revenue"]

    model = LinearRegression()
    model.fit(X, y)

    revenue_df["PredictedRevenue"] = model.predict(X)
    revenue_df["PredictedRevenue"] = revenue_df["PredictedRevenue"].round(0).astype(int)

    return revenue_df, model