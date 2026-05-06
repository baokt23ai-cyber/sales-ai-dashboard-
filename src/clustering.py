import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def customer_segmentation(df, n_clusters=3):
    cust_df = df.groupby("CustomerID").agg({"Revenue": "sum", "OrderDate": "count"}).rename(columns={"OrderDate": "Frequency", "Revenue": "TotalRevenue"})
    scaler = StandardScaler()
    scaled = scaler.fit_transform(cust_df[["TotalRevenue", "Frequency"]])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cust_df["Cluster"] = kmeans.fit_predict(scaled)
    
    def label(row):
        if row["TotalRevenue"] > cust_df["TotalRevenue"].quantile(0.7):
            return "Khách hàng VIP", "Ưu đãi đặc quyền", 1
        return "Khách hàng Tiềm năng", "Gửi mã khuyến mãi", 2

    cust_df[["ClusterName", "BusinessNote", "ClusterPriority"]] = cust_df.apply(lambda x: pd.Series(label(x)), axis=1)
    return cust_df.reset_index()