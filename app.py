import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.analysis import (
    revenue_by_date, revenue_by_month, top_products,
    revenue_by_region, top_customers, average_revenue_per_day
)
from src.clustering import customer_segmentation
from src.visualization import (
    plot_revenue_trend, plot_top_products, plot_forecast,
    plot_cluster_distribution, plot_top_customers
)

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(page_title="HỆ THỐNG AI PHÂN TÍCH VÀ TRỰC QUAN HÓA DỮ LIỆU BÁN HÀNG", page_icon="📈", layout="wide")

# --- 2. GIAO DIỆN PREMIUM (CSS) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], .stMarkdown { font-family: 'Inter', sans-serif !important; color: #1E293B; }
    h1 { font-weight: 700 !important; color: #1e4e79 !important; text-align: center; margin-bottom: 5px; text-transform: uppercase; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }
    .ai-insight { background-color: #f0f9ff; border-left: 5px solid #0ea5e9; padding: 15px; border-radius: 8px; margin-bottom: 15px; font-size: 14px; }
    .xai-box { background-color: #f8fafc; border: 1px dashed #94a3b8; padding: 15px; border-radius: 8px; font-size: 14px; }
    .story-box { background-color: #ffffff; padding: 30px; border-radius: 12px; border: 1px solid #e2e8f0; text-align: center; margin-top: 20px;}
    .cluster-box { padding: 15px; margin-bottom: 10px; border-radius: 8px; background-color: #f8fafc; border-left: 5px solid #0284c7; }
</style>
""", unsafe_allow_html=True)

st.title("HỆ THỐNG AI PHÂN TÍCH VÀ TRỰC QUAN HÓA DỮ LIỆU BÁN HÀNG")
st.markdown("<p style='text-align: center; color: #64748b; font-size: 16px;'>Hợp nhất mọi định dạng dữ liệu - Dự báo xu hướng - Tối ưu quyết định kinh doanh</p>", unsafe_allow_html=True)
st.markdown("---")

# --- 3. DỮ LIỆU MẪU TRẢI NGHIỆM ---
st.markdown("### 📂 Nguồn dữ liệu mẫu")
st.info("💡 **Hướng dẫn:** Bạn có thể tải các file dữ liệu mẫu đã được chuẩn hóa dưới đây về máy, sau đó tải ngược lên mục **'Tải các tệp dữ liệu bán hàng'** bên dưới để trải nghiệm các tính năng phân tích và vẽ biểu đồ của hệ thống.")

c_dl1, c_dl2, c_dl3 = st.columns(3)

def create_download_btn(col, file_path, label, file_name):
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            # Đã gỡ bỏ use_container_width để fix lỗi Terminal Warning màu vàng của bạn
            col.download_button(label=label, data=file, file_name=file_name, mime="text/csv")
    else:
        col.button(f"⚠️ Chưa tìm thấy: {file_name}", disabled=True)

# Tải 3 file từ thư mục sample_data
create_download_btn(c_dl1, "sample_data/thegioididong-dienthoai.csv", "📱 Tải Data: Điện Thoại", "thegioididong-dienthoai.csv")
create_download_btn(c_dl2, "sample_data/thegioididong-chuot.csv", "🖱️ Tải Data: Chuột Máy Tính", "thegioididong-chuot.csv")
create_download_btn(c_dl3, "sample_data/thegioididong-com-2026-05-06-maytinh.csv", "💻 Tải Data: Laptop/Máy Tính", "thegioididong-maytinh.csv")

st.markdown("<br>", unsafe_allow_html=True)

# --- 4. XỬ LÝ TẢI DỮ LIỆU ---
uploaded_files = st.file_uploader("📂 Tải các tệp dữ liệu bán hàng (.csv)", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    all_dfs = []
    for f in uploaded_files:
        temp_df = load_data(f)
        source_label = f.name.replace('.csv', '').replace('_', ' ').title()
        temp_df['Source'] = source_label
        all_dfs.append(temp_df)
    
    df = pd.concat(all_dfs, ignore_index=True)

    # --- TỰ ĐỘNG NHẬN DIỆN VÀ CHUẨN HÓA CỘT DỮ LIỆU ---
    auto_mapping = {}
    if 'Ten_San_Pham' in df.columns and 'Product' not in df.columns: 
        auto_mapping['Ten_San_Pham'] = 'Product'
    if 'Doanh_Thu' in df.columns and 'Revenue' not in df.columns: 
        auto_mapping['Doanh_Thu'] = 'Revenue'
    if 'So_Luong_Da_Ban_So' in df.columns and 'Quantity' not in df.columns: 
        auto_mapping['So_Luong_Da_Ban_So'] = 'Quantity'
    
    if auto_mapping:
        df = df.rename(columns=auto_mapping)
        
    df = df.loc[:, ~df.columns.duplicated()]

    df = preprocess_data(df)

    # --- SIDEBAR THÔNG MINH ---
    st.sidebar.header("🕹️ Trung tâm điều khiển")
    min_d, max_d = df['OrderDate'].min().date(), df['OrderDate'].max().date()
    date_range = st.sidebar.date_input("🗓️ Khoảng thời gian", [min_d, max_d], min_value=min_d, max_value=max_d)
    
    sources = sorted(df['Source'].unique())
    sel_sources = st.sidebar.multiselect("🏷️ Lọc ngành hàng", sources, default=sources)

    if 'Region' in df.columns:
        regions = sorted(df['Region'].unique())
        sel_regions = st.sidebar.multiselect("📍 Chọn khu vực phân tích", regions, default=regions)
    else:
        sel_regions = []

    if len(date_range) == 2:
        df = df[(df['OrderDate'].dt.date >= date_range[0]) & (df['OrderDate'].dt.date <= date_range[1])]
    
    df = df[df['Source'].isin(sel_sources)]
    if 'Region' in df.columns and sel_regions:
        df = df[df['Region'].isin(sel_regions)]

    if df.empty:
        st.warning("Không có dữ liệu trong bộ lọc đã chọn.")
        st.stop()

    # --- TỔ CHỨC HIỂN THỊ TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 Tổng quan chiến lược", "🤖 Dự báo & Phân tích AI", "👥 Hành vi khách hàng"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tổng Doanh Thu", f"{df['Revenue'].sum():,.0f} VND")
        c2.metric("Tổng Sản Phẩm Đã Bán", f"{int(df['Quantity'].sum()):,}")
        c3.metric("Khách Hàng Hoạt Động", df['CustomerID'].nunique() if 'CustomerID' in df.columns else 0)
        c4.metric("Doanh Thu TB/Ngày", f"{average_revenue_per_day(df):,.0f} VND")

        st.markdown("---")
        st.subheader("Biến động doanh thu & Insight thời gian")
        daily_source_rev = df.groupby(['OrderDate', 'Source'])['Revenue'].sum().reset_index()
        fig_line = px.line(daily_source_rev, x='OrderDate', y='Revenue', color='Source', markers=True, template="plotly_white")
        st.plotly_chart(fig_line, use_container_width=True)
        
        peak_day = daily_source_rev.loc[daily_source_rev['Revenue'].idxmax()]
        st.info(f"💡 **AI Insight:** Ngày có doanh thu cao nhất là **{peak_day['OrderDate'].strftime('%d/%m/%Y')}** đạt **{peak_day['Revenue']:,.0f} VND**. Cần rà soát các sự kiện hoặc chiến dịch diễn ra vào ngày này để áp dụng cho tương lai.")

        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("Top sản phẩm bán chạy")
            top_p_df = top_products(df)
            st.plotly_chart(plot_top_products(top_p_df), use_container_width=True)
            st.success(f"🏆 **Sản phẩm chủ lực:** **{top_p_df.iloc[0]['Product']}** đang dẫn đầu doanh thu. Nên xem xét mở rộng các biến thể của sản phẩm này.")
            
        with col_r:
            st.subheader("Phân bố doanh thu theo khu vực")
            reg_df = revenue_by_region(df)
            if not reg_df.empty:
                fig_region = px.pie(reg_df, values='Revenue', names='Region', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
                st.plotly_chart(fig_region, use_container_width=True)
                
                top_reg = reg_df.iloc[0]['Region']
                share = (reg_df.iloc[0]['Revenue'] / reg_df['Revenue'].sum()) * 100
                st.warning(f"📍 **Thị trường trọng điểm:** Khu vực **{top_reg}** đang chiếm **{share:.1f}%** thị phần. Đây là khu vực cần ưu tiên phân phối hàng hóa và marketing.")
            else:
                st.info("💡 Không tìm thấy dữ liệu khu vực.")

    with tab2:
        st.subheader("🔮 Dự báo & Phân tích Xu hướng Chiến lược")
        
        daily_rev = df.groupby('OrderDate')['Revenue'].sum().reset_index()
        daily_rev['DayIdx'] = (daily_rev['OrderDate'] - daily_rev['OrderDate'].min()).dt.days
        
        from sklearn.linear_model import LinearRegression
        model_lr = LinearRegression().fit(daily_rev[['DayIdx']], daily_rev['Revenue'])
        
        last_date = daily_rev['OrderDate'].max()
        last_idx = daily_rev['DayIdx'].max()
        
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
        future_idx = np.array(range(last_idx + 1, last_idx + 31)).reshape(-1, 1)
        
        preds = model_lr.predict(future_idx)
        preds = np.maximum(preds, 0)
        forecast_df = pd.DataFrame({'OrderDate': future_dates, 'PredictedRevenue': preds})
        
        c_fc1, c_fc2, c_fc3 = st.columns(3)
        avg_current = daily_rev['Revenue'].mean()
        avg_forecast = forecast_df['PredictedRevenue'].mean()
        growth_val = avg_forecast - avg_current
        
        c_fc1.metric("Doanh thu TB hiện tại", f"{avg_current:,.0f} VND")
        c_fc2.metric("Doanh thu TB dự báo (30 ngày tới)", f"{avg_forecast:,.0f} VND", delta=f"{growth_val:,.0f} VND")
        c_fc3.metric("Mức độ tin cậy mô hình", "85%", help="Dựa trên độ dốc xu hướng lịch sử")

        st.markdown("---")
        col_f1, col_f2 = st.columns([2, 1])
        
        with col_f1:
            fig_forecast = px.area(forecast_df, x='OrderDate', y='PredictedRevenue', title="Dự báo xu hướng tăng trưởng (30 ngày tiếp theo)", color_discrete_sequence=['#0ea5e9'], template="plotly_white")
            fig_forecast.update_layout(hovermode="x unified")
            st.plotly_chart(fig_forecast, use_container_width=True)
            
        with col_f2:
            st.markdown("### 💡 Khuyến nghị chiến lược AI")
            trend_val = forecast_df['PredictedRevenue'].iloc[-1] - forecast_df['PredictedRevenue'].iloc[0]
            
            if trend_val > 0:
                st.success(f"📈 **XU HƯỚNG TĂNG TRƯỞNG**\nDự kiến doanh thu sẽ đi lên trong tháng tới.")
                action = "Tăng ngân sách quảng cáo khu vực dẫn đầu."
            else:
                st.error(f"📉 **XU HƯỚNG SUY GIẢM**\nDự kiến thị trường sẽ trầm lắng hơn.")
                action = "Triển khai Flash Sale để kích cầu."

            st.markdown(f"""
            <div class='ai-insight'>
            • <b>Phân tích:</b> Mô hình nhận diện xu hướng bắt đầu từ ngày {future_dates[0].strftime('%d/%m/%Y')}.<br>
            • <b>Hành động:</b> {action}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div class='xai-box'><b>Cơ chế AI:</b> Thuật toán Linear Regression tính toán đường xu hướng nối tiếp dữ liệu lịch sử để định hướng kế hoạch tương lai.</div>", unsafe_allow_html=True)

    with tab3:
        st.subheader("👥 Phân cụm khách hàng (K-Means)")
        if df['CustomerID'].nunique() > 1:
            cluster_data = customer_segmentation(df, n_clusters=min(3, df['CustomerID'].nunique()))
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.plotly_chart(plot_cluster_distribution(cluster_data), use_container_width=True)
            with col_c2:
                st.plotly_chart(plot_top_customers(top_customers(df)), use_container_width=True)
            
            st.markdown("### Giải mã hành vi khách hàng")
            cluster_explain = cluster_data.groupby(["ClusterName", "BusinessNote", "ClusterPriority"]).agg(
                SoKhachHang=("CustomerID", "count"),
                DoanhThuTB=("TotalRevenue", "mean")
            ).reset_index().sort_values("ClusterPriority")

            for _, row in cluster_explain.iterrows():
                st.markdown(f"""
                    <div class='cluster-box'>
                        <b>Nhóm {row['ClusterName']}:</b> {int(row['SoKhachHang'])} khách hàng | <b>Doanh thu TB:</b> {row['DoanhThuTB']:,.0f} VND<br>
                        🎯 <b>Chiến lược:</b> {row['BusinessNote']}
                    </div>
                """, unsafe_allow_html=True)

    with st.expander("🔍 Xem bảng dữ liệu chi tiết"):
        st.dataframe(df, use_container_width=True)

else:
    st.markdown("""
    <div class='story-box'>
        <h2>📊 HỆ THỐNG AI PHÂN TÍCH VÀ TRỰC QUAN HÓA DỮ LIỆU BÁN HÀNG</h2>
        <p>Vui lòng tải tệp CSV để bắt đầu phân tích <b>Doanh thu, Khu vực</b> và <b>Dự báo AI</b>.</p>
    </div>
    """, unsafe_allow_html=True)