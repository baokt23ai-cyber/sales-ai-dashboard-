import pandas as pd
import numpy as np

def preprocess_data(df):
    # 1. Giữ nguyên tên cột thô ban đầu để không làm mất dữ liệu khi hiển thị
    # Chỉ tạo thêm các cột 'chuẩn' dựa trên dữ liệu thô đó
    
    # 2. Xử lý cột 'Source' (Để sửa lỗi KeyError tại app.py)
    if 'Source' not in df.columns:
        df['Source'] = 'Thế Giới Di Động'

    # 3. Xử lý Ngày tháng (Dựa trên thời gian thực tế mỗi lần upload)
    if 'OrderDate' not in df.columns:
        num_rows = len(df)
        # Giả lập ngày giao dịch trong vòng 60 ngày gần đây
        df['OrderDate'] = pd.Timestamp.now().normalize() - pd.to_timedelta(np.random.randint(0, 60, size=num_rows), unit='d')
    else:
        df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')

    # 4. Làm sạch Doanh thu từ cột giá thô (ví dụ: price2)
    # Tự động tìm cột có chứa thông tin giá tiền
    price_cols = [c for c in df.columns if any(k in c.lower() for k in ['giá', 'price', 'price2', 'revenue'])]
    if price_cols:
        # Lấy cột giá đầu tiên tìm được, xóa bỏ các ký tự không phải số
        df['Revenue'] = df[price_cols[0]].astype(str).str.replace(r'[^\d]', '', regex=True)
        df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce').fillna(0)
    else:
        df['Revenue'] = 0
    
    # 5. Gán nhãn sản phẩm từ cột tiêu đề thô (ví dụ: title)
    if 'Product' not in df.columns:
        df['Product'] = df['title'] if 'title' in df.columns else "Sản phẩm"

    # 6. Định danh khách hàng và khu vực (Giả lập để phục vụ AI)
    if 'CustomerID' not in df.columns:
        df['CustomerID'] = [f"KH-{np.random.randint(1000, 9999)}" for _ in range(len(df))]
    if 'Region' not in df.columns:
        df['Region'] = np.random.choice(['Hà Nội', 'TP. Hồ Chí Minh', 'Đà Nẵng', 'Cần Thơ'], size=len(df))
    if 'Quantity' not in df.columns:
        df['Quantity'] = np.random.randint(1, 5, size=len(df))

   # --- ĐỊNH DẠNG HIỂN THỊ THEO DỮ LIỆU MỖI LẦN UPLOAD ---
    # Chuyển đổi cột doanh thu thành định dạng VNĐ có dấu chấm phân cách
    df['Giá bán'] = df['Revenue'].apply(lambda x: "{:,.0f} ₫".format(x).replace(",", "."))
    # Định dạng ngày theo chuẩn Việt Nam
    df['Ngày bán'] = df['OrderDate'].dt.strftime('%d/%m/%Y')

    # Sắp xếp các cột quan trọng lên đầu bảng "Xem chi tiết" để giảng viên dễ nhìn
    display_cols = ['Ngày bán', 'Product', 'Giá bán', 'Region', 'Source']
    
    # SỬA LỖI TẠI ĐÂY: Lấy tất cả các cột còn lại (Bắt buộc giữ lại OrderDate và Revenue để app.py không bị lỗi)
    other_cols = [c for c in df.columns if c not in display_cols]
    
    # Gom các cột hiển thị đẹp lên trước, các cột tính toán/thô ra phía sau
    df = df[display_cols + other_cols]

    # Bây giờ hệ thống sẽ tìm thấy OrderDate để sắp xếp bình thường
    return df.sort_values('OrderDate', ascending=False).reset_index(drop=True)  