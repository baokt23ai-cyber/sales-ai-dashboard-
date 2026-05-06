import pandas as pd

def load_data(file):
    """
    Hàm đọc dữ liệu từ file CSV, hỗ trợ tự động nhận diện nhiều loại bảng mã (encoding)
    và reset con trỏ file (seek(0)) để tránh lỗi EmptyDataError trên Streamlit.
    """
    try:
        file.seek(0) # Đưa con trỏ về dòng đầu tiên của file
        df = pd.read_csv(file, encoding='utf-8')
    except (UnicodeDecodeError, pd.errors.EmptyDataError):
        try:
            file.seek(0) # Tua lại từ đầu trước khi thử bảng mã khác
            df = pd.read_csv(file, encoding='utf-8-sig')
        except (UnicodeDecodeError, pd.errors.EmptyDataError):
            try:
                file.seek(0)
                df = pd.read_csv(file, encoding='cp1252')
            except (UnicodeDecodeError, pd.errors.EmptyDataError):
                # Phương án dự phòng cuối cùng
                file.seek(0)
                df = pd.read_csv(file, encoding='latin1')
                
    return df