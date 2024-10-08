import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Bước 1: Đọc file Excel từ đường dẫn bạn đã cung cấp
file_path = r"B:\Documents\DataScience Research(09 2024)\BTL_KHDL\Scraping-Selenium_Kalodata-Sale_Tiktok\Data\product_creator_after_preprocessing_excel.xlsx"

# Đọc dữ liệu vào DataFrame
df = pd.read_excel(file_path)

# Bước 2: Điều chỉnh cột Rank
df['Rank'] = df['Rank'] + 1  # Tăng tất cả các giá trị trong cột Rank lên 1

# Bước 3: Hiển thị các cột trong file để kiểm tra
print("Các cột có trong file dữ liệu:", df.columns)

# In từng tên cột để đảm bảo không có nhầm lẫn
for col in df.columns:
    print(f"'{col}'")

# Bước 4: Chọn các thuộc tính cần phân tích và cột Rank
# Sử dụng đúng tên cột như hiển thị từ bước trên
X_columns = [ 'Số người theo dõi(nghìn người)', 'Doanh thu(nghìn đồng)', 
             'Lượt bán(nghìn lượt)', 'Doanh thu từ video(nghìn đồng)', 'Doanh thu Live(nghìn đồng)']

# Kiểm tra xem tất cả các cột có tồn tại trong DataFrame không
missing_columns = [col for col in X_columns if col not in df.columns]
if missing_columns:
    print(f"Các cột không có trong DataFrame: {missing_columns}")
else:
    # Nếu không có cột nào thiếu, tiếp tục quá trình
    X = df[X_columns]
    y = df['Rank']

    # Bước 5: Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Bước 6: Khởi tạo và huấn luyện mô hình hồi quy tuyến tính
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Bước 7: In ra trọng số của các thuộc tính
    print("Trọng số của các thuộc tính:")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"{feature}: {coef:.4f}")

    # Bước 8: Đánh giá mô hình
    score = model.score(X_test, y_test)
    print(f"Độ chính xác của mô hình (R^2 score): {score:.4f}")
