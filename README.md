# Project 5 Machine Learning · Nhóm 13

Bản tích hợp Q1–Q6. Người 1 phụ trách Q1/Q2/Q6, Người 2 phụ trách Q3/Q5, Người 3 phụ trách Q4. Xem PDF và nội dung LaTeX trong `bao_cao/`. Số liệu trong báo cáo lấy từ các lần chạy có lưu kết quả.

## Các file chính

| File | Chức năng |
|---|---|
| `models.py` | Định nghĩa mô hình Q1–Q6. |
| `train.py` | Huấn luyện các mô hình Q1–Q5. |
| `losses.py` | Hàm tính sai số cho Q3–Q5. |
| `chay_kiem_thu.py` | Chạy bộ chấm, lưu kết quả và tạo biểu đồ khi chọn tùy chọn tương ứng. |
| `demo_q1_q2_q6.py` | Minh họa Perceptron, hồi quy hàm sin và Attention. |
| `demo_nhan_dien_ngon_ngu.py` | Nhập từ và xem ngôn ngữ Q4 dự đoán. |
| `cong_cu/chuan_bi_du_lieu.py` | Chuẩn bị dữ liệu MNIST và nhận diện ngôn ngữ. |
| `cong_cu/kiem_thu_bo_sung.py` | Kiểm tra phép tính, gradient và tính nhân quả của Attention. |
| `cong_cu/phan_tich_ket_qua.py` | Đánh giá Q3–Q5, lưu ảnh dự đoán và ma trận nhầm lẫn. |
| `cong_cu/ve_bieu_do.py` | Vẽ đường học và đồ thị so sánh dự đoán. |

Các tên `models.py`, `train.py`, `losses.py`, `autograder.py` và `backend.py` được giữ theo mã khung để tương thích với bộ chấm Berkeley. `README.md` là hướng dẫn chạy, `requirements.txt` liệt kê thư viện cần cài. Khi học, đọc `models.py` trước rồi đến `train.py` và `losses.py`.

## Chạy trên Windows

Mở thư mục này trong VS Code, rồi mở Terminal. Đã kiểm tra trên Python 3.12, PyTorch 2.15.0+cpu, NumPy 1.26.4, Matplotlib 3.11.2, CPU 2 luồng. Nếu tạo môi trường mới, chạy:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe cong_cu/chuan_bi_du_lieu.py
.\.venv\Scripts\python.exe -u chay_kiem_thu.py all --seed 42
```

Nếu không có mạng, thêm `--zip "đường-dẫn-đến-ZIP-mã-khung-hoặc-ZIP-Q4"` vào lệnh chuẩn bị dữ liệu. Q1, Q2 và Q6 chạy được khi chưa tải dữ liệu.

## Demo và hình thực nghiệm

Trong các lệnh dưới đây, dùng Python của môi trường đã cài. `--do-thi` lưu và mở biểu đồ trên Windows. `--phan-tich` lưu kết quả phân tích vào `ket_qua/`.

```powershell
python chay_kiem_thu.py q2 --seed 42 --do-thi
python demo_q1_q2_q6.py
python cong_cu/kiem_thu_bo_sung.py
python chay_kiem_thu.py q3 --seed 42 --phan-tich
python chay_kiem_thu.py q4 --seed 42 --phan-tich
python demo_nhan_dien_ngon_ngu.py kiitos paleis cześć hello
python chay_kiem_thu.py q5 --seed 42 --phan-tich
```

- **Người 1** xem `demo_q1_q2_q6.json` để đọc số lượt hội tụ Q1, MSE và so sánh tốc độ học Q2, cùng đầu ra Q6 khi đổi token cuối.
- **Q3/Q5** có ảnh đoán đúng và sai, đường học và ma trận nhầm lẫn. Q5 báo riêng kết quả từ 50 mẫu Berkeley và từ 10.000 ảnh test MNIST.
- **Q4** có đường học và ma trận nhầm lẫn trên test. Lệnh phân tích Q4 tạo file trọng số cục bộ để chạy demo. Demo nhận một từ thuộc bảng 47 ký tự, trả nhãn dự đoán và điểm softmax.
- **Kết quả đối chiếu** nằm trong `thi_nghiem_nguoi_2/` và `thi_nghiem_nguoi_3/`. Đây là số liệu bàn giao từ môi trường của từng người. Các file `q*_seed42.json` ở ngay trong `ket_qua/` là kết quả phân tích của bản chung.

## Cách đọc kết quả

- Q1 hội tụ trên dữ liệu sinh phân tách tuyến tính. Q2 đo lỗi huấn luyện và nội suy trong khoảng đã học. Q6 kiểm tra đầu ra số của Attention.
- Q3 dùng 60.000 ảnh train, 5.000 dev, 5.000 test. Bản chung phục hồi trọng số có dev tốt nhất.
- Q4 dùng RNN 192 nút ẩn, chọn trọng số theo dev. Điểm test của thí nghiệm bàn giao và điểm của bản chung đến từ các lần chạy riêng.
- Q5 học 200 ảnh. Backend Berkeley lấy dev/test từ chính 200 ảnh này, nên điểm benchmark không ước lượng được kết quả trên ảnh mới. Kiểm tra thêm dùng tập `test_images` chính thức của MNIST, tách khỏi dữ liệu Q5 dùng để học và chọn trọng số.
- Điểm trong `tong_hop_kiem_thu.json` là điểm bộ kiểm thử Berkeley.

ZIP bàn giao gồm mã nguồn, báo cáo và kết quả. Dữ liệu, trọng số cục bộ, môi trường Python và cache được loại khi đóng gói. Khi nộp, dùng mẫu LaTeX của lớp và thêm tên thành viên cùng liên kết GitHub thật vào phần Tóm tắt. Nguồn mã khung là CS188 Fall 2025 Project 5.
