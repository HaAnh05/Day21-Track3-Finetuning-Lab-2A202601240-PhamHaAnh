# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**
Điều làm em ngạc nhiên nhất là sức mạnh vượt trội của Baseline (b) — chỉ bằng một System Prompt được viết cấu trúc cẩn thận, Base model chưa hề qua huấn luyện đã đạt tới 76.5% target accuracy và 100% format JSON, vượt xa hoàn toàn việc fine-tune nếu fine-tune không được kiểm soát chặt chẽ về số học.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**
Em mất nhiều thời gian nhất ở việc xử lý lỗi tương thích bên trong thư viện `trl` (`_patch_chunked_ce_lm_head` với `functools.partial`) và thời gian chạy 3 run đối chứng ở NB4 (~44 phút trên Kaggle GPU). Ban đầu em tưởng sẽ mất nhiều thời gian nhất ở phần tokenizer hay tải dữ liệu, nhưng thực tế phần lớn thời gian rơi vào vòng lặp huấn luyện đối chứng và quản lý tương thích phần cứng.

**3. Trước lab này bạn tin điều gì về fine-tune mà giờ bạn không còn tin?**
Trước lab này, em từng tin rằng "đã fine-tune là chắc chắn sẽ thông minh hơn và thắng base model" và "rank LoRA càng cao thì model càng giỏi". Giờ em hiểu rằng rank cao trên một không gian hẹp ($q, v$) thua xa việc dàn trải rank thấp trên toàn bộ các lớp tuyến tính, và fine-tune hoàn toàn có thể làm hỏng model (catastrophic forgetting) nếu không có tập replay.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**
Em dùng AI assistant để giải thích cấu trúc mã nguồn `labkit`, phân tích nguyên nhân gây ra lỗi `AttributeError` khi patch chunked CE trong `trl`, và hỗ trợ rà soát các con số báo cáo theo rubric. Chỗ AI lúc đầu gặp bẫy là gợi ý monkey-patch trong kernel notebook, vốn không có tác dụng khi chạy lệnh qua subprocess `!python ...` trên Kaggle.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**
Bước đầu tiên em làm không phải là mở GPU lên train ngay, mà là: đóng băng một bộ dữ liệu đánh giá kiểm thử độc lập (eval set), xây dựng một Baseline System Prompt tối ưu nhất có thể, và đo đạc benchmark ban đầu. Nếu Baseline Prompt đã đạt yêu cầu nghiệp vụ với chi phí rẻ và độ trễ thấp, em sẽ tư vấn khách hàng chưa cần vội fine-tune.
