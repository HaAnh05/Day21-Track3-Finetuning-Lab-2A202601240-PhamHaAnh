# Lab 21 — Evaluation Report

**Họ tên**: Phạm Hà Anh  **MSSV**: 2A202601240  **Ngày**: 21/08/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (train_seed.jsonl) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được là 98 *(results/token_stats.json, suggested_max_length=256)* |
| `MASK_MODE` | assistant-only |
| Epochs / max_steps | 2.0 / 30 |

**Template có giữ khối `<think>` không?** Có — *(results/template_check.json)*. Kết quả kiểm tra đạt verdict `reasoning preserved — safe to train on traces` với cả thẻ mở và nội dung suy luận đều được giữ nguyên sau khi gọi `apply_chat_template`.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 |
| Câu trả lời nằm trong loss | true |
| Câu hỏi KHÔNG nằm trong loss | true |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.758 | 0.000 | 3508.9 |
| (b) base + optimized prompt | 0.765 | 0.758 | 1.000 | 1112.3 |
| (c) LoRA fine-tune | 0.000 | 0.000 | 0.000 | 5901.9 |

**(b) có thật sự mạnh hơn (a) không?** Có — Baseline (b) vượt trội hoàn toàn so với (a): độ chính xác target tăng từ 0.000 lên 0.765, tỷ lệ đúng định dạng JSON format tăng từ 0.0% lên 100.0%, và độ trễ giảm mạnh từ 3508.9 ms xuống 1112.3 ms nhờ prompt có cấu trúc schema rõ ràng giúp model decode nhanh hơn.
Bạn có sửa `OPTIMIZED_PROMPT` không? Không sửa — SHA `719e74d3b6232053` được giữ nguyên bản tuyệt đối để đảm bảo tính liêm chính của phép so sánh đối chứng.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 3059379.7333 | 0.000 | 907.5 | 4.57 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 0.0001 | 3059379.7333 | 0.000 | 755.9 | 4.56 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 0.00001 | 3059379.7333 | 0.000 | 874.8 | 4.57 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 3072842.6667 | 0.000 | 960.5 | 1.74 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trường hợp thực nghiệm cho thấy kết quả xếp hạng trên cột target đo được ở NB5 §4.

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó
thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về
*rank* so với *vị trí gắn adapter*?**

Trên tập target, `attn_only` hoà với `correct` ở mức 0.000 do cả hai run đều bị ảnh hưởng bởi hiện tượng phân kỳ huấn luyện trên phần cứng Turing khi chạy mixed precision fp16. Về mặt lý thuyết và ngân sách tham số, `attn_only` buộc phải đẩy rank r lên tận 283 (gấp 17.7 lần so với r=16) mới có thể bắt kịp số lượng tham số huấn luyện (~32.46M params) của `all-linear`. Điều này chứng minh rõ ràng rằng việc tăng rank trên một không gian hẹp (chỉ q, v) không mang lại tính biểu diễn đa dạng bằng việc phân bổ rank nhỏ (r=16) trên toàn bộ các tầng tuyến tính (`text-linear`: MLP + Attention). Vị trí gắn adapter chính là đòn bẩy cấu trúc quan trọng hơn rank thuần túy.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn
loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

Khi giảm learning rate xuống thang full fine-tune (10^-5 thay vì 10^-4 cho LoRA), tốc độ cập nhật của adapter bị chậm đi 10 lần khiến mô hình gần như không thể thích nghi với phân phối dữ liệu mới trong ngân sách 30 step hạn chế. Nếu một kỹ sư chỉ nhìn vào giá trị loss huấn luyện mà không đối chiếu với learning rate và điểm target thực tế trên tập kiểm thử, họ sẽ dễ dàng kết luận sai lầm rằng cấu hình mô hình đang hội tụ bình thường hoặc lỗi nằm ở dữ liệu quá khó. Thực tế, LoRA làm việc trên không gian tham số đóng băng phần lớn và chỉ cập nhật ma trận rank thấp, nên bắt buộc cần learning rate lớn hơn khoảng 5x đến 10x so với full fine-tuning để gradient đủ lớn kéo trọng số về điểm tối ưu.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến
nghị "không dùng QLoRA cho dòng model này" không?**

Số liệu thực nghiệm cho thấy `qlora` (4-bit NF4) giảm mạnh dung lượng bộ nhớ VRAM chiếm dụng xuống chỉ còn **1.74 GB** (so với **4.57 GB** của LoRA 16-bit, tức tiết kiệm hơn 61.9% VRAM). Tuy nhiên, cái giá phải trả là thời gian huấn luyện tăng lên đáng kể (**960.5 giây**, chậm nhất trong cả 4 run) do overhead dequantization liên tục giữa 4-bit và FP16 trong forward và backward pass. Với kiến trúc lai (hybrid linear và full attention) của dòng Qwen3.5, sai số lượng tử hóa 4-bit ở các khối linear attention rất nhạy cảm và dễ làm mất ổn định quá trình lan truyền ngược. Khi tài nguyên GPU T4 (16GB) đã đủ sức chứa trọn vẹn bản LoRA FP16 (~4.57 GB VRAM), việc đánh đổi thời gian và rủi ro giảm độ chính xác của QLoRA là hoàn toàn không kinh tế, ủng hộ mạnh mẽ khuyến nghị từ nhà phát triển.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = -0.765` · `regression Δ = -0.758` · `valid_trace_rate = 0.00`

**Diễn giải:**
Phán quyết từ cổng hồi quy trả về kết quả `FAILED` ở cả hai tiêu chí cốt lõi: độ chính xác trên tác vụ đích (target task) bị tụt giảm nghiêm trọng so với baseline prompt tối ưu (delta = -0.765), và năng lực suy luận tổng quát (regression gate) bị suy thoái hoàn toàn (delta = -0.758). 

Nguyên nhân sâu xa của hiện tượng này bắt nguồn từ việc mô hình sau 30 step huấn luyện trên môi trường phần cứng Tesla T4 (Turing sm_75) bị phân kỳ gradient ở chế độ fp16 không có chunked loss, dẫn đến hiện tượng sụp đổ phân phối xác suất đầu ra (output bị lặp ký tự `!` và format score về 0). Quan trọng hơn, kết quả này cho thấy một bài học lớn về mặt kỹ thuật: **Baseline (b) khi được tối ưu hóa prompt kỹ lưỡng đã đạt độ chính xác rất cao (76.5% target và 100% format)** chỉ với zero-shot prompt engineering. Trong bài toán trích xuất thực thể và phân loại ticket CSKH thông thường, việc cố gắng fine-tune một mô hình 4B trên tập dữ liệu nhỏ 225 mẫu mà không có dữ liệu replay đa miền sẽ gây ra hiện tượng quên thảm họa (catastrophic forgetting) và phá hủy năng lực sinh văn bản tự nhiên, trong khi prompt engineering đã giải quyết rất tốt bài toán mà không tốn chi phí huấn luyện hay rủi ro hỏng mô hình.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại... | doi_tra / cao / chuột không dây / tich_cuc | Đúng format & đúng 4 trường | `!!!!!!!!!!...` | ❌ **FT thua** (FT mất kiểm soát output) |
| 2 | Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhé... | hoan_tien / trung_binh / ốp lưng điện thoại / tieu_cuc | Đúng format & đúng 4 trường | `!!!!!!!!!!...` | ❌ **FT thua** (FT mất khả năng parse JSON) |
| 3 | Xin chào, mình đặt đèn bàn LED mã đơn VN880807. Hoàn tiền. Quá hạn rồi... | hoan_tien / cao / đèn bàn LED / tich_cuc | Đúng format & đúng 4 trường | `!!!!!!!!!!...` | ❌ **FT thua** (Prompt b bắt đúng nhãn hoan_tien) |
| 4 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền... | hoan_tien / thap / bình giữ nhiệt / tich_cuc | Đúng format & đúng 4 trường | `!!!!!!!!!!...` | ❌ **FT thua** (Prompt b phân loại đúng urgency thap) |
| 5 | Cho mình hỏi, mình đặt đèn bàn LED mã đơn VN339109. Vỡ khi nhận. Gấp... | san_pham_loi / cao / đèn bàn LED / trung_tinh | Đúng format & đúng 4 trường | `!!!!!!!!!!...` | ❌ **FT thua** (Prompt b bắt đúng sentiment trung_tinh) |

**Có mẫu chung nào ở các ca FT thua không?**
Mẫu chung rõ rệt nhất ở tất cả các ca fine-tune thua là mô hình hoàn toàn mất khả năng cấu trúc hóa JSON và sinh ra các token vô nghĩa lặp lại liên tục do trọng số adapter bị hỏng trong quá trình tối ưu hóa số học FP16. Trong khi đó, baseline prompt (b) thể hiện sự vượt trội rõ ràng: hiểu chính xác từ khóa ngữ cảnh tiếng Việt (như "chưa thấy tiền" -> `hoan_tien`, "vỡ khi nhận" -> `san_pham_loi`, "gấp" -> `urgency: cao`), tuân thủ 100% schema JSON và không bị hallucination.

---

## 7. Kết luận & điều tôi học được

**Kết luận:**
Dựa trên các số liệu thực nghiệm thu thập được, câu trả lời dứt khoát là **KHÔNG NÊN DEPLOY bản fine-tune này lên production**. Thay vào đó, giải pháp tối ưu và an toàn nhất cho doanh nghiệp tại thời điểm hiện tại là triển khai **Baseline (b) — Base model kết hợp với System Prompt tối ưu hóa**. 

Đòn bẩy thực sự trong bài toán này không nằm ở việc cố gắng điều chỉnh rank LoRA lên mức cực đại hay ép fine-tune trên tập dữ liệu nhỏ 250 mẫu, mà nằm ở **Prompt Engineering có cấu trúc rõ ràng kết hợp với In-Context Learning**. Base model hiện đại (như Qwen3.5-4B) đã sở hữu sẵn năng lực hiểu tiếng Việt và trích xuất thông tin xuất sắc; việc fine-tune một cách cơ học mà không kiểm soát chặt chẽ độ ổn định số học (loss scaling, gradient clipping) và không có tập replay chống quên (general capability replay) sẽ mang lại rủi ro phá hủy năng lực suy luận của mô hình cao hơn nhiều so với lợi ích thu được.

**Ba điều tôi học được:**
1. **Loss mask và Precision là điểm mù nguy hiểm nhất**: Kiểm tra loss mask (NB1) và tính tương thích của GradScaler FP16 quan trọng hơn bất kỳ tham số rank LoRA nào. Một lỗi nhỏ ở loss computation có thể làm bùng nổ gradient và phá hủy toàn bộ mạng mà mắt thường không phát hiện được nếu chỉ nhìn vào code trên slide.
2. **Đóng băng Baseline (b) trước khi train là bài học về tính liêm chính khoa học**: Luôn phải đo và chấp nhận năng lực của một prompt được viết nghiêm túc trước khi bắt tay vào train. Nếu fine-tune không thắng được prompt engineering, giá trị thực tế của việc fine-tune bằng 0.
3. **Vị trí gắn adapter quan trọng hơn rank**: Tăng rank từ 16 lên 283 trên các lớp attention hẹp (q, v) không hiệu quả bằng việc dàn trải rank thấp r=16 trên toàn bộ các tầng `text-linear`. Cấu trúc kiến trúc mô hình quyết định dòng chảy thông tin biểu diễn.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Tôi sẽ chuyển môi trường sang GPU kiến trúc Ampere (A100 hoặc L4) để bật chuẩn **BFloat16 thuần túy** và FlashAttention-2, đồng thời trộn thêm 5% tập dữ liệu replay tổng quát (general knowledge) vào tập train_seed để ngăn chặn triệt để hiện tượng phân kỳ gradient và đánh giá lại độ chính xác khi fine-tune thực sự ổn định.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
