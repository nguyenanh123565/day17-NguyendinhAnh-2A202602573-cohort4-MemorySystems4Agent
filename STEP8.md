# Phân tích kết quả và chọn bonus

Phân tích theo bước 8–9 của [Guide.md](Guide.md) và tiêu chí của [Rubric.md](Rubric.md), sử dụng nguyên số liệu thực tế được cung cấp. Benchmark hiện tại chạy offline; token được ước lượng bằng `estimate_tokens()` và chất lượng được chấm heuristic, nên các kết quả phản ánh hành vi trong bộ benchmark này, không phải chi phí API hay chất lượng tổng quát của model.

## 1. Baseline không có persistent memory

Ở cả hai benchmark, Baseline có **Cross-session recall = 0.00** và **Memory growth = 0 bytes**. Trong `src/agent_baseline.py`, `sessions` chỉ khóa theo `thread_id`; `user_id` không được dùng làm khóa memory. Facts được dựng lại từ message của chính thread đó, không có `User.md`. Vì vậy Baseline nhớ trong cùng thread nhưng thread mới không kế thừa thông tin người dùng, dù cùng `user_id`.

## 2. Advanced cải thiện cross-session recall

Recall tăng từ **0.00 lên 1.00** ở cả Standard và Stress; Response quality cũng tăng từ **0.50 lên 1.00**. Luồng trong `src/agent_advanced.py` là `extract_profile_updates(message)` → `upsert_fact(user_id, key, value)` → ghi `User.md` theo người dùng → đọc lại bằng `profile_store.facts(user_id)` khi trả lời ở thread mới. Đây là persistent memory, tách biệt với short-term context theo thread.

Memory growth của Advanced là **367 bytes** ở Standard và **362 bytes** ở Stress, cùng với đường ghi/đọc file trong `src/memory_store.py`, là bằng chứng persistent memory thực sự tồn tại. Hai số này là tăng trưởng của hai lần chạy riêng, không chứng minh file sẽ luôn giữ kích thước nhỏ như vậy.

## 3. Advanced có overhead ở hội thoại ngắn

Standard có Prompt tokens processed **14406** cho Baseline và **20786** cho Advanced, tức Advanced xử lý thêm **6380** prompt tokens. `_estimate_prompt_context_tokens()` tính cả nội dung `User.md`/profile, summary và messages gần nhất; profile giúp recall nhưng làm tăng tải ngữ cảnh mỗi lượt.

Standard có **Compactions = 0** ở cả hai agent, nên compact chưa tạo lợi ích bù overhead. Agent tokens only của Advanced là **3025**, thấp hơn Baseline **3209**, nhưng điều đó không đồng nghĩa prompt load thấp hơn: đây là hai chỉ số khác nhau.

## 4. Compact có lợi ở long-context

Trong Stress, Prompt tokens processed giảm từ **22083** xuống **18162**:

`(22083 - 18162) / 22083 × 100 ≈ 17.8%`.

Advanced có **1 compaction**, Baseline có **0**. `CompactMemoryManager.append()` nén messages cũ khi vượt ngưỡng token, giữ summary và một số messages gần nhất. Kết quả phù hợp với cơ chế giảm lịch sử phải xử lý lại, dù Advanced vẫn mang thêm profile. So sánh này đo toàn bộ hai agent, không tách riêng tác động compact bằng một thí nghiệm bật/tắt compact.

Compact tối ưu **Prompt tokens processed**, không phải **Agent tokens only**. Agent tokens only đếm token của message người dùng mới và câu trả lời; ở Stress, Advanced thực tế là **2769**, hơi cao hơn Baseline **2710** (thêm **59**). Vì vậy không thể kết luận compact làm giảm Agent tokens only từ kết quả này.

## 5. Trade-off và rủi ro

Persistent memory tăng recall nhưng tạo state lâu dài: memory file có thể tăng theo thời gian và fact sai/nhiễu có thể tiếp tục ảnh hưởng qua nhiều session. Compact giới hạn ngữ cảnh nhưng summary có thể mất chi tiết; implementation hiện tại giới hạn số đoạn trích và độ dài nội dung trong `summarize_messages()`.

Hệ thống mạnh hơn nhưng phức tạp hơn vì phải quản lý đồng thời short-term, persistent và compact memory. Cần guardrail khi ghi fact, xử lý correction, kiểm soát tăng trưởng file và kiểm tra thông tin quan trọng sau compact. Các số recall/quality tốt trong benchmark không loại bỏ những rủi ro này.

## Bonus: Conflict Handling

- **Problem:** Người dùng có thể sửa fact, ví dụ nơi ở cũ là Hà Nội, sau đó xác nhận “Mình hiện ở Đà Nẵng.” Nếu append cả hai giá trị, recall ở session mới có thể trả về thông tin mâu thuẫn.
- **Mechanism hiện có:** `extract_profile_updates()` dùng regex để trích khai báo, bỏ một số câu hỏi/giả định/thông tin tạm thời, loại phần phủ định fact cũ trong một số mẫu “không còn … nữa, …” và lấy assertion cuối cho cùng key. Advanced chuyển các updates sang `UserProfileStore.upsert_fact()`: thay dòng đầu cùng key, bỏ các dòng trùng key và chỉ append khi key chưa tồn tại. Vì vậy `location` mới thay giá trị cũ trong `User.md`, không append thêm fact mâu thuẫn cùng key. Đây là conflict handling đã có trong code, không chỉ là bonus proposal.
- **Benefit:** Giữ một giá trị hiện tại cho mỗi key giúp tăng độ chính xác cross-session recall và hạn chế memory growth không cần thiết do tích lũy fact cũ/trùng. Benchmark chưa đo riêng mức cải thiện do bonus này nên không gán thêm số liệu cho nó.
- **Risk:** Correction bị nhận diện sai có thể overwrite một fact đúng. Cơ chế hiện tại dựa trên regex và giá trị mới nhất; chưa có confidence threshold hay bước xác nhận riêng cho xung đột.
- **Mitigation:** Chỉ overwrite khi pattern correction hoặc khai báo giá trị hiện tại đủ rõ; trường hợp mơ hồ không tự ghi. Các bộ lọc hiện có là guardrail bước đầu, chưa bảo đảm xử lý mọi câu mơ hồ. Việc siết điều kiện ghi ở trường hợp đó là đề xuất bổ sung, chưa được triển khai trong bước phân tích này.

## Bảng benchmark thực tế

| Benchmark | Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---:|---:|---:|---:|---:|---:|
| Standard | Baseline | 3209 | 14406 | 0.00 | 0.50 | 0 | 0 |
| Standard | Advanced | 3025 | 20786 | 1.00 | 1.00 | 367 | 0 |
| Long-context stress | Baseline | 2710 | 22083 | 0.00 | 0.50 | 0 | 0 |
| Long-context stress | Advanced | 2769 | 18162 | 1.00 | 1.00 | 362 | 1 |

## Kết luận

Advanced cải thiện recall qua session nhờ `User.md`, đổi lại overhead prompt ở hội thoại ngắn. Trong Stress, một lần compact đi cùng mức giảm khoảng **17.8% Prompt tokens processed**, dù Agent tokens only hơi tăng. Conflict Handling hiện có giúp thay fact cũ theo key; độ tin cậy khi nhận diện correction và mất chi tiết khi compact vẫn cần guardrail.
