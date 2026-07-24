# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> Ở temperature 0.0 và 0.7, model bám chặt vào một sự thật duy nhất (ví dụ hồ B-52, phố đường tàu) và trình bày mạch lạc, có cấu trúc rõ ràng từ đầu đến cuối. Ở 1.2, phản hồi vẫn đọc được nhưng bắt đầu lạc đề giữa chừng: thêm một "sự thật khác" không liên quan vào cuối câu trả lời thay vì đào sâu một ý. Đến 1.8, phản hồi rõ ràng kém mạch lạc nhất: model nhảy giữa nhiều chủ đề không liên quan (ẩm thực phố Hàng rồi đột ngột chuyển sang phố đường tàu) mà không có mạch dẫn dắt, cho thấy temperature càng cao thì tính "trôi" và ngẫu nhiên trong việc chọn từ/ý càng tăng, đánh đổi lấy sự đa dạng bằng sự tập trung.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> Trợ lý soạn hợp đồng pháp lý: temperature ~0.0–0.2. Văn bản pháp lý cần tính chính xác, nhất quán về thuật ngữ và không được "sáng tạo" thêm điều khoản hay diễn đạt mơ hồ — cùng một yêu cầu đầu vào nên luôn cho ra kết quả gần như giống nhau, dễ kiểm tra và tái lập. Trợ lý viết slogan quảng cáo: temperature ~1.0–1.3. Ở đây mục tiêu là sự đa dạng và bất ngờ về ý tưởng, chấp nhận rủi ro một vài kết quả không dùng được để đổi lấy nhiều phương án sáng tạo khác nhau cho người viết chọn lọc.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> Workload: 20.000 người dùng × 2 lượt/ngày × 500 token đầu ra = 20.000.000 token đầu ra/ngày (= 20.000 × 1K token). Với giá output trong bảng: gpt-4o (0.010 USD/1K) → 20.000 × 0.010 = $200/ngày . gpt-4o-mini (0.0006 USD/1K) → 20.000 × 0.0006 = $12/ngày — model lớn đắt hơn khoảng 16,7 lần. Model lớn xứng đáng khi tác vụ đòi hỏi suy luận phức tạp, độ chính xác cao và sai sót tốn kém — ví dụ trợ lý phân tích hợp đồng pháp lý hay chẩn đoán lỗi kỹ thuật, nơi một câu trả lời sai gây thiệt hại lớn hơn nhiều lần chênh lệch chi phí. Model nhỏ là lựa chọn đúng cho các tác vụ khối lượng lớn, đơn giản và lặp lại — ví dụ phân loại yêu cầu hỗ trợ khách hàng hoặc trả lời FAQ — nơi tốc độ và chi phí quan trọng hơn độ tinh tế của câu trả lời.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> Với persona "nhà thơ", phản hồi dài (~293 từ), hoàn toàn không dùng thuật ngữ kỹ thuật — thay vào đó xây dựng một phép ẩn dụ xuyên suốt (đứa trẻ học nhận biết mưa qua quan sát, rồi liên hệ sang việc máy học nhận ra "hoa hồng" qua dữ liệu), giọng văn giàu hình ảnh, cảm xúc. Với persona "kỹ sư senior", phản hồi mở đầu bằng lối xưng hô trực tiếp, chuyên nghiệp ("Chào bạn, dưới góc nhìn của một Kỹ sư Phần mềm...") và có xu hướng đi thẳng vào định nghĩa chính xác kèm ví dụ cụ thể thay vì ẩn dụ. Có thể thấy system prompt điều khiển rất mạnh giọng văn (thơ ca vs. kỹ thuật), mức độ dùng thuật ngữ, và cấu trúc lập luận (ẩn dụ liên tục vs. định nghĩa — ví dụ), dù nội dung cốt lõi (khái niệm machine learning) không đổi.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> Với đoạn văn tiếng Việt 182 từ về Hà Nội: `count_tokens` (tiktoken, encoding của gpt-4o) đếm được **237 token**, trong khi ước lượng thô `số từ / 0.75 = 242,7 token`. Hai con số chỉ chênh khoảng **2,4%** — gần nhau hơn dự kiến. Nếu dùng ước lượng thô để dự toán ngân sách, bạn sẽ dự toán **thừa nhẹ** (~2–3%) chứ không thiếu, vì công thức `/0.75` vốn được hiệu chỉnh cho văn bản tiếng Anh (trung bình ~0,75 từ/token) nhưng vô tình lại xấp xỉ khá tốt với tiếng Việt có dấu — do mỗi từ tiếng Việt có dấu thường bị tách thành nhiều token hơn từ tiếng Anh tương đương, kéo tỷ lệ token/từ lên gần bằng mức ước lượng thô. Dù vậy, sai số này không cố định qua mọi đoạn văn nên chỉ nên dùng làm ước lượng sơ bộ, không thay thế đếm token thật khi cần độ chính xác cao.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> Chatbot văn bản (a) hưởng lợi nhiều nhất từ streaming vì người dùng đang chủ động chờ và nhìn màn hình theo thời gian thực — thấy chữ xuất hiện dần giúp giảm cảm giác chờ đợi (perceived latency) và cho phép người dùng bắt đầu đọc/hiểu ý trước khi câu trả lời hoàn tất, đặc biệt quan trọng với câu trả lời dài. Trợ lý giọng nói đọc to (b) hưởng lợi ở mức trung gian: streaming giúp giảm độ trễ trước khi câu đầu tiên được phát ra (time-to-first-audio), nhưng vì text-to-speech phải xử lý theo câu/cụm từ trọn vẹn nên lợi ích không rõ rệt bằng hiển thị chữ trực tiếp. Pipeline dịch tài liệu chạy ngầm ban đêm (c) hầu như không cần streaming vì không có người theo dõi trực tiếp — hệ thống chỉ cần nhận toàn bộ kết quả cuối cùng để ghi vào file, nên gọi API ở chế độ non-streaming đơn giản và ổn định hơn để xử lý theo lô (batch).

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> Với delay cố định, khi hàng nghìn client đồng thời gặp lỗi (ví dụ server vừa quá tải), tất cả sẽ retry sau đúng cùng một khoảng thời gian — tạo ra các đợt sóng request đồng loạt (thundering herd) liên tục dội vào server đang phục hồi, khiến nó khó gượng dậy. Exponential backoff giãn cách các lần retry ngày càng xa nhau (base_delay × 2^attempt), giúp giảm dần áp lực tổng thể lên server theo thời gian và cho hệ thống backend cơ hội phục hồi thay vì bị dồn dập ngay khi vừa rảnh tay. Vấn đề còn sót lại: dù độ trễ tăng theo cấp số nhân, các client vẫn có xu hướng đồng bộ theo từng "đợt" retry tại đúng các mốc thời gian giống nhau (vì cùng công thức, cùng thời điểm lỗi ban đầu) — vẫn tạo ra các đỉnh tải theo chu kỳ. "Jitter" (thêm một khoảng trễ ngẫu nhiên vào công thức backoff) phá vỡ sự đồng bộ này, rải đều các lần retry ra theo thời gian thay vì tập trung thành từng đợt, giúp tải lên server thực sự mượt và ổn định hơn.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> System prompt đã dùng: *"Bạn là trợ giảng AI thân thiện của khóa học lập trình. Luôn trả lời bằng tiếng Việt, ngắn gọn trong 3-4 câu. Nếu người dùng hỏi về chủ đề ngoài lập trình/AI, hãy lịch sự từ chối và hướng họ quay lại chủ đề khóa học. Không bịa đặt thông tin nếu không chắc chắn."* Hai chỗ mà nếu xóa sẽ đổi hành vi rõ rệt: (1) Cụm *"ngắn gọn trong 3-4 câu"* — xóa đi thì các câu trả lời (đã kiểm chứng thực tế: mỗi câu trả lời về Python/decorator/list-tuple đều gói trong đúng 3-4 câu) sẽ dài và lan man hơn nhiều, không còn kiểm soát được độ dài. (2) Cụm *"Nếu người dùng hỏi về chủ đề ngoài lập trình/AI, hãy lịch sự từ chối..."* — xóa đi thì khi tôi thử hỏi về "số yêu thích của tôi" (một yêu cầu ngoài chủ đề), trợ lý sẽ không còn tự động lái hội thoại quay lại lập trình như đã quan sát được ("mình xin phép tập trung hỗ trợ bạn về các chủ đề học tập..."), mà có thể trả lời tự do bất kỳ chủ đề gì được hỏi.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> Đã tái hiện được lỗi này bằng thực nghiệm: Lượt 1 tôi nói "Số yêu thích của tôi là 42, hãy nhớ giúp tôi", sau đó hỏi 4 câu về lập trình Python (Python là gì / list comprehension / list vs tuple / decorator). Đến lượt 6, khi hỏi lại "Số yêu thích của tôi là gì?", trợ lý trả lời hoàn toàn không nhớ ("mình chưa có cơ hội biết được con số yêu thích của bạn là gì đâu nè") — vì `history = history[-8:]` chỉ giữ 4 lượt gần nhất (8 message), lượt 1 (chứa thông tin số 42) đã bị cắt bỏ khỏi bộ nhớ trước khi câu hỏi ở lượt 6 được gửi đi. Cách khắc phục: thay vì cắt cứng theo số lượt, dùng kỹ thuật tóm tắt hội thoại cũ (summarization) — khi history vượt quá ngưỡng, gọi API tóm tắt các lượt sắp bị loại thành 1-2 câu (ví dụ "Người dùng thích số 42") rồi chèn tóm tắt này vào đầu system prompt hoặc như một message ẩn, thay vì xóa hẳn. Cách này giữ được các thông tin định danh/sở thích quan trọng qua nhiều lượt mà không phải tốn token giữ nguyên toàn bộ lịch sử dài.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
