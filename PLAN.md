# Kế hoạch tham gia MVA2027 SMOTP4SB (Track 1 + Track 2)

Ngày lập: 2026-11-22. Nguồn dữ kiện: `context.txt` (website thi), repo baseline, paper MVA2025 (arXiv 2507.12832) và paper người thắng 2025 (arXiv 2507.12087), một số paper tra cứu thêm.

> **Cập nhật vòng 2 (cùng ngày):** đã đọc thêm `docs/track2.md`, `docs/cmc.md`, `docs/ppt_setup.md` và paper SAHI. Kết quả làm thay đổi kế hoạch, nhất là phần IMU (W3) và phần chọn phương pháp. Xem **mục 9 ở cuối file**, nơi có phương pháp được chọn và lý do. Mục 9 ghi đè các chỗ mâu thuẫn ở phía trên.

## Cách đọc tài liệu này

Mỗi khẳng định có một nhãn độ tin cậy:

| Nhãn | Nghĩa |
|---|---|
| **[V]** | Đã đọc trực tiếp từ nguồn (website thi, repo baseline, paper). |
| **[S]** | Chỉ thấy trong kết quả tìm kiếm (snippet), chưa đọc toàn văn. |
| **[R]** | Suy luận của tôi từ các dữ kiện trên. Đây là **giả thuyết**, chưa kiểm chứng. |

Các số liệu trong bảng kết quả MVA2025 do công cụ tóm tắt trang web, bảng bị lỗi tiêu đề. Phải đối chiếu với PDF trước khi trích dẫn chính thức (xem mục 3.3).

---

## 0. Giả định và những điều chưa biết

Các điểm sau chưa được trả lời. Kế hoạch dưới đây chọn giả định mặc định và ghi rõ phần nào thay đổi nếu giả định sai.

| # | Chưa biết | Giả định mặc định | Ảnh hưởng nếu sai |
|---|---|---|---|
| 1 | Tham gia track nào | **Cả hai.** Track 2 chứa Track 1 nên phần lớn công sức dùng chung. | Nếu chỉ Track 1 thì bỏ W4, W5. |
| 2 | GPU, VRAM, ổ đĩa của bạn | Ít nhất 1 GPU 24 GB, NVMe ≥ 500 GB. | Nếu VRAM ≤ 12 GB: dùng detector S, tile nhỏ hơn, bỏ ensemble khi huấn luyện (mục 4, W1). |
| 3 | Làm một mình hay nhóm | Một mình, khoảng 10–15 giờ/tuần. | Nhóm thì tách W1–W2 và W4 song song. |
| 4 | Kinh nghiệm MOT / dự đoán quỹ đạo | Đã dùng PyTorch, chưa quen TrackEval. | Quyết định mức độ tự viết lại so với dùng lại baseline. |
| 5 | Dung lượng thật của dữ liệu | Khoảng 100 GB, ước tính ở mục 4, W0. | Ảnh hưởng thiết kế pipeline I/O. |

---

## 1. Những dữ kiện đã xác lập

### 1.1 Về cuộc thi
- **[V]** Track 1 xếp hạng bằng SO-HOTA, đầu vào là video và IMU (tùy chọn), đầu ra là hộp và ID mỗi frame.
- **[V]** Track 2 xếp hạng bằng SO-HOTA-TP. Đầu vào là **60 frame (2 giây)** và IMU tương ứng, **không có quỹ đạo ground-truth**. Đầu ra là 6 quỹ đạo × 150 frame (5 giây), kèm file `tracking/` để scorer khớp ID.
- **[V]** Chỉ kết quả **private test** quyết định thứ hạng. Public test giới hạn 10 lần nộp/ngày. Nộp code trước 17/2/2027, kết quả 10/3/2027.
- **[V]** IMU chỉ có cho 27/128 sequence train và 7/38 sequence public test, tức khoảng 21% và 18%. Zoom quang học thay đổi tới ~10 lần trong một sequence.
- **[V]** Dữ liệu train có 1.256 track, trung bình 180 frame mỗi track. Hầu hết chim nhỏ hơn 32×32 px.
- **[V]** 41% chim biến mất khỏi tầm nhìn trước khi hết 5 giây dự đoán.
- **[V]** Luật cho phép dữ liệu và mô hình pretrained công khai, **kể cả SOD4SB (MVA2023) và SMOT4SB (MVA2025)**. Cấm dùng ground-truth tracking của Track 1 cho Track 2. Cấm dùng frame/IMU tương lai trong Track 2. Các submission chỉ khác nhau chút ít phải gộp làm một.
- **[V]** Private test chạy trên tối đa 4 GPU 48 GB, không internet, "không quá khoảng một ngày mỗi track". Phải tái hiện được điểm public test. Trong Track 2, mỗi cửa sổ chạy độc lập, cấm đọc frame của cửa sổ khác.
- **[V]** Track 1 nhận **cả sequence** làm đầu vào. Suy ra tracking offline hai chiều là hợp lệ. Luật cấm "dữ liệu tương lai" chỉ nêu ở Track 2.

### 1.2 Số liệu baseline (repo, fold validation = 26 sequence, mỗi 5 sequence lấy 1) **[V]**

| Thành phần | Kết quả |
|---|---|
| Track 1, YOLOv8-SMOT-L | SO-HOTA **0.310** (DetA 0.196, AssA 0.490, DetRe 0.240, DetPr 0.507) |
| Track 1, YOLOv8-SMOT-S | SO-HOTA 0.248 |
| Sequence sông 0101, 0111, 0121, 0126 | **Không có output nào** với cả hai detector |
| Track 2, đầu vào ground-truth, LSTM huấn luyện theo SO-HOTA-TP | 0.195 |
| Track 2, đầu vào ground-truth, constant velocity | 0.158 |
| Track 2, đầu vào ground-truth, stay (đứng yên) | 0.127 |
| Track 2, đầu vào là tracker L, LSTM/PPT | **0.043** |
| Track 2, đầu vào là tracker L, constant velocity | 0.023 |

Lỗi của tracker baseline trên fold này **[V]**:
- Recall 25,6% và precision 54,1% (bản L).
- 474 ID sinh ra, trong khi ground-truth có 248 ID (phân mảnh định danh nặng).

### 1.3 Kết quả MVA2025 (SMOT4SB)

- **[V, qua tóm tắt, cần đối chiếu PDF]** Có hai bộ số cho đội thắng: 50,59 (bảng A) và 54,90 (bảng B, đồng thời bài YOLOv8-SMOT báo 55,2 trên public test). Báo cáo không giải thích rõ vì sao hai bộ số khác nhau.
- **[V]** Hạng 1–5: DL Team, zhwa2003 (Cascade R-CNN + Swin + Hybrid-SORT), elsalabA (Co-DETR + SAHI + BoostTrack++), sgm (YOLOv7 đầu vào ~3200–3560 px + YOLOv12, OC-SORT, WBF), xmba15 (ba CenterNet + OC-SORT có bù chuyển động affine, DISK + LightGlue).
- **[V]** Phân rã đóng góp của đội thắng (bản L): OC-SORT mặc định 44,20 → +EMA hướng vận tốc (47,92) → +mở rộng bbox (51,39) → +distance penalty (55,21). Tổng **+11,0 điểm SO-HOTA từ phần association**.
- **[V]** Đổi detector từ L sang S chỉ giảm **1,4 điểm** (55,21 → 53,81) nhưng nhanh gấp 3 lần.
- **[V]** Bài của đội thắng **không có ablation** giữa SliceTrain và huấn luyện full-frame. Công dụng thực sự của SliceTrain mới chỉ là giả thuyết của tác giả.
- **[V]** Trong cùng báo cáo có sự không nhất quán: số frame 183.192 so với 108.192, và SO-DetA 51,761 so với 51,716 cho cùng cấu hình.

---

## 2. Suy luận định lượng: điểm số nằm ở đâu?

### 2.1 Ý nghĩa của DotD với S ≈ 24 px **[R, từ công thức [V]]**

DotD = exp(−D/S). Một cặp là TP ở ngưỡng α khi D ≤ −S·ln α. Lấy trung bình trên 19 ngưỡng α từ 0,05 đến 0,95, xác suất một dự đoán tính là TP xấp xỉ **exp(−D/S)**:

| Sai lệch D | Mức "điểm" xấp xỉ |
|---|---|
| 0 px | 1,00 |
| 12 px | 0,61 |
| 24 px | 0,37 |
| 50 px | 0,13 |
| 100 px | 0,016 |

Hệ quả: **chỉ các dự đoán sai lệch dưới ~50 px mới đóng góp điểm**. Với chim bay vài chục pixel mỗi giây thì chủ yếu là **1–2 giây đầu** của tầm dự đoán 5 giây. Phần đuôi gần như không có giá trị. Điều này khớp quan sát [V] rằng "stay" (0,127) và constant velocity (0,158) gần với mô hình học (0,195): trần của mô hình dự đoán thấp.

### 2.2 Có nên phát ra vị trí dự đoán khi không chắc? **[R]**

Từ bảng đếm [V] (visible+present sai = FN+FP; visible+absent = FN; gone+present = FP; gone+absent = không tính) và DetA = TP/(TP+FN+FP):

Với một frame bird visible, gọi p là xác suất dự đoán trúng ngưỡng. Phát ra vị trí làm DetA tăng nếu xấp xỉ **p > DetA/(1+DetA)**. Khi DetA ≈ 0,1 thì ngưỡng là p ≈ 0,09. Nghĩa là **hầu như luôn nên phát ra**, trừ khi gần như chắc chim đã rời khung hình hoặc sai quá xa. Hai ràng buộc cần nhớ:
- Điểm này chỉ là xấp xỉ, bỏ qua tương tác giữa các ngưỡng α và giữa các track. Phải kiểm chứng bằng mô phỏng trên evaluator của repo (nhiệm vụ trong W4).
- Quyết định "dừng" nên dựa trên xác suất biến mất, không phải một ngưỡng cố định.

### 2.3 Best-of-K và 6 mode **[R, từ mô tả [V]]**

Scorer chọn, **cho mỗi track-window**, mode có DotD trung bình cao nhất trên các frame visible, chỉ mode đó được tính điểm. Hệ quả:
- Các mode sai không bị phạt. Do đó **6 mode nên trải rộng** để bao phủ phân phối hậu nghiệm, không nên là 6 bản sao nhỏ của dự đoán trung bình.
- **Thời điểm dừng cũng là một chiều đa dạng**: mỗi mode có thể dừng ở một frame khác nhau (được phép, nhưng validator sẽ cảnh báo, xem mục 4, W7).
- Hàm loss đúng là winner-takes-all: min trên K của `1 − exp(−d/S)` trung bình trên frame visible. Baseline đã làm đúng như vậy [V] (`train_sohota.py`).

### 2.4 Trần của Track 2 do tracker, không do predictor **[R]**

Từ 1.2: chuyển từ quỹ đạo ground-truth sang tracker làm điểm giảm 75–90% (0,195 → 0,043). Chênh lệch giữa các predictor là khoảng 0,04 (GT) nhưng chênh lệch do tracker là 0,15. **Quy tắc phân bổ công sức: khoảng 65–70% cho phát hiện và tracking, 30–35% cho dự đoán**, kể cả khi chỉ thi Track 2.

Hai nguyên nhân lỗi đã được xác định [V]: recall thấp (nhất là cảnh sông) và phân mảnh ID. Cửa sổ Track 2 chỉ dài 60 frame, scorer khớp ID bằng Hungarian trên các frame quan sát (chi phí là khoảng cách tâm trung bình, yêu cầu ≥ 5 frame chung). Nên **phân mảnh ID ít gây hại hơn Track 1 nếu các mảnh vẫn khớp được với ground-truth**, nhưng đoạn track ngắn thì lịch sử vận tốc kém, và đây là điều chưa kiểm chứng (cần đo trên evaluator).

### 2.5 Ngân sách tính toán lúc private test **[R, từ [V]]**

Private test 2025 có khoảng 25.000 frame [V]. Với "một ngày mỗi track" thì ngân sách tối đa khoảng 3 giây/frame trên 1 GPU, hoặc nhiều hơn khi dùng nhiều GPU. Đội thắng 2025 chạy 0,175 s/ảnh [V]. Do đó **ensemble nhiều mô hình, TTA đa tỷ lệ và tracking hai pha (offline) đều khả thi** và nên được tính vào thiết kế, miễn là public test cũng chạy trong thời gian hợp lý và ghi lại số liệu để khai báo trên form.

---

## 3. Bằng chứng khoa học về "SOTA" cho phương pháp tổng hợp

Lưu ý quan trọng: **không có SOTA công bố cho đúng bài toán này** (theo dõi và dự đoán quỹ đạo chim nhỏ từ video drone có IMU). Dưới đây là bằng chứng từng thành phần, kèm độ tin cậy.

### 3.1 Phát hiện vật thể rất nhỏ

| Bằng chứng | Nguồn | Ý nghĩa |
|---|---|---|
| YOLOv8 L + SliceTrain (tile 1280, chồng 20%) rồi suy luận full-frame: 55,2 SO-HOTA (public test) | arXiv 2507.12087 **[V]** | Mốc tham chiếu mạnh và đã tái lập được trong repo. Chưa có ablation SliceTrain, nên cần tự kiểm chứng. |
| Đội sgm: YOLOv7 đầu vào 3200–3560 px + YOLOv12 + WBF, hạng 2 (53,85) | arXiv 2507.12832 **[V, tóm tắt]** | Đầu vào độ phân giải gốc cộng ensemble cạnh tranh sát với đội thắng. Ensemble WBF là hướng thứ hai khả thi. |
| Co-DETR + SAHI + copy-paste + chim tổng hợp: hạng 3 (43,9) | arXiv 2507.12832 **[V, tóm tắt]** | Mô hình rất lớn (304M) nhưng thấp hơn YOLOv8 44M. Gợi ý lớn hơn không đồng nghĩa tốt hơn (bị lẫn với tracker). |
| Đầu P2 (stride 4) cải thiện vật rất nhỏ: SDD-YOLO, LAF-YOLOv10, CollabOD | arXiv 2603.25218, 2602.13378, 2603.05905 **[S]** | Bằng chứng gián tiếp (bài toán drone chống UAV, không phải chim). Là một giả thuyết để ablation, không phải kết luận. |
| Phân biệt chim và drone là khó | arXiv 2504.19347 **[S]** | Không liên quan trực tiếp vì tập này chỉ có một lớp "bird". |

**Phản biện:** (a) Các số 2025 do các đội tự báo trên cùng test set nhưng khác pipeline huấn luyện và khác tracker, nên không thể tách đóng góp của detector khỏi tracker. (b) Chưa tìm được benchmark SAHI/slicing so với full-image trên chim nhỏ. (c) Baseline 2027 (YOLOv8-SMOT-L) chỉ đạt 0,310 trên fold validation trong khi bài 2025 báo 0,552 trên public test. Tôi chưa giải thích được chênh lệch này: có thể do tập đánh giá khác, cách tính S khác, hoặc baseline huấn luyện kém hơn. **Đây là việc cần kiểm tra đầu tiên (W0).**

### 3.2 Tracking cho vật thể nhỏ, camera chuyển động

| Bằng chứng | Nguồn | Ý nghĩa |
|---|---|---|
| OC-SORT + EMA hướng + mở rộng bbox + distance penalty: +11,0 SO-HOTA | arXiv 2507.12087 **[V]** | Bằng chứng mạnh nhất: association quan trọng hơn kích thước detector. |
| Hybrid-SORT, BoostTrack++, OC-SORT + bù chuyển động affine nằm sau trong bảng 2025 | arXiv 2507.12832 **[V, tóm tắt]** | Bị lẫn với detector. Không thể kết luận tracker nào tốt hơn. |
| Bù chuyển động camera (CMC), nhận biết độ cao, xác thực quỹ đạo cho vật rất nhỏ trên drone | arXiv 2504.20234; SocialTrack 2508.12777; SFTrack 2410.20079 **[S]** | Hướng CMC được nhiều bài dùng, nhưng là cảnh giao thông/đám đông, không phải chim trên bầu trời trống. |
| JitTrack lập luận CMC tường minh làm tăng chi phí và tìm cách tránh | arXiv 2608.10485 **[S]** | Nhắc rằng CMC không miễn phí về mặt tính toán. |
| Baseline CMC dùng IMU loại bỏ chỉ ~1/5 chuyển động camera và chưa cải thiện điểm (LSTM 0,394 có CMC so với 0,414 không CMC) | repo baseline **[V]** | Bằng chứng trực tiếp: IMU hiện **chưa** có lợi. |

**Phản biện:** (a) Tập này chỉ 21% sequence có IMU, nên mô hình phụ thuộc IMU phải xử lý thiếu dữ liệu ở 79% còn lại. (b) Bầu trời trống nghèo đặc trưng, nên CMC thị giác (ORB/LightGlue) có thể thất bại đúng chỗ cần nhất. (c) Hướng chuyển động camera tương lai là không biết, nên bù chuyển động quá khứ chưa chắc giúp dự đoán tương lai.

### 3.3 Dự đoán quỹ đạo

| Bằng chứng | Nguồn | Ý nghĩa |
|---|---|---|
| PPT (ECCV 2024): huấn luyện ba giai đoạn (bước kế tiếp, đích, toàn quỹ đạo) cộng bộ dự đoán Transformer | arXiv 2407.11588 **[S]** | Đây là kiến trúc mà baseline dùng. Được đánh giá trên người đi bộ (ETH/UCY), không phải chim. |
| Vận tốc hằng số là baseline mạnh cho tầm ngắn | arXiv 1911.09814 **[S]** | Khớp với quan sát của repo. |
| Dữ liệu chim: chỉ tìm được LSTM dự đoán bồ câu, so sánh với hồi quy chứ không với constant velocity | arXiv 2312.12461 **[S]** | Không có SOTA riêng cho chim. |
| Dự đoán egocentric dùng ego-motion/IMU làm đầu vào riêng cho kết quả tốt hơn (TrajMamba trên PIE/JAAD) | arXiv 2603.14739 **[S]** | Hỗ trợ ý tưởng "đưa IMU vào như một nhánh riêng", nhưng là xe/người, không phải drone nhìn chim. |
| Diffusion/flow cho đa phương thức (MID, Leapfrog, BiFlow) | arXiv 2604.00416 và các bài khác **[S]** | Ứng viên sinh 6 mode, nhưng dữ liệu của ta nhỏ (1.256 track) nên rủi ro overfit cao. |
| LSTM huấn luyện theo SO-HOTA-TP (0,195) ngang PPT (0,192) và hơn nhiều bản minADE (0,149 đến 0,164) | repo baseline **[V]** | **Loss quan trọng hơn kiến trúc.** Đây là bằng chứng rõ nhất cho thiết kế predictor. |

**Kết luận về mức độ chắc chắn:** không thể khẳng định phương pháp dự đoán nào là SOTA cho bài này. Điều duy nhất có bằng chứng [V] là (1) huấn luyện theo đúng metric, (2) có đầu dự đoán lúc biến mất, và (3) trần thấp của mọi predictor so với tracker.

---

## 4. Kế hoạch theo từng khâu

Mỗi khâu gồm: mục tiêu, các bước, giả thuyết, cách kiểm chứng, tiêu chí quyết định, rồi **phản biện**.

### W0. Hạ tầng, dữ liệu, tái lập baseline (tuần 1)

**Mục tiêu:** có pipeline chạy được từ ảnh tới file nộp, đánh giá cục bộ khớp với repo, và biết mọi con số baseline là thật.

**Các bước**
1. Clone repo với submodule, dựng môi trường (uv, Python 3.10, torch cu121, `scripts/setup_openmmlab.sh`, TrackEval nhánh `smot4sb_dev`). Lưu ý: nhánh `master` của TrackEval **sẽ chấm bằng IoU mà không báo lỗi** [V].
2. Tải hai gói dữ liệu (Track 1 và Track 2) từ Google Drive. Kiểm tra checksum và số sequence.
3. Ước lượng dung lượng: ~66.600 frame train và ~16.500 frame public test; mỗi ảnh 4K JPEG khoảng 1–1,5 MB cho ra khoảng 80–120 GB **[R]**, khớp với ước lượng của bạn. Đặt dữ liệu trên NVMe, **không chuyển sang PNG**.
4. Chạy `run_pipeline.sh` trên fold validation của repo và đối chiếu với 0,310 (Track 1) và 0,043 (Track 2). Nếu lệch đáng kể thì dừng và điều tra trước khi làm gì khác.
5. Thử `tools/validate_submission.py` và nộp **một** bản baseline lên cả hai CodaBench để xác nhận định dạng.
6. Thiết lập giao thức kiểm định (xem W6).
7. Đọc phần chưa đọc: `docs/track2.md`, `docs/cmc.md`, `docs/ppt_setup.md`, README dữ liệu.

**Phản biện**
- *Có thể trần thấp hơn chúng ta nghĩ*: nếu baseline L chỉ 0,310 trên val còn bài 2025 báo 0,55, có thể metric/split khác. Nếu sau bước 4 vẫn chênh, cần hỏi trên Discord/email `mva2027-tec@mva-org.jp` trước khi tối ưu theo con số sai.
- *Rủi ro tiêu tốn thời gian*: cài openmmlab hay lỗi. Dành tối đa 2 ngày, nếu không được thì chỉ dùng nhánh Ultralytics (YOLOv8) và bỏ mmyolo.

**Tiêu chí hoàn thành:** tái lập baseline trong ±10% và có một bản nộp hợp lệ trên cả hai leaderboard.

### W1. Phát hiện chim nhỏ (tuần 2–5)

**Mục tiêu:** tăng SO-DetRe và SO-DetPr, nhất là ở cảnh sông/đá và nước chuyển động. Đây là điểm yếu đã biết (4/26 sequence val không ra kết quả nào).

**Giả thuyết (mỗi giả thuyết phải được ablation riêng)**
- **H1.1** SliceTrain (tile 1280, chồng 20%) tốt hơn huấn luyện full-frame ở cùng bộ nhớ. *(Đội thắng không ablation, nên tự đo.)*
- **H1.2** Tiền huấn luyện trên SOD4SB (57.019 ảnh, được luật cho phép) rồi tinh chỉnh trên tập này giúp các cảnh khó. **[V]** luật cho phép, **[R]** về lợi ích.
- **H1.3** Thêm đầu P2 (stride 4) hoặc tăng độ phân giải suy luận giúp chim < 16 px. **[S]** gián tiếp.
- **H1.4** Khai thác âm tính khó (hard negative) từ cảnh sông giúp giảm FP và tăng recall ở nền phức tạp.
- **H1.5** Ensemble 2–3 detector + WBF, kèm TTA (lật, đa tỷ lệ), nâng DetPr/DetRe. **[V]** đội sgm dùng WBF, hạng 2.
- **H1.6** Lọc dương tính giả bằng bộ phân loại chim nhị phân (xmba15 dùng).

**Các bước**
1. Xây pipeline tile với **lấy mẫu thiên về dương tính**: mỗi frame cắt thành 8 tile (4 cột × 2 hàng với tile 1280, stride 1024 cho 3840×2160) **[R]**, ~533.000 tile/epoch nếu dùng hết, nên chỉ lấy tile chứa chim cộng một tỷ lệ cố định tile âm tính khó. Cache tile đã giải mã trên NVMe.
2. Chạy ablation H1.1 → H1.4 với YOLOv8-S (nhanh, chỉ kém 1,4 điểm so với L [V]) trên fold CV, rồi chuyển cấu hình tốt nhất lên L.
3. Đánh giá detector **bằng chính SO-HOTA của tracker cố định**, không chỉ AP, vì metric của cuộc thi gắn với Dot Distance.
4. Báo cáo riêng theo cảnh (trời, khuôn viên, sông) và theo độ phân giải 1080p/4K.

**Tiêu chí quyết định (đặt trước khi chạy):** một thay đổi chỉ được giữ nếu cải thiện SO-HOTA (cố định tracker) **trên trung bình CV và không làm xấu đi nhóm cảnh sông**; chênh lệch nhỏ hơn độ lệch chuẩn giữa các seed thì coi là không có tác dụng.

**Phản biện**
- *Overfit cảnh*: dữ liệu có thể chứa nhiều sequence từ cùng địa điểm. Phải chia CV theo nhóm cảnh, không theo frame (xem W6).
- *Ensemble làm tăng FP*: WBF được chọn ở bảng 2025 dựa trên cải thiện DetPr, nhưng ensemble kém điều chỉnh có thể làm giảm SO-AssA do khung hình nhấp nháy. Phải đo AssA riêng.
- *Mô hình lớn hơn không chắc tốt hơn* [V]: Co-DETR 304M thua YOLOv8 44M, và L chỉ hơn S 1,4 điểm. Ưu tiên cải tiến dữ liệu và huấn luyện hơn là tăng kích thước.
- *P2 không có bằng chứng trực tiếp trên chim*: chỉ làm nếu H1.1–H1.4 đã xong và còn thời gian.

### W2. Tracking (tuần 3–7, chạy gối với W1)

**Mục tiêu:** tăng SO-AssA và SO-DetA qua association, vì đây là nơi bằng chứng mạnh nhất (+11,0 điểm [V]).

**Giả thuyết**
- **H2.1** Tái lập YOLOv8-SMOT (EMA 0,8, mở rộng bbox ×2, distance penalty, ngưỡng IoU 0,25) trong code của mình để hiểu và làm chủ.
- **H2.2** Thay hack "mở rộng bbox + penalty" bằng **chi phí gắn trực tiếp với DotD** (cùng metric với cách chấm), tính theo kích thước S toàn tập. **[R]**, ý tưởng hợp lý nhưng chưa có bằng chứng.
- **H2.3** **Tracking offline hai chiều** trong Track 1 (có cả sequence): nối các tracklet bằng tối ưu toàn cục, nội suy khoảng trống, chạy cả xuôi và ngược. **[R]**
- **H2.4** Bù chuyển động camera (từ IMU hoặc thị giác) trước khi gán. **Chưa có bằng chứng lợi ích cho tập này** (xem W3), chỉ làm nếu W3 vượt cổng quyết định.
- **H2.5** Chuẩn hóa chuyển động theo kích thước bbox, vì zoom làm thay đổi tỷ lệ pixel. **[R]**

**Các bước**
1. Dựng bộ đánh giá tracking nhanh (TrackEval nhánh `smot4sb_dev`), chạy trên detection đã cache, **tách hẳn khỏi vòng huấn luyện detector** để lặp hàng trăm lần mỗi giờ.
2. Tái lập H2.1, rồi thêm lần lượt H2.2, H2.3, H2.5 và chạy bảng ablation.
3. Với Track 2: chạy tracker **theo cửa sổ 60 frame độc lập** và điều chỉnh riêng (ví dụ ưu tiên recall và kéo dài tracklet, vì chỉ cần khớp được ≥5 frame với ground-truth).
4. Bảng thống kê lỗi theo kiểu: số ID so với ground-truth (baseline 474/248), độ phân mảnh, các trường hợp chim bay cắt nhau (flocking).

**Phản biện**
- *Offline có thể không giúp Track 2*: trong cửa sổ chỉ có 60 frame và scorer khớp bằng Hungarian trên các frame quan sát, nên nối tracklet có thể không đổi gì. Cần đo trước khi đầu tư.
- *Nhiều đội thua dù dùng tracker "tốt hơn"*: các tracker khác nhau bị lẫn với detector, nên không thể chọn bằng tên. Tự ablation trên cùng detector cố định.
- *Tham số đội thắng được tinh chỉnh trên public test* [V: grid search]: có nguy cơ không chuyển giao sang private test. Chọn tham số bằng CV, không bằng leaderboard.

### W3. Chuyển động camera và IMU (tuần 6–9, có cổng quyết định)

**Mục tiêu:** quyết định bằng thực nghiệm xem có nên dùng IMU, và nếu có thì dùng thế nào.

**Cơ sở suy luận**
- **[V]** Baseline CMC loại bỏ khoảng 1/5 chuyển động và chưa cải thiện điểm. **[V]** Chỉ 21% sequence có IMU. **[V]** Gimbal trễ 0,5–1 giây so với hướng drone, góc gimbal và zoom thực tế không được ghi lại (chỉ có lệnh của phi công).
- **[R]** Với 79% sequence không có IMU, một mô hình sống còn nhờ IMU là rủi ro. Mọi đường dẫn IMU phải cho phép **thiếu dữ liệu** (mask).

**Các bước**
1. Đo "trần": nếu có chuyển động camera hoàn hảo (ước lượng từ chính nền ở các sequence có điểm đặc trưng), điểm tăng bao nhiêu? Nếu trần < ~1 điểm thì dừng hướng này.
2. So sánh ba cách: (a) không CMC, (b) CMC thị giác (affine/homography từ ORB hoặc LightGlue, như xmba15), (c) CMC từ IMU theo baseline `cmc/`.
3. Với dự đoán: đưa **đặc trưng IMU** (vận tốc, yaw rate, lệnh gimbal/zoom) như nhánh riêng, tương tự ý tưởng của TrajMamba **[S]**, với mask khi thiếu.
4. Thử biểu diễn dự đoán trong "hệ tọa độ ổn định" rồi cộng phân phối chuyển động camera tương lai (ví dụ giữ nguyên, hoặc dừng) như một chiều đa dạng của các mode. **[R]**, mang tính thăm dò.

**Cổng quyết định:** chỉ giữ IMU nếu cải thiện điểm **trên các sequence có IMU** mà **không làm xấu đi** các sequence không có IMU, và khoảng cải thiện vượt độ lệch chuẩn giữa các seed.

**Phản biện**
- *Có thể kết luận là "không dùng IMU"* và đó là một kết quả hợp lệ. Ban tổ chức tự nói đây là vấn đề mở [V] và việc dùng IMU là tùy chọn.
- *Rủi ro cao, lợi ích chưa rõ*: đặt sau W1, W2 và chỉ chiếm phần ngân sách nhỏ.
- *Quyền lợi bài báo*: một phân tích âm tính có hệ thống về IMU có thể là đóng góp khoa học dù không tăng điểm.

### W4. Dự đoán quỹ đạo (tuần 7–10)

**Mục tiêu:** bộ dự đoán 6 mode, huấn luyện theo SO-HOTA-TP, có đầu dự đoán biến mất, **huấn luyện trên đầu vào giống lúc kiểm tra**.

**Giả thuyết**
- **H4.1** (đã có bằng chứng [V]) Huấn luyện bằng loss `1 − exp(−d/S)` với mode tốt nhất và có đầu "visibility".
- **H4.2** **Huấn luyện trên đầu ra tracker, không phải ground-truth.** Tracker thật có nhiễu, ngắt quãng và ID sai. Để tránh detector "thuộc lòng" tập train, dùng **dự đoán out-of-fold** (K=3 fold, mô hình S) để tạo đầu vào huấn luyện cho predictor. **[R]**
- **H4.3** Dữ liệu ít (1.256 track) nên dùng **cửa sổ trượt** (stride nhỏ) và tăng cường (lật, xoay, co giãn, nhiễu vị trí kiểu tracker). Predictor nhỏ (LSTM) có khả năng đủ [V: LSTM ≈ PPT].
- **H4.4** Đa dạng hóa mode (cả hướng, tốc độ, thời điểm dừng) bằng loss winner-takes-all và có thể cộng thêm hạng tử đa dạng. **[R]**
- **H4.5** Quy tắc phát/dừng từ mục 2.2 (p > DetA/(1+DetA)) hiệu chỉnh bằng mô phỏng trên evaluator.
- **H4.6** Thử một mô hình sinh (diffusion/flow) chỉ khi H4.1–H4.5 đã xong, vì dữ liệu nhỏ.

**Các bước**
1. Dựng bộ sinh cửa sổ huấn luyện từ ground-truth và từ đầu ra tracker (OOF). Phải bám đúng định dạng cửa sổ của scorer (window_start 1, 151, 301; 60 frame quan sát, 150 frame dự đoán).
2. Tái lập LSTM/PPT của repo, rồi lần lượt thêm H4.2, H4.4, H4.5. Báo cáo bảng với **hai** loại đầu vào: ground-truth (trần) và tracker thật (điểm thực tế).
3. Đo riêng độ chính xác của đầu "visibility" (thời điểm rời khung hình).

**Phản biện**
- *Trần thấp*: ngay cả với đầu vào ground-truth, mô hình tốt nhất chỉ đạt 0,195 [V]. Nếu W4 mang lại +0,02 thì với đầu vào tracker (hiện 0,043) có thể chỉ thêm khoảng +0,005. Đó là lý do W4 chỉ chiếm ~1/3 công sức.
- *Quỹ đạo huấn luyện OOF tốn kém*: nếu thiếu tài nguyên, thay bằng **mô phỏng nhiễu tracker** (bỏ ngẫu nhiên frame, nhiễu vị trí, đổi ID) và so sánh với OOF trên một fold để xem có đủ giống không.
- *Hạng tử "dừng"*: tăng recall của detector làm thay đổi phân bố thời điểm dừng quan sát được. Predictor huấn luyện với tracker cũ sẽ lệch. Phải **huấn luyện lại predictor mỗi khi đổi detector/tracker**.

### W5. Tích hợp Track 2 (tuần 9–11)

**Mục tiêu:** pipeline hoàn chỉnh theo cửa sổ: ảnh → track → dự đoán → file nộp.

**Các bước**
1. Với mỗi cửa sổ (60 frame quan sát), chạy tracker rồi predictor, sinh `prediction/` và `tracking/` đúng định dạng. Giữ ID nhất quán giữa hai thư mục (ID dự đoán không có trong tracking sẽ không được chấm [V]).
2. Chỉ dự đoán cho chim còn có thể quan sát được ở cuối cửa sổ. Quy tắc loại track quá ngắn (<5 frame chung) vì scorer sẽ không khớp.
3. Chạy `tools/validate_submission.py`, sửa các cảnh báo "mode phủ các frame khác nhau": hợp lệ nhưng thường là lỗi.
4. Kiểm tra tính cô lập: **không đọc frame của cửa sổ khác** (cấm theo luật) và chạy được từng cửa sổ một.

**Phản biện:** đây là nơi dễ có lỗi lệch chỉ số (frame 0 hay 1, tọa độ 1080p so với 4K, tâm hay góc hộp). Tạo một bộ kiểm thử đơn vị nhỏ cho từng quy ước.

### W6. Giao thức kiểm định và chống overfit leaderboard (xuyên suốt)

**Vấn đề:** fold validation của repo chỉ có 26 sequence và 51 cửa sổ [V]. Độ lệch chuẩn giữa seed của LSTM đã là ±0,006 [V]. Public test chỉ cho 10 lần nộp/ngày, và ban tổ chức nói thẳng mục đích là để hạn chế tinh chỉnh quá mức [V].

**Thiết kế**
1. **CV theo nhóm sequence**, chia theo cảnh/địa điểm/thiết bị, không chia theo frame. Dùng 5 fold; giữ cố định.
2. Báo cáo trung bình ± độ lệch chuẩn trên ít nhất 3 seed với mọi so sánh. Thay đổi trong phạm vi nhiễu thì bỏ.
3. Báo cáo riêng: cảnh sông, sequence có IMU, độ phân giải 1080p.
4. Dùng public leaderboard **như kiểm tra cuối**, không như hàm mục tiêu. Mỗi lần nộp ghi vào sổ cùng commit và cấu hình.

**Phản biện:** với dữ liệu nhỏ, CV vẫn có thể lạc quan. Chấp nhận rằng chênh lệch dưới ~0,01 SO-HOTA là không phân biệt được.

### W7. Đóng gói private test (tuần 10–12, kết thúc trước 17/2/2027)

**Các bước**
1. Một lệnh `bash run.sh INPUT_DIR OUTPUT_DIR` cho từng track [V]. Dockerfile (khuyến nghị), trọng số, README tên phương pháp và đội.
2. Không dùng internet khi chạy. Đóng gói sẵn mọi trọng số pretrained.
3. Dùng `tools/private_test/package_inputs.py` để dựng đầu vào giả lập từ public test, chạy `run.sh`, và xác nhận **điểm tái lập được điểm trên CodaBench**.
4. Ghi lại thời gian chạy và số GPU để khai báo trên form.
5. Chạy thử trên máy sạch (hoặc container) và trên GPU 48 GB nếu có thể (VRAM lớn hơn có thể làm lộ khác biệt).

**Phản biện:** repo không chứa Dockerfile cho private test (chỉ "khuyến nghị") [V]. Tự viết và kiểm thử sớm, vì sự cố môi trường vào phút chót là rủi ro lớn nhất. Cũng lưu ý hệ điều hành mục tiêu của ban tổ chức là Linux; nếu bạn phát triển trên Windows, hãy kiểm thử trong WSL2/Docker từ đầu.

### W8. Bài báo kỹ thuật và tuân thủ (tuần 12 trở đi)

- Chuẩn bị tài liệu, ablation và code công khai (người nhận giải bắt buộc [V]). Giấy phép là MIT có chỉnh sửa; dùng thương mại phải liên hệ ban tổ chức [V].
- Bài trình bày phải **trực tiếp tại Fukuoka** để có bài trong kỷ yếu [V].
- Nếu lên top thì hạn nộp bài là 31/3/2027.

---

## 5. Lịch (từ 22/11/2026)

Công thức: ~10,5 tuần đến khi public test đóng (3/2/2027), ~12,4 tuần đến hạn nộp code (17/2/2027).

| Tuần | Khoảng thời gian | Nội dung | Cổng |
|---|---|---|---|
| 1 | 22/11–29/11 | W0: môi trường, dữ liệu, tái lập baseline, nộp thử | Baseline tái lập ±10% |
| 2–5 | 30/11–27/12 | W1: ablation detector (H1.1–H1.4) | Bảng ablation trên CV |
| 3–7 | 7/12–17/1 | W2: tracker và đo trên detection cache | Tracker mới > tái lập H2.1 |
| 6–9 | 28/12–24/1 | W3: IMU/CMC, có thể dừng sớm | Cổng IMU |
| 7–10 | 4/1–31/1 | W4: predictor | Huấn luyện trên tracker thật |
| 9–11 | 18/1–7/2 | W5 + W7: tích hợp, đóng gói | Chạy `run.sh` ổn định |
| **Khóa mô hình** | **~20/1** | Dừng thử nghiệm mới, chỉ sửa lỗi | — |
| Hạn | 3/2 | Public test đóng | — |
| Hạn | 17/2 | Nộp code | — |

Đệm: khoảng hai tuần cuối cho đóng gói và sự cố. Nếu W3 bị cắt, thời gian dồn vào W1/W2.

---

## 6. Các khẳng định mà tôi KHÔNG thể đưa ra (và cách xử lý)

| Điều chưa thể khẳng định | Lý do | Cách kiểm chứng |
|---|---|---|
| SliceTrain tốt hơn full-frame | Bài gốc không có ablation | H1.1 |
| Đầu P2 hoặc detector mới hơn (YOLO11/12, RT-DETR…) tốt hơn YOLOv8 cho chim nhỏ | Chỉ có bằng chứng gián tiếp từ bài toán drone chống UAV **[S]**; ứng viên "mới hơn" là kiến thức nền của tôi, chưa tra cứu | H1.3, ablation riêng |
| Tracker X tốt hơn tracker Y | Bảng 2025 bị lẫn detector | W2, detector cố định |
| IMU có ích | Baseline cho kết quả âm [V] | Cổng ở W3 |
| Con số 50,59 hay 54,90 là điểm chính thức của đội thắng | Hai bộ số trong tóm tắt | Đối chiếu PDF |
| Chênh lệch 0,310 so với 0,55 là do tập đánh giá | Chưa kiểm tra | W0 bước 4 |
| Quy tắc phát/dừng p > DetA/(1+DetA) tối ưu | Xấp xỉ, bỏ qua tương tác giữa ngưỡng α | Mô phỏng ở W4 (H4.5) |

### Vấn đề tuân thủ cần hỏi ban tổ chức trước khi dùng

1. Bộ SMOT4SB (MVA2025) có 128 sequence train với 1.256 ID, **trùng số liệu với tập train của năm nay**; public test hai năm cũng đều 38 sequence **[V]**. Nếu nhãn test 2025 được công bố và trùng với test 2027, **việc dùng chúng sẽ có nguy cơ vi phạm tinh thần luật**. Tôi **không đưa vào kế hoạch** và đề nghị hỏi ban tổ chức (mva2027-tec@mva-org.jp) trước khi dùng bất kỳ nhãn nào ngoài tập train.
2. Gán nhãn giả (pseudo-label) trên ảnh public test: ảnh được công bố, nhưng luật không nói rõ. Hỏi trước, không tự giả định.
3. Luật cấm các bài chỉ khác nhau chút ít trong private test [V]. Nếu bạn nộp cả hai track bằng cùng một họ phương pháp, đó là hai cuộc thi riêng, nhưng nên xác nhận.

---

## 7. Rủi ro tổng hợp

| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| Không đủ VRAM/I-O cho dữ liệu 4K | Trung bình | Cache tile, S/M detector, mixed precision, lấy mẫu dương tính |
| Overfit public leaderboard | Cao | W6: CV theo nhóm, chỉ dùng leaderboard cuối |
| Tracker là nút cổ chai Track 2 | Cao (đã biết) | 65–70% công sức vào W1, W2 |
| Cảnh sông không ra kết quả | Cao (đã biết) | Khai thác âm tính khó, huấn luyện theo cảnh |
| Môi trường private test lỗi | Trung bình | Docker và thử máy sạch từ tuần 10 |
| IMU không có ích | Cao | Cổng quyết định; coi kết quả âm là hợp lệ |
| Thiếu thời gian | Trung bình | Khóa mô hình ~20/1 |
| Hiểu sai metric/định dạng | Trung bình | Kiểm thử đơn vị quy ước, dùng validator của repo |

---

## 8. Nguồn

- Website và Codabench thi: `context.txt` (đã đọc).
- Repo baseline và docs: https://github.com/IIM-TTIJ/MVA2027-SMOTP4SB (`README`, `docs/private_test.md`, `docs/so_hota_tp.md`, `docs/baselines.md`).
- MVA 2025 challenge report: https://arxiv.org/html/2507.12832v1
- YOLOv8-SMOT (đội thắng 2025): https://arxiv.org/html/2507.12087v2
- PPT, Progressive Pretext Task Learning (ECCV 2024): https://arxiv.org/abs/2407.11588v1
- Tracking vật thể nhỏ trên drone: https://arxiv.org/pdf/2504.20234v1.pdf, https://arxiv.org/pdf/2508.12777v3, https://www.arxiv.org/pdf/2410.20079, https://arxiv.org/pdf/2608.10485
- Đầu P2 / vật thể rất nhỏ: https://arxiv.org/pdf/2603.25218, https://www.arxiv.org/pdf/2602.13378, https://arxiv.org/html/2603.05905
- Dự đoán quỹ đạo egocentric: https://arxiv.org/html/2603.14739, https://arxiv.org/html/2403.19026v3
- Dự đoán chuyển động chim bằng LSTM: https://arxiv.org/html/2312.12461v1
- Vận tốc hằng số là baseline mạnh: https://arxiv.org/pdf/1911.09814
- SAHI, slicing-aided fine-tuning (ICIP 2022): https://arxiv.org/abs/2202.06934
- Repo baseline, các docs đọc thêm ở vòng 2: `docs/track2.md`, `docs/cmc.md`, `docs/ppt_setup.md`.

---

## 9. Vòng suy luận 2: phương pháp được chọn và lý do

Phải nói thẳng: **không có phương pháp nào "chắc chắn" theo nghĩa toán học** khi chưa chạy thực nghiệm trên dữ liệu này. Thứ tôi làm được là chọn phương pháp **có bằng chứng định lượng trực tiếp [V], phụ thuộc ít nhất vào các giả thuyết chưa kiểm chứng, và có quy tắc dừng rõ ràng**. Sau đây là các dữ kiện mới và hệ quả.

### 9.1 Dữ kiện mới (đều [V], từ `docs/cmc.md`, `docs/track2.md`, `docs/baselines.md`, SAHI)

**Về IMU và chuyển động camera (đảo ngược một phần kế hoạch cũ)**
- Chuyển động hình ảnh trung vị của chim trong cửa sổ 2 giây là **66 px**, trong đó điểm cố định trong thế giới cũng dịch **60 px** chỉ vì camera. **Gần như toàn bộ chuyển động trong ảnh là do camera.**
- Trên 27 sequence có IMU (5.196 cửa sổ): vận tốc hằng số trên tọa độ ảnh thô đạt **0,298**; "đứng yên" đạt 0,202; vận tốc chim đã trừ camera đạt 0,206.
- Nếu biết **chuyển động camera tương lai thật** thì đạt **0,423**. Khoảng cách 0,298 → 0,423 là đòn bẩy lớn nhất còn lại, nhưng **không ai dự báo được nó**: các bộ hồi quy ridge từ IMU hoặc từ chuyển động đo trên frame chỉ cho 0,197–0,219, **thấp hơn** cả việc "giữ vận tốc camera cuối cùng" (0,236), và thấp hơn vận tốc hằng số thô (0,298).
- Đưa thêm đặc trưng camera/IMU vào LSTM làm **giảm** điểm (held-out 4 recording, thô 0,414): thêm chuyển động đã bù 0,394; thêm lệnh gimbal/zoom/yaw 0,349; thêm cả hai 0,338. Lệnh của phi công giảm mạnh nhất. Lý do tác giả đưa ra: chỉ có 4 recording, mô hình học thói quen bay của từng chuyến và không chuyển giao được.
- Chỉ có 5/26 sequence validation có IMU, quá ít để đánh giá.

**Về tập dữ liệu và đánh giá**
- Public test chỉ có **105 cửa sổ**, validation có **51 cửa sổ**. Tập huấn luyện stride-1 có 65.165 cửa sổ (rất tương quan với nhau).
- Cảnh sông: bốn sequence 0101, 0111, 0121, 0126 đều **1920×1080** và chim **lớn hơn** (căn bậc hai diện tích trung bình 41,7 px, so với 16,5 px ở 4K) nhưng baseline cho **không có đầu ra nào**. Tài liệu kết luận nguyên nhân là nền lộn xộn (đá, nước chuyển động), không phải kích thước.
- Recall chim của baseline L là **25,6%** (8.459/32.999 hộp), precision 54,1%; có 474 ID so với 248 ID ground-truth.
- Bẫy khi chạy PPT của repo: `block_size` 128 phải thành 256, `minValue=200` làm không lưu checkpoint, `d_scale` phải đặt 500000, nếu không loss làm đa dạng mode về 0. Đã được vá trong repo.

**Về slicing (SAHI, `arXiv:2202.06934`, AP50, một lần chạy mỗi cấu hình)**
- Trên VisDrone: huấn luyện thường 25,8–29,4; chỉ SAHI khi suy luận cho +5,1 đến +5,3; huấn luyện theo lát (SF) + SAHI cho **+12,7 đến +14,1**.
- Trên xView (vật thể rất nhỏ): huấn luyện thường 2,1–2,2; SF + SAHI đạt **15,8–20,6**.
- Thêm suy luận full-image cạnh SAHI làm **giảm** AP vật nhỏ trên xView.
- Hạn chế: chỉ 3 detector (FCOS, VFNet, TOOD), bộ xView tự chia, không có phương sai giữa các lần chạy.
- Một chỗ chưa khớp: một số số "tăng" trong văn bản không trùng với bảng. Dùng số trong bảng.

**Chưa giải quyết được:** bản PDF của báo cáo MVA2025 vượt giới hạn 10 MB của công cụ tải, nên **chưa đối chiếu được hai bộ điểm 50,59 và 54,90** (mục 1.3). Cách giải quyết: bạn tải PDF về máy và đưa đường dẫn, hoặc dán các bảng kết quả vào cuộc trò chuyện.

### 9.2 Hệ quả cho từng quyết định

**(a) Dự đoán: làm việc trên tọa độ ảnh thô, bỏ phân rã camera/chim và bỏ IMU. Mức độ chắc: cao.**
Cả ba chuỗi bằng chứng [V] cùng chỉ một hướng: (1) phân rã thua vận tốc hằng số thô (0,236 so với 0,298), (2) dự báo camera tương lai không làm được, (3) thêm đặc trưng IMU làm giảm điểm do học thuộc chuyến bay. Thêm vào đó 79% sequence không có IMU. Vì vậy **W3 được hạ từ "khám phá có cổng" xuống "chỉ làm nếu còn dư thời gian"**, ở dạng một thí nghiệm âm tính (phục vụ bài báo), không phải thành phần của mô hình chính. Cổng cũ (IMU chỉ giữ nếu có lợi) vẫn giữ nguyên, nhưng xác suất vượt cổng, theo bằng chứng, thấp.

Hệ quả phụ: 6 mode được học bằng winner-takes-all sẽ tự khám phá các kiểu "camera tiếp tục / dừng / chim rẽ", không cần mô hình hóa camera tường minh.

**(b) Phát hiện: SliceTrain có bằng chứng ủng hộ độc lập. Mức độ chắc: trung bình đến cao.**
Bài của đội thắng không có ablation, nhưng SAHI cho thấy huấn luyện theo lát cho +12 đến +17 AP50 trên hai bộ dữ liệu vật nhỏ. Đây là bằng chứng độc lập, khác miền (không phải chim), nên H1.1 nâng từ "chưa có bằng chứng" lên "có bằng chứng gián tiếp khá". Vẫn nên đo trên YOLOv8-S để chắc. Bằng chứng SAHI cũng gợi ý **không cộng suy luận full-image với suy luận theo lát** cho vật nhỏ; đội thắng suy luận full-frame bằng mạng tích chập hoàn toàn nên không mâu thuẫn.

**(c) Phát hiện: lỗ hổng lớn nhất là cảnh sông và recall. Mức độ chắc: cao rằng đây là nơi cần xử lý, thấp về giải pháp.**
Một nửa lỗi Track 2 đến từ recall 25,6%. Bốn trên 26 sequence validation (khoảng 15%) bằng 0. Giả thuyết mới, **H1.7**: do các cảnh sông 1080p có chim lớn gấp ~2,5 lần và nền đặc biệt, detector huấn luyện chủ yếu trên chim ~16 px sẽ thiếu mẫu. Cách kiểm chứng rẻ:
1. Đếm số sequence 1080p và cảnh sông trong tập train (W0). Nếu rất ít thì đây là thiếu dữ liệu.
2. Thử tăng cường co giãn mạnh (phóng 4K tile ×2 để tạo chim ~32–40 px) và copy-paste chim lên nền đá/nước (elsalabA báo AP50 0,54 → 0,59 nhờ paste [V, tóm tắt]).
3. Tinh chỉnh riêng trên các sequence 1080p và đo trên các sequence sông được giữ lại. Đây là việc phải làm theo nhóm cảnh, không theo frame.

**(d) Tracking: giữ YOLOv8-SMOT làm lõi. Mức độ chắc: cao.**
Là thành phần duy nhất có ablation định lượng trực tiếp (+11,0). Các đề xuất khác (cost bằng DotD, nối offline hai chiều, chuẩn hóa theo kích thước bbox) là **giả thuyết cần ablation**, không phải lõi. Với bù chuyển động camera trong tracker: chuyển động camera giữa hai frame kề nhau khoảng 1–2 px so với bbox ~16 px, nên lợi ích dự kiến nhỏ; xếp cuối.

**(e) Một đòn bẩy rẻ chưa được kế hoạch cũ nhắc đủ: lọc track theo evaluator.**
Precision track chỉ ~54% và 474 ID so với 248. Track giả phát ra dự đoán sẽ là false positive. Cần tinh chỉnh **ngưỡng độ tin cậy, độ dài tối thiểu của track, và quy tắc loại track ở rìa khung** trực tiếp trên `evaluation.so_hota_tp` bằng CV. Chi phí thấp, kết quả dễ đo; **làm trong tuần đầu sau khi có detector chạy được**.

**(f) Huấn luyện dự đoán: LSTM huấn luyện theo SO-HOTA-TP, có đầu visibility, đầu vào là đầu ra tracker (OOF). Mức độ chắc: trung bình đến cao.**
Phần "huấn luyện theo metric" là [V] (+0,03 đến +0,05 so với minADE). Phần "huấn luyện trên đầu ra tracker" vẫn là [R] nhưng hợp lý vì baseline đã cho thấy điểm giảm 75–90% khi đổi đầu vào.

### 9.3 Ước lượng thô về điểm số (suy luận, không phải số đo)

Với tracker baseline, tỷ số điểm Track 2 so với đầu vào ground-truth là 0,043/0,195 ≈ 0,22, gần bằng recall 0,256. Nếu giả thuyết "điểm ≈ điểm ground-truth × recall hiệu dụng" đúng một cách xấp xỉ, thì **nâng recall từ 0,26 lên ~0,6 thì Track 2 lên ~0,10 (gấp ~2,3 lần)**, còn cải thiện predictor trọn vẹn nhất (+0,02 trên ground-truth) chỉ thêm ~+0,005. Đây chỉ là **một điểm dữ liệu** để rút ra tỷ lệ nên độ tin cậy thấp. Dùng nó để **sắp xếp thứ tự ưu tiên**, không để dự báo điểm.

### 9.4 Phương pháp được chọn, xếp theo mức độ chắc chắn

| Bậc | Thành phần | Bằng chứng | Mức chắc | Xử lý |
|---|---|---|---|---|
| **A** (làm chắc chắn trước) | YOLOv8 L/M + SliceTrain (tile 1280, chồng 20%, lấy mẫu thiên dương tính), suy luận full-frame | Đội thắng 2025 [V] + SAHI [V] | Cao | Tái lập, ablation trên S |
| A | Tracker YOLOv8-SMOT (EMA 0,8, bbox ×2, distance penalty) | Ablation +11,0 [V] | Cao | Tái lập trong code riêng |
| A | LSTM, loss `1 − exp(−d/S)` mode tốt nhất, đầu visibility, tọa độ ảnh thô, không IMU | Baseline [V] + CMC [V] | Cao | Dùng làm mô hình dự đoán chính |
| A | Lọc track theo evaluator (ngưỡng conf, độ dài) | Suy luận từ precision 54% [R] | Trung bình (rẻ để đo) | Làm sớm |
| **B** (đo rồi mới giữ) | Pretrain SOD4SB, tăng cường co giãn/copy-paste, hard negative cảnh sông | elsalabA [V, tóm tắt], luật cho phép [V] | Trung bình | Ablation theo nhóm cảnh |
| B | Predictor huấn luyện trên đầu ra tracker OOF | [R] | Trung bình | So với mô phỏng nhiễu |
| B | Nối tracklet offline hai chiều (Track 1) | [R] | Thấp–trung bình | Chỉ giữ nếu AssA tăng trên CV |
| B | Ensemble + WBF, TTA | sgm hạng 2 [V, tóm tắt] | Trung bình | Chỉ khi FP không tăng |
| **C** (chỉ nếu dư thời gian) | IMU/CMC, P2, diffusion/flow | Bằng chứng âm hoặc gián tiếp | Thấp | Chỉ coi là thí nghiệm |

### 9.5 Điều gì sẽ làm cho kết luận "chắc chắn" thật sự

Mỗi dòng dưới đây là một **phép đo quyết định**. Khi đã có đủ, bậc A mới được xem là đã xác nhận:

1. **Tái lập baseline** (W0): 0,310 và 0,043 trong ±10%. Nếu không, dừng và hỏi.
2. **Ablation SliceTrain** trên S, nhiều seed, trung bình ± độ lệch chuẩn trên CV theo nhóm.
3. **Tách lỗi cảnh sông**: đếm sequence 1080p/sông trong train, đo recall từng nhóm trước và sau H1.7.
4. **Tracker cố định, thay detector** (và ngược lại) để tách đóng góp, vì bảng MVA2025 bị lẫn hai yếu tố.
5. **Đối chiếu PDF MVA2025** để chốt hai bộ điểm.
6. Bốn phép đo trên cùng một lệnh đánh giá `evaluation.so_hota_tp` và `TrackEval` nhánh `smot4sb_dev`.

### 9.6 Giới hạn của chính phân tích này

- **Phương sai rất lớn**: 105 cửa sổ public và 51 cửa sổ validation, độ lệch chuẩn giữa seed ±0,006 trên 0,195. Chênh lệch dưới khoảng 10% tương đối giữa hai phương pháp **không thể phân biệt**. Quy tắc chọn: trong số các ứng viên nằm trong một sai số chuẩn, chọn cái **đơn giản và nhất quán nhất**. Thứ hạng cuối cùng trên Track 2 vì thế sẽ có một phần may rủi mà không phương pháp nào xóa được.
- Tất cả con số của CMC là trên **4 recording**; kết luận "IMU không giúp" có thể thay đổi nếu private test có nhiều chuyến bay hơn, nhưng không có cách nào kiểm chứng điều đó trước.
- Nhiều số liệu ở mục 1.3 vẫn là từ tóm tắt tự động (xem 9.1, phần "Chưa giải quyết được").
- Các ưu tiên ở 9.3 dựa trên một điểm dữ liệu duy nhất.
