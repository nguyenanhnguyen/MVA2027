# MVA2027 SMOTP4SB: cả hai track trong một codebase

Kế hoạch và lý do thiết kế: [PLAN.md](PLAN.md) (mục 9 là phiên bản mới nhất). Nội dung cuộc thi: [context.txt](context.txt).

## Cấu trúc

```
src/smotp/
  data.py      đường dẫn (.env), danh sách sequence, nhãn train, khóa nội dung, ánh xạ ID hai gói, chia fold
  formats.py   đọc/ghi/kiểm tra file nộp của CẢ HAI track, zip
  detect.py    tile SliceTrain, huấn luyện YOLOv8, suy luận full-frame, cache detection theo nội dung
  track.py     tracker (một lõi, dùng cho cả Track 1 cả sequence và Track 2 từng cửa sổ)
  predict.py   Track 2: cửa sổ, đặc trưng, LSTM 6 mode + loss SO-HOTA-TP, baseline vận tốc hằng số, quy tắc phát/dừng
  evaluate.py  ước lượng nhanh (proxy) + gọi evaluator chính thức của repo baseline
  cli.py       điểm vào duy nhất: python -m smotp.cli <lệnh>
configs/default.toml     toàn bộ tham số
private_test/            Dockerfile + run_track1.sh + run_track2.sh (gói nộp cuối)
tests/test_core.py       kiểm thử bằng dữ liệu tổng hợp
splits/                  folds.json, idmap_pub_test.json (sinh ra, đưa vào git để cả nhóm dùng chung)
experiments/ledger.csv   sổ ghi mọi lần chạy và mọi lần nộp
third_party/             repo baseline (submodule) + TrackEval nhánh smot4sb_dev
```
Không vào git: `work/` (cache, tile), `models/`, `submissions/`, mọi `.pt`, `.zip`.

## Luồng làm việc

```
gói T1 ─┐                                       ┌─ track1 ──► {seq}.txt            (Track 1)
        ├─ detect (cache theo NỘI DUNG) ────────┤
gói T2 ─┘                                       └─ track2 ──► prediction/ tracking/ (Track 2)
   train ─► tiles ─► train-det ─► det_final.pt      train-pred ─► pred.pt
```

```powershell
copy .env.example .env                      # sửa đường dẫn; đặt trong Windows: python -m pytest tests -q
python -m smotp.cli manifest --track 1      # đếm sequence/frame, ghi work/MANIFEST_t1.json
python -m smotp.cli idmap                   # pub_test hai gói có trùng nội dung không?
python -m smotp.cli folds                   # chia 5 fold -> splits/folds.json
# Detector (GPU; laptop 4 GB không đủ => Kaggle):
python -m smotp.cli tiles --val-fold 0
python -m smotp.cli train-det --val-fold 0  # -> models/det_fold0.pt ; lặp fold 0..4 để có detector OOF
#   copy detector cuối thành models/det_final.pt
python -m smotp.cli detect --split pub_test --track 1
# Track 1
python -m smotp.cli track1 --split pub_test             # -> submissions/track1_*.zip
# Track 2
python -m smotp.cli train-pred --source gt              # hoặc --source tracker (cần detector OOF)
python -m smotp.cli eval-pred  --source gt              # proxy nhanh, quét tau
python -m smotp.cli track2 --split pub_test             # -> submissions/track2_*.zip
```
Sau khi nộp, ghi điểm vào `experiments/ledger.csv`. Score chính thức: `third_party/…/evaluation`.

## Đã kiểm thử và chưa kiểm thử

| Thành phần | Trạng thái |
|---|---|
| Định dạng file, cửa sổ, kiểm tra, zip | Đã test (dữ liệu tổng hợp) |
| Lưới tile, hộp trong tile | Đã test |
| Tracker (hai chim, bỏ sót 15%, nhiễu) | Đã test: giữ đúng ID, nội suy lấp khoảng trống |
| Dựng cửa sổ, khớp track↔GT, bộ dự đoán `cv`, proxy chấm điểm | Đã test, gồm tính chất phát thừa bị phạt |
| Pipeline Track 1 và Track 2 với detector giả | Đã test từ đầu đến file nộp, qua kiểm tra |
| Augmentation huấn luyện | Đã test hình học |
| **Huấn luyện LSTM, loss, suy luận bằng mạng** | **Chưa chạy**: máy chưa có torch (test tự bỏ qua) |
| **Tạo tile, huấn luyện và suy luận YOLOv8** | **Chưa chạy**: chưa có ultralytics, dữ liệu, GPU |
| **Dockerfile / run_track*.sh** | **Chưa build** |

## Giả định cần kiểm tra ngay khi có dữ liệu

1. `load_gt`: cấu trúc `annotations/train.json` (file_name chứa thư mục sequence, trường `frame_id`, `track_id`, `bbox`).
2. Tọa độ nộp cho sequence 1080p: trang Codabench nói "khung 3840×2160 gốc"; code hiện giữ tọa độ gốc của từng sequence. **Phải đối chiếu với `tools/validate_submission.py` và hỏi ban tổ chức.**
3. Tracker là bản viết lại theo paper; công thức chuẩn hóa khoảng cách và mọi ngưỡng cần ablation (tái lập +11 điểm) trước khi tin.
4. `folds.json` chia theo (độ phân giải, IMU) và khối id liên tiếp, chỉ là xấp xỉ "theo cảnh"; thay bằng nhãn cảnh thật nếu có.
5. Cú pháp `run_official` lấy từ tài liệu; kiểm tra bằng `--help` khi có repo.
6. IMU **không được dùng** (PLAN.md mục 9.2a). Thêm lại chỉ khi có thí nghiệm chứng minh có lợi.
