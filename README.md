# MVA2027 SMOTP4SB: cả hai track

Kế hoạch đầy đủ: [PLAN.md](PLAN.md). Nội dung cuộc thi: [context.txt](context.txt).

## Ý tưởng tổ chức

- **Track 2 chứa Track 1.** Detector và tracker viết **một lần** trong `src/smotp/`, hai track chỉ khác lớp cuối.
- **Code ở đây (git), dữ liệu ở ổ E: (ngoài git).** Đường dẫn qua biến môi trường, xem `.env.example`.
- **Mọi thứ sinh ra đều xoá và tạo lại được** (`work/derived/<loại>/<config_hash>`).
- **Mọi lần chạy ghi một dòng vào `experiments/ledger.csv`** (commit, config, điểm CV, điểm public).

```
src/smotp/
  common/      formats.py (đọc/ghi + kiểm tra file nộp), paths.py   <- dùng chung 2 track
  detection/   detector, SliceTrain, suy luận                         <- dùng chung
  tracking/    YOLOv8-SMOT-style tracker, lọc track                   <- dùng chung
  prediction/  cửa sổ 60->150 frame, LSTM, loss SO-HOTA-TP            <- chỉ Track 2
  eval/        gọi evaluator của repo baseline, CV theo nhóm
configs/       common/ track1/ track2/   (mỗi run = 1 file config; hash ghi vào ledger)
scripts/       manifest.py, (tải/giải nén, tạo bộ mini, đóng gói nộp)
splits/        folds.json  <- chia THEO NHÓM CẢNH, cả nhóm dùng chung, không tự chia riêng
private_test/  track1/ track2/  run.sh + Dockerfile cho gói nộp cuối
notebooks/kaggle/   notebook mỏng, chỉ gọi code trong src/
third_party/   repo baseline (submodule): MVA2027-SMOTP4SB, TrackEval nhánh smot4sb_dev
experiments/   ledger.csv
tests/         pytest cho các quy ước định dạng (dễ lỗi lệch frame/tâm/độ phân giải)
```

Thư mục **không** vào git: `work/`, `runs/`, `submissions/`, trọng số, zip.

## Bắt đầu

```powershell
git submodule add https://github.com/IIM-TTIJ/MVA2027-SMOTP4SB third_party/MVA2027-SMOTP4SB
copy .env.example .env      # rồi sửa đường dẫn
python scripts/manifest.py  # sau khi đã tải dữ liệu
python -m pytest tests -q
```

## Luồng dữ liệu

```
raw (E:, chỉ-đọc) -> derived/tiles -> detector -> derived/detections -> tracker -> derived/tracks
                                                                          |-> Track 1: {seq}.txt
                                                                          '-> predictor -> Track 2: prediction/ + tracking/
```
Tracker và predictor chạy trên **detection cache**, nên gần như không cần GPU và lặp được hàng trăm lần mỗi giờ.

## Quy tắc nhóm

1. Một lệnh đánh giá, một nhánh TrackEval (`smot4sb_dev`, nhánh `master` chấm bằng IoU mà không báo lỗi), một `folds.json`.
2. Một người giữ quyền nộp CodaBench (10 lượt/ngày cho cả nhóm); mọi lượt nộp ghi vào ledger.
3. Chọn tham số bằng CV, không bằng leaderboard public.
