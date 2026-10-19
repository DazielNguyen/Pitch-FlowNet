# Prompt cho máy train: camera-ready PitchFlowNet

Dán toàn bộ phần dưới đây vào AI trên máy train. Tác giả ngồi ở máy đó và là người cho phép từng bước dài.

---

Bạn hỗ trợ phần thí nghiệm camera-ready của bài SOICT 2026, paper 4478, PitchFlowNet:
"PitchFlowNet: An Empirical Study of Architectural Integration in Single-Stage Text-to-Speech".
- **Tình trạng bài:** email ngày 18/10 xác nhận chấp nhận; bản cuối dài 8–11 trang, hạn nộp 30/10/2026.
- **Mốc nội bộ:** thí nghiệm phải xong muộn nhất tối 27/10; ngày 28–29/10 dành cho viết bài.
- **Ngôn ngữ:** trao đổi bằng tiếng Việt; code và tài liệu viết tiếng Anh.

## 0. Đọc trước, chưa chạy gì

1. **Lấy repo:** clone repo Pitch-FlowNet, nhánh `camera-ready-20261019`.
   - HEAD tối thiểu là `d528a90`. Nếu HEAD trên máy này khác, báo lại; không reset.
   - Repo này là kho evidence. Các thư mục `code/`, `configs/`, `runs/`, `datasets/`, `evaluation/`, `history/` ở gốc repo phải giữ nguyên.
   - Mọi việc mới chỉ làm trong `camera-ready-20261019/`.
   - **Kích hoạt ngay quy định commit:** `git config core.hooksPath tools/hooks`, rồi kiểm bằng `git config --get core.hooksPath`.
     Hook `tools/hooks/commit-msg` chặn mọi dòng attribution của AI và message sai định dạng.
2. **Đọc theo thứ tự:**
   1. `AGENTS.md`;
   2. `camera-ready-20261019/PRIORITY.md`: thứ tự công việc và trạng thái. Cập nhật file này mỗi khi một hàng đổi trạng thái, kèm commit làm bằng chứng;
   3. `camera-ready-20261019/PLAN.md`;
   4. `camera-ready-20261019/preflight.md`;
   5. `camera-ready-20261019/data-sync.md`;
   6. `camera-ready-20261019/data-manifest.json`;
   7. các file trong `camera-ready-20261019/configs/`.
3. **Ghi nhớ trạng thái hiện có:**
   - Source của workspace là bản chép của ứng viên lịch sử `badf381`, kèm các sửa đổi đã ghi trong `source-provenance-post.json`.
   - Đây không phải source đã chạy trong lịch sử, và không được gọi là source đó.

## 1. Quy tắc bắt buộc

**Phạm vi:**
- Chỉ làm Conference PitchFlowNet. Không trộn Journal HiFi-MobiNet, bản port BanhmiTTS hay ParIR/SeqIR.

**Tài nguyên máy:**
- Chỉ dùng GPU đã xác minh là trống. Kiểm read-only bằng `nvidia-smi`, `ps` và `tmux ls`.
- Không kill, tạm dừng hay giảm tài nguyên của tiến trình khác.
- Nếu GPU bận, báo GPU_BLOCKED rồi dừng.

**Môi trường:**
- Tạo env riêng cho workspace. Không sửa env đang phục vụ việc khác; không nâng driver, CUDA hay system package.
- Phiên bản đã dùng trên PC (`preflight.md`): Python 3.10.20, torch 2.12.1+cu130, pytorch-lightning 2.6.5, numpy 2.2.6.
- Nếu không tái tạo được đúng các phiên bản này, ghi rõ khác biệt vào `preflight-<máy>.md`. Không bịa phiên bản.

**Dữ liệu:**
- Không commit dữ liệu, checkpoint, WAV hay TensorBoard events. Không để dữ liệu trong thư mục OneDrive.
- Không sửa `dataset.jsonl`. File này chứa đường dẫn tuyệt đối tới cache, và split manifest kiểm đúng hash của nó.
- Hãy tạo lại đúng thư mục tuyệt đối ghi trong dòng đầu của `dataset.jsonl` (symlink cũng được).

**Tập test và kết quả:**
- Tập test không được dùng để tuning, early-stop hay chọn checkpoint.
- Không warm-start từ checkpoint Full cũ cho ablation.
- Không chạy evaluation trên tập test khi protocol đánh giá chưa được khoá.

**Bằng chứng:**
- Không dùng `weights_only=False`. Checkpoint lịch sử load qua `piper_train/camera_ready/ckpt_io.py`.
- Không coi author account A1–A6 là đã trả lời, và không đoán lý do chọn checkpoint hay session trong lịch sử.
- Giả thuyết 2.9 không phải nguyên nhân lịch sử đã xác nhận.

**Repo (quy định bắt buộc của tác giả, xem `AGENTS.md` mục Commits):**
- **Không attribution:** không bao giờ thêm `Co-Authored-By` của Claude, Codex hay bất kỳ AI nào; không thêm "Generated with…", emoji robot hay link phiên AI.
- **Định danh:** commit bằng Git identity của tác giả; không đổi `user.name` hay `user.email`.
- **Định dạng message:**
  - subject `type(scope): summary`, tối đa 72 ký tự, không có dấu chấm cuối;
  - một dòng trống;
  - body nói rõ đã đổi gì và vì sao, đã kiểm gì kèm kết quả, còn gì mở hoặc lỗi. Không ghi một phép kiểm chưa chạy.
- **Phạm vi:** mỗi commit một phạm vi, stage đường dẫn cụ thể, trên nhánh `camera-ready-20261019`.
- **Trước mỗi commit:** chạy `git diff --cached --stat`, `git diff --cached --check` và `python tools/check_tracked_package.py`.
- **Không sửa lịch sử:** không amend, rebase hay force-push lịch sử đã chia sẻ.
- Không push, merge, upload, sửa manuscript hay ledger, trừ khi tác giả yêu cầu.

**Cách làm việc:**
- Sửa lỗi tận gốc, không vá tạm hay nới test cho qua.
- Mọi run dài chỉ khởi chạy sau khi tác giả xác nhận lệnh cụ thể.

## 2. Preflight trên máy này

Ghi vào `camera-ready-20261019/preflight-<tên máy>.md`:
- **Hệ thống:** OS và WSL nếu có; GPU (model, VRAM, UUID, tiến trình đang chạy); CPU, RAM, dung lượng trống.
  - Nếu dùng WSL, ghi rõ ổ chứa file `.vhdx` và dung lượng trống thật của ổ đó.
- **Môi trường:** phiên bản Python, torch, CUDA, cuDNN, Lightning, numpy trong env mới.
- **Dữ liệu:** kiểm theo `data-sync.md`.
  - `dataset.jsonl` và `config.json` khớp hash trong `data-manifest.json`;
  - 13.100 WAV gốc khớp `raw_wav_sha256` trong `split-manifest.json`;
  - cache có đủ 39.300 file. Nếu có `file-manifest.json`, đối chiếu hash từng file.
- **Checkpoint 773 và 1079:** nếu có trên máy, kiểm SHA-256 theo `data-manifest.json`.
- **Extension MAS:** build theo PLAN mục O0.

Nếu thiếu dữ liệu hoặc hash lệch, báo BLOCKED_MISSING_INPUT kèm danh sách file, rồi dừng.

## 3. Đóng các mục mở trước khi train (PLAN O1, O2)

- **O1, test resume:**
  - chạy ở chế độ `torch.use_deterministic_algorithms(True)` với 1 thread CPU, rồi so sánh bằng nhau tuyệt đối;
  - nếu một op vẫn không tất định, so "resume vs không ngắt" với ngưỡng nhiễu "không ngắt vs không ngắt";
  - kiểm thêm optimizer, scheduler, bộ đếm, trạng thái best của ledger và seed RNG từng epoch.
- **O2, DDP so với 1 tiến trình:** viết `tests/ddp_validate_check.py` dùng `strategy="ddp"` (subprocess, gloo trên CPU), 3 câu validation (chia 2 + 1); so sánh giá trị từng câu và metric toàn cục.
- **Chạy toàn bộ test:** `python -m unittest discover -s camera-ready-20261019/tests -v`, rồi lưu log. Mọi test phải pass. Nếu không, báo nguyên nhân.

## 4. Smoke và pilot trên GPU (PLAN O3)

Chỉ làm khi GPU trống và tác giả đã đồng ý.
1. **Kiểm input:** chạy `--dry-run` cho `pilot-H-new.json` và `pilot-H-noF0.json`.
2. **Chạy từng pilot:**
   - với `--limit-train-batches 100` (2 epoch × 100 batch = 200 update G);
   - ngắt sau epoch 1 bằng `--stop-after-epochs 1`, rồi `--resume`;
   - tối đa 30 phút mỗi pilot; quá giới hạn thì dừng sạch.
3. **Đo và ghi lại:**
   - s/it ở trạng thái ổn định sau warm-up;
   - VRAM đỉnh;
   - thời gian validation và thời gian lưu checkpoint;
   - gradient hữu hạn và module bật/tắt đúng cấu hình;
   - các file `run_manifest.json`, `checkpoint_ledger.jsonl` và `val_diagnostics.jsonl` (có số liệu từng rank).
4. **Giới hạn của pilot:** checkpoint pilot không phải kết quả và không dùng để warm-start.

**Lệnh mẫu.** Chạy từ thư mục `camera-ready-20261019/src/python` sau khi đã kích hoạt env riêng. Các giá trị `<...>` do bạn điền và phải được xác nhận:

```sh
export CUDA_VISIBLE_DEVICES=<GPU đã xác minh trống>
python -m piper_train.camera_ready.train \
  --config ../../configs/pilot-H-new.json \
  --dataset-dir <thư mục data_preprocessed> \
  --split-manifest ../../split-manifest.json \
  --run-dir <thư mục outputs ngoài repo>/pilot-H-new \
  --limit-train-batches 100 --dry-run
```

## 5. Ngân sách, ETA và các quyết định (PLAN O4, O5)

Viết `camera-ready-20261019/eta.md` dựa trên số đo pilot thật, không dựa vào log cũ.
- **Lịch:** lập lịch cho cặp T1 (H-new và H-noF0) cùng ngân sách theo update G, cộng 25–30% dự phòng và thời gian evaluation, để xong trước tối 27/10.
- **So sánh phương án GPU:** DDP 2 GPU chạy nối tiếp, so với mỗi run 1 GPU. Không giả định cách thứ hai nhanh hơn; nếu đổi batch hiệu dụng thì phải ghi rõ.
- **Ngân sách rút gọn:** nếu phải rút, gắn nhãn "budget-matched pilot" và tạo config mới (`H-new-<budget>.json`, `H-noF0-<budget>.json`). Không sửa config cũ.
- **Nếu không kịp:** báo NO_FEASIBLE_TRAINING_PAIR và đề xuất chỉ đánh giá checkpoint có sẵn.

Dừng lại và hỏi tác giả, rồi ghi câu trả lời vào `experiment-plan.md` trước khi launch:
- (a) `train_shuffle`: `false` (thứ tự như lịch sử) hay `true`;
- (b) MAS decay: `4e-6`/update G (như lịch sử) hay `2e-6`/update G (đúng giá trị VITS2);
- (c) ngân sách `max_g_updates` và phương án GPU;
- (d) có thêm seed nhân bản 2345/3456 không (không chọn seed theo kết quả test);
- (e) quy tắc chọn checkpoint: `val_mel_l1_global` nhỏ nhất, chốt trước khi xem test.

## 6. Launch training T1 (chỉ sau khi tác giả xác nhận lệnh)

- **Script launch:** viết `camera-ready-20261019/launch_train.sh`.
  - Mặc định chạy dry-run.
  - Chặn khi: thư mục run đã tồn tại (trừ `--resume`), thiếu input, GPU bận, có tiến trình training khác, hoặc config còn placeholder.
  - Chỉ chạy thật khi có `--execute`.
  - Chạy trong tmux riêng `cr-<run_id>`, log ghi ra file.
- **Chống ngắt (nếu là WSL):** tạo keep-alive như runbook của máy (WSL dừng khi không còn client) và ghi lại PID. Sau mỗi lần sleep hoặc reboot, kiểm lại.
- **Theo dõi** chỉ bằng thao tác đọc: `run_manifest.json`, `checkpoint_ledger.jsonl`, `val_diagnostics.jsonl`, TensorBoard.
- **Khi run bị ngắt:**
  - kiểm `last.ckpt` (zip CRC, bộ đếm `camera_ready`, giá trị hữu hạn);
  - resume bằng `--resume` trên cùng thư mục run;
  - không bao giờ chạy hai bản cùng lúc và không tự train lại từ đầu.

## 7. Chuẩn bị evaluation song song (PLAN O7)

Chỉ chuẩn bị và test trên validation; chưa chạy hàng loạt trên test. Các phần cần chuẩn bị:
- **Synthesis:** seed cố định theo ID câu, lưu mọi lần chạy.
- **Metric:** WER macro và corpus; UTMOSv2 ghi rõ là điểm dự đoán.
- **F0 từ waveform:** cùng extractor cho bản tổng hợp và bản gốc; RMSE (Hz hoặc cents), correlation trên voiced, voicing error, độ phủ.
- **Can thiệp pitch −2/0/+2 semitone:** đặt sau bước expansion F0, giữ cố định z, duration và noise; dùng hook có test.
- **RTF:** CPU cô lập, thread khai báo rõ, có warm-up, tách model-only với end-to-end. Chỉ đo khi máy rảnh.
- **Human eval:** chuẩn bị manifest và UI blind A/B ngẫu nhiên. Không gửi lời mời, không upload.

Viết `launch_eval.sh`, mặc định dry-run.

## 8. Bàn giao

Cập nhật `camera-ready-20261019/handoff_to_mac.md`:
- việc đã xong, đang dở, thất bại;
- lệnh đã chạy;
- hash checkpoint, run manifest và ledger;
- mọi thay đổi source so với lịch sử;
- những gì còn PENDING.

Commit trên nhánh; không push trừ khi tác giả yêu cầu. Báo ngắn bằng tiếng Việt, kết thúc bằng một trong các trạng thái:
- READY_FOR_TRAINING_REVIEW
- TRAINING_RUNNING
- BLOCKED_MISSING_INPUT
- GPU_BLOCKED
- NO_FEASIBLE_TRAINING_PAIR
