# Prompt: train the two ablations on the PC (2 × RTX 4070 Ti)

Bạn hỗ trợ phần thí nghiệm camera-ready của bài SOICT 2026, paper 4478, PitchFlowNet.
Prompt này dành cho máy PC của tác giả, Windows có WSL `Ubuntu-24.04`, user `dev`. Máy có 2 × RTX 4070 Ti, mỗi card 12 GB.
- **Ngôn ngữ:** trao đổi với tác giả bằng tiếng Việt; code, tài liệu và commit viết tiếng Anh.
- **Mốc thời gian:** thí nghiệm xong muộn nhất tối 27/10; ngày 28–29/10 dành cho viết bài.

## 0. Bối cảnh, đọc trước, chưa chạy gì

**Ba review đều nêu cùng một điểm yếu chính:** chỉ có baseline và full, thiếu ablation. Tác giả đã quyết định làm ablation bỏ từng nhóm (leave-one-group-out), cùng một công thức cho mọi run (`docs/camera-ready/experiment-plan.md`):
- **Ngân sách:** **200 epoch** mỗi run = 200 × 782 batch = **156.400 update G**, tức 18,2% số epoch lịch sử. Tác giả hạ từ 275 xuống 200 ngày 21/10, để chạy đủ 8 tổ hợp.
- **Cấu hình:** batch 16, 1 GPU mỗi run, không DDP; seed 1234; `train_shuffle=false`; MAS decay `4e-6`.
- **Chọn checkpoint:** `val_mel_l1_global` nhỏ nhất, không bao giờ dùng tập test.

| Run | GPU | Trạng thái lúc viết prompt (20/10 03:00) |
|---|---|---|
| `PFN-Full-275ep-s1234` (mốc so, dừng ở epoch 200) | MardeusPC, RTX 5070 | **đang train**, bắt đầu 20/10 02:29, dự kiến xong khoảng 21/10 10:30 |
| `PFN-NoF0-200ep-s1234`, rồi `PFN-Baseline-VITS2-200ep`, rồi `PFN-Baseline-F0-200ep` | MardeusPC, RTX 5070 | chạy sau Full |
| **`PFN-NoBigVGAN-200ep-s1234`**, rồi **`PFN-Baseline-BigVGAN-200ep`** | **PC này, GPU 0** | **việc của bạn** |
| **`PFN-NoVITS2-200ep-s1234`**, rồi **`PFN-Baseline-200ep`** | **PC này, GPU 1** | **việc của bạn** |

**Cặp lịch sử (Baseline 773, Full 1079) vẫn là kết quả chính của bài.** Không train lại, không warm-start từ các checkpoint đó.

**Đọc theo thứ tự:**
1. `AGENTS.md`;
2. `docs/camera-ready/README.md`: bố cục repo;
3. `docs/camera-ready/experiment-plan.md`;
4. `docs/camera-ready/PRIORITY.md`;
5. `docs/camera-ready/runbook-MardeusPC.md`: cùng bộ công cụ, khác máy;
6. `docs/camera-ready/eta.md`.

## 1. Quy tắc bắt buộc

**Dự án khác trên máy này:**
- Máy đang hoặc vừa chạy Piper-SeqIR, với các tmux session `piper-ir`, `post-seqir`, `tb-seqir`.
- **Không attach, không gửi phím, không kill, không tạm dừng, không giảm tài nguyên** của các session và tiến trình đó.
- Không sửa env `base`/`edgetts` hay bất kỳ env nào khác.
- Chỉ kiểm bằng lệnh chỉ đọc (`nvidia-smi`, `ps`, `tmux ls`). Nếu SeqIR hoặc phiên evaluation/timing tự động của nó còn chạy trên GPU, báo **GPU_BLOCKED** và dừng.

**Dữ liệu:** dữ liệu gốc nằm ở `~/data_preprocessed`, `~/data/wavs`, `~/test_entries_500.jsonl` (home của user `dev`, tức `<WSL_DEV_HOME>`). Chỉ đọc, không sửa, không chạy preprocess lại.

**Thư mục riêng cho camera-ready:**

| Mục đích | Đường dẫn |
|---|---|
| Repo | `~/Pitch-FlowNet-cr` |
| Kết quả run | `~/outputs-pfn-cr` |
| Env | `pfn-cr` |
| TensorBoard | cổng `6007` (cổng 6006 có thể đang dùng cho `tb-seqir`) |

**Tập test và kết quả:** không dùng tập test để tuning, early-stop hay chọn checkpoint. Không evaluate trên tập test khi protocol chưa khoá.

**Bằng chứng:**
- Không bịa kết quả.
- Không dùng `weights_only=False`.
- UTMOSv2 là điểm dự đoán, không phải MOS.

**Repo:**
- Commit nhỏ, đúng phạm vi.
- **Không thêm dòng `Co-Authored-By` hay "Generated with" của AI vào commit hoặc PR.**
- Không push, merge, upload nếu tác giả chưa yêu cầu.
- Trên máy này không có các file gốc chứa đường dẫn máy, nên **không chạy `tools/sanitize_for_git.py`** (xem `preflight-MardeusPC.md`). Trước commit chỉ chạy `tools/check_tracked_package.py`.
- Để tránh xung đột với MardeusPC, máy này chỉ commit các file của riêng nó: `docs/camera-ready/preflight-Daziel.md`, `docs/camera-ready/runs-Daziel.md` và config mới nếu tác giả yêu cầu. Đừng sửa `PRIORITY.md`; ghi đề xuất cập nhật vào `runs-Daziel.md`.

**Cách làm việc:**
- Sửa lỗi tận gốc, không nới test cho qua.
- **Mỗi lệnh `--execute` chỉ chạy sau khi tác giả xác nhận đúng lệnh đó.**

## 2. Preflight, chỉ đọc (ghi vào `docs/camera-ready/preflight-Daziel.md`)

1. **GPU rảnh:**
   - `nvidia-smi`: mỗi GPU dùng dưới 3.000 MiB, tải dưới 10%.
   - `ps -eo pid,etime,cmd | grep -iE "seqir|piper|train" | grep -v grep`.
   - `tmux ls`.
2. **Chỗ trống:**
   - `df -h ~`.
   - Ổ C của Windows, nơi chứa file `.vhdx` của WSL, có khoảng 50 GB trống vào 19/10. Mỗi run cần khoảng 3 GB (checkpoint, TensorBoard, log). Dưới 15 GB trống thì báo tác giả.
3. **Dữ liệu:** kiểm SHA-256 của `dataset.jsonl`, `config.json` và `test_entries_500.jsonl` theo `manifests/camera-ready/data-manifest.json`. Ngày 19/10 cả ba đều đã khớp. Kiểm thêm có đủ 39.300 file cache. Các kiểm tra đều chạy với `nice -n19 ionice -c3`.
4. **Hệ thống:** RAM, CPU, phiên bản driver.

## 3. Dựng môi trường riêng (không đụng gì của SeqIR)

1. **Repo:**
   ```sh
   git clone -b camera-ready-20261019-v2 https://github.com/DazielNguyen/Pitch-FlowNet.git ~/Pitch-FlowNet-cr
   ```
   - Ghi lại HEAD.
   - **Bật hook commit ngay:** `cd ~/Pitch-FlowNet-cr && git config core.hooksPath tools/hooks`, rồi kiểm bằng `git config --get core.hooksPath`. Đọc mục **Commits** trong `AGENTS.md`: commit bằng Git identity của tác giả, không thêm attribution của AI, message dạng `type(scope): summary`.
   - Nhánh trên GitHub phải chứa commit có config `configs/PFN-*/PFN-*-200ep.json` và `tools/camera_ready/launch_train.sh` có tuỳ chọn `--gpu`. Nếu thiếu, báo tác giả để push; không tự tạo lại.
2. **Env:**
   - Dùng conda có sẵn, tạo env mới `pfn-cr`: Python 3.10.20, `torch==2.12.1` (index cu130), `pytorch-lightning==2.6.5`, `numpy==2.2.6`, cùng các gói còn lại trong `code/camera-ready-source/src/python/requirements.txt`.
   - Kiểm `torch.cuda.device_count() == 2`. Ghi phiên bản thật vào preflight.
3. **Extension MAS:**
   ```sh
   cd ~/Pitch-FlowNet-cr/code/camera-ready-source/src/python/piper_train/vits/monotonic_align
   mkdir -p monotonic_align && cythonize -i core.pyx && mv core*.so monotonic_align/
   ```
4. **Biến môi trường:** tạo `~/pfn-cr.env`, rồi `source ~/pfn-cr.env` trước mọi lệnh:
   ```sh
   export CR_OUTPUTS=$HOME/outputs-pfn-cr
   export CR_PYTHON=<đường dẫn python của env pfn-cr>
   export CR_TENSORBOARD=<đường dẫn tensorboard của env pfn-cr>
   export CR_TB_PORT=6007
   ```
5. **Test:**
   ```sh
   cd ~/Pitch-FlowNet-cr && $CR_PYTHON -m unittest discover -s tests/camera_ready -v
   ```
   - Lưu log vào `$CR_OUTPUTS/tests/`.
   - **Kết quả mong đợi: 24/25 pass.** Chỉ có `ValidationDDP` báo ERROR (PLAN O2), và nó không cần cho run 1 GPU.
   - Kết quả khác thì báo nguyên nhân và dừng.
6. **Phía Windows, tác giả làm:**
   - tắt sleep khi cắm điện: `powercfg /change standby-timeout-ac 0`;
   - tạm dừng Windows Update;
   - giữ WSL chạy: `powershell -ExecutionPolicy Bypass -File <repo>\tools\camera_ready\wsl_keepalive.ps1 start -Distro Ubuntu-24.04`. Cần một bản repo trên Windows, hoặc gọi file trong `~/Pitch-FlowNet-cr` của user `dev` qua đường dẫn mạng `\wsl.localhostBUNTU-24.04`.

## 4. Launch hai run (chỉ sau khi tác giả xác nhận lệnh)

```sh
source ~/pfn-cr.env && cd ~/Pitch-FlowNet-cr
bash tools/camera_ready/tensorboard.sh start                     # http://localhost:6007
bash tools/camera_ready/launch_train.sh configs/PFN-NoBigVGAN/PFN-NoBigVGAN-200ep.json --gpu 0   # dry run
bash tools/camera_ready/launch_train.sh configs/PFN-NoVITS2/PFN-NoVITS2-200ep.json --gpu 1       # dry run
# after the author's OK:
bash tools/camera_ready/launch_train.sh configs/PFN-NoBigVGAN/PFN-NoBigVGAN-200ep.json --gpu 0 --execute
bash tools/camera_ready/launch_train.sh configs/PFN-NoVITS2/PFN-NoVITS2-200ep.json --gpu 1 --execute
```

**Launcher (`--gpu N`):**
- Đặt `CUDA_DEVICE_ORDER=PCI_BUS_ID` và `CUDA_VISIBLE_DEVICES=N`, nên số GPU khớp với `nvidia-smi`.
- Giữ khoá `$CR_OUTPUTS/locks/gpu<N>.lock`, mỗi GPU chỉ chạy một run.
- Chặn khi GPU đó đang bận.

**Sau khi launch:**
- Sau 2–3 phút, kiểm cả hai run đều đang train: đọc log, `nvidia-smi`, `run_manifest.json` (commit đúng, `dirty_paths: []`, `max_g_updates: 215050`).
- **Hết epoch đầu:** ghi tốc độ thật (s/it, phút mỗi epoch), VRAM đỉnh và thời gian xong dự kiến vào `runs-Daziel.md`. Để tham khảo, trên RTX 5070: pilot đo 13,2 phút mỗi epoch (`eta.md`), còn run Full đang chạy cho thấy khoảng 1,6 it/s. Đó là số của 5070, không phải của 4070 Ti.
- **VRAM:** nếu một run OOM, không tự đổi batch hay config. Báo tác giả kèm log.

## 5. Theo dõi và khôi phục

- **Theo dõi** chỉ bằng thao tác đọc: `tmux attach -t cr-<run_id>` rồi `Ctrl-b d` (không bao giờ `Ctrl-c`), `checkpoint_ledger.jsonl`, `val_diagnostics.jsonl`, TensorBoard.
- **Sau khi máy tắt hoặc mất điện:**
  ```sh
  source ~/pfn-cr.env && cd ~/Pitch-FlowNet-cr
  bash tools/camera_ready/tensorboard.sh start
  bash tools/camera_ready/resume_train.sh configs/PFN-NoBigVGAN/PFN-NoBigVGAN-200ep.json --gpu 0   # dry run, checks last.ckpt
  bash tools/camera_ready/resume_train.sh configs/PFN-NoVITS2/PFN-NoVITS2-200ep.json --gpu 1
  # then the same with --execute, after the author's OK
  ```
  - Nếu `last.ckpt` hỏng, thêm `--fallback` để dùng `epoch-*.ckpt` mới nhất.
  - **Không bao giờ train lại từ đầu, và không chạy hai bản của cùng một run.**
  - Ghi mỗi lần bị ngắt vào `runs-Daziel.md`.

## 6. Khi hai run đầu xong (dự kiến khoảng 23/10)

1. **Chốt checkpoint của mỗi run** ngay khi run kết thúc với exit 0: `bash tools/camera_ready/finalize_run.sh <config> --expect-epochs 200`. Lệnh này chép bản chỉ đọc của `best-epoch=…ckpt` (checkpoint dùng để evaluate) và `last-epoch=199.ckpt` (epoch 200) vào `$CR_OUTPUTS/final/<run_id>/`, kèm `SHA256SUMS` và `final.json`. Nếu dùng watchdog (`tools/camera_ready/auto_resume.sh … --start --gpu N`), nó tự làm bước này.
   **Ghi kết quả:** kiểm `best-*.ckpt` và `last.ckpt` bằng `tools/camera_ready/check_ckpt.py`. Ghi SHA-256, số byte, epoch, `g_updates` và điểm validation tốt nhất vào `runs-Daziel.md`.
2. **Run tiếp theo trên PC** đã có trong kế hoạch (`experiment-plan.md`, bảng 8 tổ hợp, mở rộng ngày 21/10). Config 275 epoch đã có sẵn. Launch từng run **chỉ sau khi tác giả xác nhận đúng lệnh đó**:
   - GPU 0: `configs/PFN-Baseline-BigVGAN/PFN-Baseline-BigVGAN-200ep.json` (thêm-một: chỉ BigVGAN, đúng câu hỏi của Review 1);
   - GPU 1: `configs/PFN-Baseline/PFN-Baseline-200ep.json` (mốc so cho nhóm thêm-một; baseline strict mới, **không phải** baseline lịch sử 773);
   - `PFN-Baseline-F0-200ep` dự kiến chạy trên MardeusPC. Nếu máy này rảnh trước và tác giả đồng ý, có thể chạy nó ở đây (`configs/PFN-Baseline-F0/PFN-Baseline-F0-200ep.json`). Không bao giờ rút ngắn số epoch;
   - `PFN-Baseline-VITS2` chạy trên MardeusPC, không chạy ở máy này.
   Các lệnh vẫn là `launch_train.sh <config> --gpu N`: dry-run trước, rồi `--execute`.
3. **Không evaluate trên tập test** trước khi tác giả khoá protocol (`tools/camera_ready/launch_eval.sh`).

## 7. Bàn giao

- **Cập nhật `docs/camera-ready/runs-Daziel.md`:**
  - lệnh đã chạy;
  - thời gian bắt đầu và kết thúc;
  - các lần bị ngắt;
  - tốc độ;
  - hash checkpoint;
  - các đề xuất cập nhật `PRIORITY.md`.
- **Commit** trên nhánh `camera-ready-20261019-v2`, không có dòng co-author của AI. Không push nếu tác giả chưa yêu cầu.
- **Báo ngắn bằng tiếng Việt,** kết thúc bằng một trong các trạng thái: `READY_FOR_TRAINING_REVIEW`, `TRAINING_RUNNING`, `BLOCKED_MISSING_INPUT`, `GPU_BLOCKED`.
