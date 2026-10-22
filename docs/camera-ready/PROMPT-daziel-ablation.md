# Prompt: train four ablation runs on the PC (2 × RTX 4070 Ti)

Bạn hỗ trợ phần thí nghiệm camera-ready của bài SOICT 2026, paper 4478, PitchFlowNet.
Prompt này dành cho máy PC của tác giả: Windows có WSL `Ubuntu-24.04`, user `dev`, 2 × RTX 4070 Ti (12 GB mỗi card).
- **Ngôn ngữ:** trao đổi với tác giả bằng tiếng Việt; code, tài liệu và commit viết tiếng Anh.
- **Mốc thời gian:** thí nghiệm xong muộn nhất tối 27/10; ngày 28–29/10 dành cho viết bài.

## 0. Bối cảnh, đọc trước, chưa chạy gì

**Ba review đều nêu cùng một điểm yếu chính: chỉ có baseline và full, thiếu ablation.**
- Tác giả quyết định làm ablation đủ **2³ = 8 tổ hợp** của ba nhóm BigVGAN (B), VITS2 (V), F0 (F), chia cho hai máy (`docs/camera-ready/experiment-plan.md`).
- **Cặp lịch sử (Baseline 773, Full 1079) vẫn là kết quả chính của bài.** Không train lại, không warm-start từ chúng.

**Công thức chung cho cả 8 run** (đã chốt trước mọi kết quả test):
- **200 epoch** = 200 × 782 batch = **156.400 update G**, batch 16, 1 GPU mỗi run, không DDP, bf16-mixed;
- seed 1234; `train_shuffle=false`; MAS decay `4e-6` mỗi update G;
- checkpoint để evaluate: `val_mel_l1_global` nhỏ nhất. **Không bao giờ dùng tập test để chọn checkpoint.**

| Run | Máy / GPU | Trạng thái lúc viết prompt (22/10 00:40) |
|---|---|---|
| `PFN-Full` (mốc so; run `PFN-Full-275ep-s1234` dừng ở epoch 200) | MardeusPC, RTX 5070 | **xong** 21/10 10:08, best epoch 188 |
| `PFN-NoF0-200ep` → `PFN-Baseline-VITS2-200ep` → `PFN-Baseline-F0-200ep` | MardeusPC, RTX 5070 | NoF0 đang train |
| **`PFN-NoBigVGAN-200ep`** → **`PFN-Baseline-BigVGAN-200ep`** | **PC này, GPU 0** | **việc của bạn** |
| **`PFN-NoVITS2-200ep`** → **`PFN-Baseline-200ep`** | **PC này, GPU 1** | **việc của bạn** |

Trên RTX 5070, một epoch mất khoảng 8,3 phút, tức mỗi run khoảng 28 giờ. Tốc độ của RTX 4070 Ti sẽ đo ở epoch đầu.

**Đọc theo thứ tự** (sau bước 2, tức là sau khi đã pull `main`):
1. `AGENTS.md`, nhất là mục **Commits**;
2. `docs/camera-ready/README.md`: bố cục repo;
3. `docs/camera-ready/experiment-plan.md`;
4. `docs/camera-ready/runbook-MardeusPC.md`: cùng bộ công cụ, khác máy;
5. `docs/camera-ready/runs-MardeusPC.md`: các sự cố đã gặp (lỗi CUDA, cách resume);
6. `docs/camera-ready/eval-protocol.md`.

## 1. Quy tắc bắt buộc

**Dự án khác trên máy này:**
- Máy đang hoặc vừa chạy Piper-SeqIR, với các tmux session `piper-ir`, `post-seqir`, `tb-seqir`.
- **Không attach, không gửi phím, không kill, không tạm dừng, không giảm tài nguyên** của các session và tiến trình đó.
- Không sửa env `base`, `edgetts` hay bất kỳ env nào khác.
- Chỉ kiểm bằng lệnh chỉ đọc (`nvidia-smi`, `ps`, `tmux ls`).
- Nếu GPU còn bận vì SeqIR hoặc phiên evaluation/timing tự động của nó, báo **GPU_BLOCKED** và dừng.

**Dữ liệu:**
- Dữ liệu gốc: `~/data_preprocessed`, `~/data/wavs`, `~/test_entries_500.jsonl`. Đây là thư mục home của `dev`, tức `<WSL_DEV_HOME>`.
- Chỉ đọc, không sửa, không chạy preprocess lại.

**Thư mục riêng cho camera-ready:**

| Mục đích | Đường dẫn |
|---|---|
| Repo | `~/Pitch-FlowNet-cr` |
| Gốc mọi kết quả (`$CR_OUTPUTS`) | `~/outputs-pfn-cr` |
| Env train | `pfn-cr` |
| Env evaluation | `pfn-eval` |
| TensorBoard | cổng **6007** (cổng 6006 có thể đang dùng cho `tb-seqir`) |

**Cấu trúc thư mục kết quả phải giống hệt MardeusPC.** Xem mục **Output layout** trong `runbook-MardeusPC.md`.

| Thư mục | Nội dung |
|---|---|
| `runs/<run_id>/` | thư mục làm việc của run |
| `final/<run_id>/` | bản chốt chỉ đọc |
| `eval/<label>/` | evaluation trên test, label = `<variant>-200ep-best` |
| `logs/` | mọi log |
| `locks/` | khoá GPU |
| `tests/` | test, chạy thử, chẩn đoán, việc bị dừng giữa chừng, tên dạng `<what>-<YYYYMMDD>/` |

- **Không tạo thư mục nào khác** dưới `$CR_OUTPUTS`. Không để kết quả ở chỗ khác. Không chép kết quả vào repo.
- Evaluation chạy thử luôn vào `tests/`, **không bao giờ vào `eval/`**.
- Không gửi dữ liệu đi đâu. MardeusPC sẽ tự lấy `eval/<label>/` (không gồm `wavs/`) qua Tailscale để so sánh.

**Tập test:** không dùng để tuning, early-stop hay chọn checkpoint. Không evaluate trên test trừ khi tác giả yêu cầu, và khi đó theo đúng `docs/camera-ready/eval-protocol.md`.

**Bằng chứng:**
- Không bịa kết quả.
- Không dùng `weights_only=False`.
- UTMOSv2 là điểm dự đoán, không phải MOS.

**Commit:**
- Theo mục **Commits** của `AGENTS.md`: Git identity của tác giả; **không có attribution AI** (không `Co-Authored-By`, không "Generated with"); subject dạng `type(scope): summary`, tối đa 72 ký tự.
- **Không push** khi tác giả chưa yêu cầu.

**Repo:**
- Máy này không có các file gốc chứa đường dẫn máy, nên **không chạy `tools/sanitize_for_git.py`**. Trước mỗi commit chỉ chạy `tools/check_tracked_package.py`.
- Chỉ commit các file của riêng máy này: `docs/camera-ready/preflight-Daziel.md`, `docs/camera-ready/runs-Daziel.md`.
- **Đừng sửa** `PRIORITY.md` và `experiment-plan.md` (MardeusPC cập nhật hai file này); ghi đề xuất cập nhật vào `runs-Daziel.md`.

**Cách làm việc:**
- Sửa lỗi tận gốc.
- **Mỗi lệnh `--execute` chỉ chạy sau khi tác giả xác nhận đúng lệnh đó.**
- Không bao giờ chạy hai bản của cùng một run, và không train lại một run từ đầu.

## 2. Lấy `main` và kiểm tra kỹ (làm trước mọi việc khác)

MardeusPC đã đưa toàn bộ phần chuẩn bị lên `main` qua các PR #3–#7. Bạn phải lấy **đủ** và kiểm từng mục dưới đây.

1. **Lấy repo:**
   - Nếu `~/Pitch-FlowNet-cr` chưa có:
     ```sh
     git clone https://github.com/DazielNguyen/Pitch-FlowNet.git ~/Pitch-FlowNet-cr
     ```
   - Nếu đã có: vào đó, kiểm `git status --short` phải rỗng. Có thay đổi chưa commit thì **dừng và báo**, không tự xoá. Rồi chạy:
     ```sh
     git fetch origin && git checkout main && git pull --ff-only origin main
     ```
2. **Kiểm commit và cây file:**
   - `git rev-parse HEAD` phải bằng `git rev-parse origin/main`.
   - Các commit sau phải nằm trong lịch sử. Kiểm từng cái bằng `git merge-base --is-ancestor <commit> HEAD`:
     - `3ac8756` (8 config 200 epoch);
     - `65b5cb3` (`finalize_run.sh`);
     - `f773be2` (công cụ evaluation);
     - `33dca31` (protocol evaluation);
     - commit sửa `auto_resume.sh` để truyền biến `CR_*` vào tmux. Tìm bằng `git log --oneline -- tools/camera_ready/auto_resume.sh`; subject chứa "pass the CR_ settings".
   - `git status --short` rỗng.
3. **Kiểm các file bắt buộc** (thiếu file nào thì báo **BLOCKED_MISSING_INPUT** và dừng):
   - `configs/PFN-NoBigVGAN/PFN-NoBigVGAN-200ep.json`, `configs/PFN-NoVITS2/PFN-NoVITS2-200ep.json`, `configs/PFN-Baseline-BigVGAN/PFN-Baseline-BigVGAN-200ep.json`, `configs/PFN-Baseline/PFN-Baseline-200ep.json`;
   - `tools/camera_ready/`: `launch_train.sh`, `resume_train.sh`, `auto_resume.sh`, `finalize_run.sh`, `check_ckpt.py`, `tensorboard.sh`, `wsl_keepalive.ps1`;
   - `docs/camera-ready/`: `experiment-plan.md`, `runbook-MardeusPC.md`, `eval-protocol.md`;
   - `manifests/camera-ready/split-manifest.json`, `manifests/camera-ready/data-manifest.json`.
4. **Kiểm nội dung 4 config của máy này.** Đọc JSON và kiểm `protocol.max_g_updates == 156400`, `trainer.devices == 1`, `trainer.strategy == "auto"`, `model.batch_size == 16`. Kiểm cờ nhóm:

   | Config | `use_bigvgan` | `use_vits2` | `use_f0` |
   |---|---|---|---|
   | NoBigVGAN | false | true | true |
   | NoVITS2 | true | false | true |
   | Baseline-BigVGAN | true | false | false |
   | Baseline | false | false | false |

5. **Bật hook commit:** `git config core.hooksPath tools/hooks`, rồi kiểm bằng `git config --get core.hooksPath`.
6. **Tạo nhánh làm việc cho các commit của máy này:** `git switch -c PFN-Daziel-22102026`. Không push.
7. **Ghi kết quả bước 2** vào đầu `docs/camera-ready/preflight-Daziel.md`: HEAD, các commit đã kiểm, danh sách file đã kiểm.

## 3. Preflight, chỉ đọc (ghi tiếp vào `preflight-Daziel.md`)

1. **GPU rảnh:**
   - `nvidia-smi`: GPU 0 và GPU 1 mỗi cái dùng dưới 3.000 MiB, tải dưới 10%.
   - `ps -eo pid,etime,cmd | grep -iE "seqir|piper|train" | grep -v grep`.
   - `tmux ls`.
   - Chưa rảnh thì báo **GPU_BLOCKED** và dừng.
2. **Chỗ trống:**
   - `df -h ~`.
   - Ổ C của Windows, nơi chứa file `.vhdx` của WSL, có khoảng 50 GB trống vào 19/10. Mỗi run cần khoảng 3 GB trong run dir, thêm khoảng 1,75 GB cho bản lưu cuối.
   - Dưới 20 GB trống thì báo tác giả.
3. **Dữ liệu:** chạy với `nice -n19 ionice -c3`.
   - SHA-256 của `dataset.jsonl`, `config.json`, `test_entries_500.jsonl` phải khớp `manifests/camera-ready/data-manifest.json`. Ngày 19/10 cả ba đều đã khớp.
   - Cache phải đủ **39.300 file**.
4. **Hệ thống:** RAM, CPU, phiên bản driver, model GPU và UUID.

## 4. Môi trường riêng

1. **Env `pfn-cr`:**
   - Dùng conda có sẵn để tạo env mới: Python 3.10.20, `torch==2.12.1` (index cu130), `pytorch-lightning==2.6.5`, `numpy==2.2.6`, cùng các gói còn lại của `code/camera-ready-source/src/python/requirements.txt`.
   - Kiểm `torch.cuda.device_count() == 2`. Ghi phiên bản thật vào preflight.
2. **Extension MAS:**
   ```sh
   cd ~/Pitch-FlowNet-cr/code/camera-ready-source/src/python/piper_train/vits/monotonic_align
   mkdir -p monotonic_align && cythonize -i core.pyx && mv core*.so monotonic_align/
   ```
3. **File `~/pfn-cr.env`:** `source` nó trước mọi lệnh. Các script tự truyền các biến này vào tmux.
   ```sh
   export CR_OUTPUTS=$HOME/outputs-pfn-cr
   export CR_PYTHON=<đường dẫn python của env pfn-cr>
   export CR_TENSORBOARD=<đường dẫn tensorboard của env pfn-cr>
   export CR_TB_PORT=6007
   export CR_EVAL_PYTHON=<đường dẫn python của env pfn-eval>
   ```
4. **Test:**
   ```sh
   cd ~/Pitch-FlowNet-cr && $CR_PYTHON -m unittest discover -s tests/camera_ready -v
   ```
   - Lưu log vào `$CR_OUTPUTS/tests/`.
   - **Kết quả mong đợi: 24/25 pass.** Chỉ có `ValidationDDP` báo ERROR (PLAN O2), và nó không cần cho run 1 GPU.
   - Kết quả khác thì báo nguyên nhân và dừng.
5. **Env evaluation `pfn-eval`** (dùng ở mục 7). Phiên bản phải đúng như evaluation lịch sử, xem `eval-protocol.md`:
   ```sh
   conda create -y -n pfn-eval --clone pfn-cr
   echo "torch==2.12.1" > /tmp/pfn-eval-constraints.txt
   <python của pfn-eval> -m pip install -c /tmp/pfn-eval-constraints.txt "openai-whisper==20250625" "jiwer==4.0.0" \
     "transformers==5.12.1" "timm==1.0.27" "scipy==1.15.3" "librosa==0.11.0" \
     "git+https://github.com/sarulab-speech/UTMOSv2.git@cc2700db57bb83ee13dc31ebe1b868c254e15d09"
   <python của pfn-eval> -m pip install --no-deps "torchaudio==2.11.0"
   ```
   - **Kiểm:** `import whisper, jiwer, utmosv2` chạy được, và torch vẫn là 2.12.1. Ghi phiên bản thật vào preflight.
   - **Chạy thử trên 5 câu validation**, dùng bất kỳ checkpoint camera-ready nào có trên máy. Ví dụ, sau khi run đầu tiên lưu epoch 0, dùng `last.ckpt` của nó:
     ```sh
     $CR_EVAL_PYTHON tools/camera_ready/evaluate_like_history.py evaluate --checkpoint <ckpt> --label smoke \
       --output $CR_OUTPUTS/tests/eval-smoke-<YYYYMMDD> --split validation --limit 5 --threads 4
     ```
     Kết quả chạy thử chỉ để kiểm môi trường, không phải kết quả, và phải nằm trong `tests/`.
6. **Phía Windows, tác giả làm:**
   - tắt sleep và tắt tự tắt màn hình khi cắm điện:
     ```
     powercfg /change standby-timeout-ac 0
     powercfg /change monitor-timeout-ac 0
     ```
   - tạm dừng Windows Update;
   - giữ WSL chạy:
     ```
     powershell -ExecutionPolicy Bypass -File <repo>\tools\camera_ready\wsl_keepalive.ps1 start -Distro Ubuntu-24.04
     ```
     Cần một bản repo trên Windows, hoặc gọi file trong `~/Pitch-FlowNet-cr` của user `dev` qua đường dẫn mạng `\\wsl.localhost\Ubuntu-24.04`.

## 5. Launch hai run đầu (chỉ sau khi tác giả xác nhận lệnh)

```sh
source ~/pfn-cr.env && cd ~/Pitch-FlowNet-cr
bash tools/camera_ready/tensorboard.sh start                                                    # http://localhost:6007
bash tools/camera_ready/launch_train.sh configs/PFN-NoBigVGAN/PFN-NoBigVGAN-200ep.json --gpu 0   # dry run
bash tools/camera_ready/launch_train.sh configs/PFN-NoVITS2/PFN-NoVITS2-200ep.json --gpu 1       # dry run
# after the author's OK:
bash tools/camera_ready/launch_train.sh configs/PFN-NoBigVGAN/PFN-NoBigVGAN-200ep.json --gpu 0 --execute
bash tools/camera_ready/auto_resume.sh configs/PFN-NoBigVGAN/PFN-NoBigVGAN-200ep.json --start --gpu 0
bash tools/camera_ready/launch_train.sh configs/PFN-NoVITS2/PFN-NoVITS2-200ep.json --gpu 1 --execute
bash tools/camera_ready/auto_resume.sh configs/PFN-NoVITS2/PFN-NoVITS2-200ep.json --start --gpu 1
```

**Launcher (`--gpu N`):**
- Đặt `CUDA_DEVICE_ORDER=PCI_BUS_ID` và `CUDA_VISIBLE_DEVICES=N`, nên số GPU khớp với `nvidia-smi`.
- Giữ khoá `$CR_OUTPUTS/locks/gpu<N>.lock`, mỗi GPU chỉ chạy một run.
- Chặn khi GPU đó đang bận.

**Watchdog (`auto_resume.sh`):**
- Chỉ tự resume sau lỗi `CUDA error: unknown error`, tối đa 5 lần mỗi run.
- Khi run kết thúc với exit 0, nó tự chạy `finalize_run.sh` rồi dừng.
- Xem `runbook-MardeusPC.md`.

**Sau khi launch:**
- Sau 2–3 phút, kiểm cả hai run đang train:
  - log;
  - `nvidia-smi`: mỗi GPU một tiến trình;
  - `run_manifest.json` của từng run: commit đúng, `dirty_paths: []`, `max_g_updates: 156400`;
  - `source.tree_sha256` phải bằng `1ee13cb4efe39da5657398021f3b85eb3a2a3eb5b89a1f1392dc33c4dfe90a52`, tức cùng code train với run Full trên MardeusPC.
- **Hết epoch đầu:** ghi tốc độ (phút mỗi epoch), VRAM đỉnh và thời gian xong dự kiến vào `runs-Daziel.md`.
- **Nếu OOM:** không đổi batch hay config. Báo tác giả kèm log.

## 6. Theo dõi, khôi phục, chốt

- **Theo dõi** chỉ bằng thao tác đọc:
  - `tmux attach -t cr-<run_id>` rồi `Ctrl-b d`; **không bao giờ `Ctrl-c`**;
  - `checkpoint_ledger.jsonl`, `val_diagnostics.jsonl`;
  - log watchdog `$CR_OUTPUTS/logs/watchdog-<run_id>.log`;
  - TensorBoard.
- **Sau khi máy tắt hoặc WSL khởi động lại** (watchdog cũng mất theo):
  ```sh
  source ~/pfn-cr.env && cd ~/Pitch-FlowNet-cr
  bash tools/camera_ready/tensorboard.sh start
  bash tools/camera_ready/resume_train.sh configs/PFN-NoBigVGAN/PFN-NoBigVGAN-200ep.json --gpu 0   # dry run, checks last.ckpt
  # after the author's OK: the same with --execute, then
  bash tools/camera_ready/auto_resume.sh configs/PFN-NoBigVGAN/PFN-NoBigVGAN-200ep.json --start --gpu 0
  ```
  - Làm tương tự cho run trên GPU 1.
  - Nếu `last.ckpt` hỏng, thêm `--fallback`.
  - Ghi mỗi lần bị ngắt vào `runs-Daziel.md`.
- **Khi một run xong:**
  - kiểm `$CR_OUTPUTS/final/<run_id>/`: `best-epoch=…ckpt`, `last-epoch=199.ckpt`, `SHA256SUMS`, `final.json`, tất cả chỉ đọc;
  - nếu watchdog không chạy, chốt bằng tay: `bash tools/camera_ready/finalize_run.sh <config> --expect-epochs 200`;
  - ghi SHA-256, epoch best và `val_mel_l1_global` best vào `runs-Daziel.md`.

## 7. Evaluate hai run đầu, TRƯỚC khi train hai run tiếp theo

**Đây là bước bắt buộc.** Khi `PFN-NoBigVGAN-200ep-s1234` và `PFN-NoVITS2-200ep-s1234` đều đã chốt (có `final/<run_id>/final.json`), evaluate cả hai trước khi launch bất cứ run nào khác.

**Protocol:** đúng `docs/camera-ready/eval-protocol.md`, đã khoá ngày 21/10.
- 500 câu test; tổng hợp trên CPU;
- WER bằng Whisper `base.en`; UTMOSv2 (điểm dự đoán, không phải MOS);
- F0 lấy từ bộ dự đoán F0; MCD; UTMOS của audio gốc;
- seed riêng cho từng câu.

**Không đổi tham số nào.**

```sh
source ~/pfn-cr.env && cd ~/Pitch-FlowNet-cr
B0=$(ls $CR_OUTPUTS/final/PFN-NoBigVGAN-200ep-s1234/best-epoch=*.ckpt)
B1=$(ls $CR_OUTPUTS/final/PFN-NoVITS2-200ep-s1234/best-epoch=*.ckpt)
bash tools/camera_ready/launch_eval.sh --split test --threads 4 "PFN-NoBigVGAN-200ep-best=$B0" "PFN-NoVITS2-200ep-best=$B1"            # dry run
bash tools/camera_ready/launch_eval.sh --split test --threads 4 --execute "PFN-NoBigVGAN-200ep-best=$B0" "PFN-NoVITS2-200ep-best=$B1"  # after the author's OK
```

**Cách chạy:**
- Mỗi model chạy trong tmux riêng `cr-eval-<label>`, CPU 4 thread, mức ưu tiên thấp. Khoảng 35–75 phút mỗi model.
- GPU không cần cho bước này.
- **Label phải đúng tên trên.** Kết quả nằm trong `$CR_OUTPUTS/eval/<label>/`.

**Kiểm khi xong**, với từng label:
1. `summary.json` có `num_sentences: 500`, và `mcd.n: 500`, `n_skipped: 0`.
2. `protocol.json` có:
   - `checkpoint_sha256` trùng SHA-256 của file best trong `final/<run_id>/SHA256SUMS`;
   - `split: test`;
   - các phiên bản gói đúng `eval-protocol.md`.
3. UTMOS của audio gốc (`mcd.mean_gt_utmos`) phải ra **≈ 3,831**. MardeusPC đo được 3,8308; số trong bài là 3,830. Lệch rõ là dấu hiệu môi trường sai. Khi đó **dừng và báo**, đừng train tiếp.
4. Ghi vào `runs-Daziel.md`: WER, UTMOSv2, MCD, F0, giờ bắt đầu/kết thúc, SHA-256 của checkpoint.
   - **RTF đo trong lúc chạy không dùng để báo cáo được** nếu máy đang chạy việc khác.
   - **Không tự so sánh và không kết luận.** Chỉ ghi số. MardeusPC sẽ gom kết quả và chạy `compare`.

Báo tác giả. Chỉ khi evaluation đã xong và qua đủ các kiểm tra trên, mới sang mục 8.

## 8. Hai run tiếp theo (mỗi lệnh cần tác giả xác nhận)

Khi run trên GPU đó đã chốt **và đã evaluate xong (mục 7)**:
- **GPU 0:** `configs/PFN-Baseline-BigVGAN/PFN-Baseline-BigVGAN-200ep.json` (thêm-một: chỉ BigVGAN, đúng câu hỏi của Review 1).
- **GPU 1:** `configs/PFN-Baseline/PFN-Baseline-200ep.json` (mốc so cho nhóm thêm-một; baseline strict mới, **không phải** baseline lịch sử 773).

Lệnh giống mục 5: dry-run, rồi `--execute`, rồi `auto_resume.sh --start --gpu N`. Không bao giờ rút ngắn số epoch.

**Khi hai run này xong:** evaluate đúng như mục 7, với label `PFN-Baseline-BigVGAN-200ep-best` và `PFN-Baseline-200ep-best`.

## 9. Bàn giao

- **`docs/camera-ready/runs-Daziel.md`** ghi:
  - lệnh đã chạy;
  - thời gian bắt đầu và kết thúc;
  - các lần bị ngắt và tự resume;
  - tốc độ;
  - hash của checkpoint cuối;
  - kết quả evaluation của từng label;
  - đề xuất cập nhật `PRIORITY.md`.
- **Commit** trên `PFN-Daziel-22102026`, đúng định dạng, không attribution AI. Không push nếu chưa được yêu cầu.
- **Báo ngắn bằng tiếng Việt,** kết thúc bằng một trong các trạng thái: `READY_FOR_TRAINING_REVIEW`, `TRAINING_RUNNING`, `BLOCKED_MISSING_INPUT`, `GPU_BLOCKED`.
