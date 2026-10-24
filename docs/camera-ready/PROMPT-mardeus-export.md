# Prompt: export three MardeusPC runs to the PC on an external drive

Bạn hỗ trợ phần thí nghiệm camera-ready của bài SOICT 2026, paper 4478, PitchFlowNet, trên **MardeusPC** (WSL, user `dev`, RTX 5070).
- **Ngôn ngữ:** trao đổi với tác giả bằng tiếng Việt; code, tài liệu và commit viết tiếng Anh.
- **Việc cần làm:** chép kết quả evaluation và checkpoint đã chốt của ba run ra ổ cứng ngoài, để PC (2 × RTX 4070 Ti) gom đủ 7 model và chạy `compare`.

## 0. Bối cảnh

**Quyết định của tác giả (23/10):**
- PC gom kết quả của 7 model để đánh giá trước, phục vụ sửa luận văn. 7 model là tám tổ hợp trừ `PFN-Baseline-F0`.
- `PFN-Baseline-F0` đang train trên máy này; nó được thêm vào sau khi train xong.
- Đường chuyển: **ổ cứng ngoài**, không qua mạng, không qua dịch vụ đám mây.
- Gom `eval/<label>/` **không có `wavs/`**, cùng toàn bộ thư mục `final/<run_id>/`.

| Run ID | Label evaluation | Có sẵn trên máy này |
|---|---|---|
| `PFN-Full-275ep-s1234` (dừng ở epoch 200) | `PFN-Full-200ep-best` | `final/`, `eval/` |
| `PFN-NoF0-200ep-s1234` | `PFN-NoF0-200ep-best` | `final/`, `eval/` |
| `PFN-Baseline-VITS2-200ep-s1234` | `PFN-Baseline-VITS2-200ep-best` | `final/`, `eval/` (tác giả xác nhận đã evaluate) |

Bốn run của PC (`PFN-NoBigVGAN`, `PFN-NoVITS2`, `PFN-Baseline`, `PFN-Baseline-BigVGAN`) đã có `final/` và `eval/` trên PC; không cần chép gì từ PC sang đây.

## 1. Quy tắc bắt buộc

- **`PFN-Baseline-F0` đang train.**
  - Không dừng, không attach, không gửi phím, không giảm tài nguyên của nó hay watchdog của nó.
  - Mọi lệnh đọc và ghi lớn chạy với `nice -n19 ionice -c3`.
- **Chỉ đọc nguồn.** Không sửa, không di chuyển, không đổi quyền gì trong `$CR_OUTPUTS/final/` hay `$CR_OUTPUTS/eval/`.
- **Chỉ ghi vào ổ ngoài**, trong một thư mục mới. Không ghi đè thư mục đã có.
- **Không gửi dữ liệu đi đâu khác**, không commit dữ liệu, checkpoint hay WAV vào Git.
- **Không unpickle checkpoint.** Bước này chỉ chép file và tính hash.
- **Không train, không evaluate lại, không chạy `compare`.** `compare` của 7 model chạy trên PC.
- **Commit:** theo mục **Commits** của `AGENTS.md`, không có attribution AI. Không push khi tác giả chưa yêu cầu.

## 2. Kiểm trước khi chép (chỉ đọc)

```sh
O=~/outputs   # $CR_OUTPUTS on MardeusPC
for r in PFN-Full-275ep-s1234 PFN-NoF0-200ep-s1234 PFN-Baseline-VITS2-200ep-s1234; do
  echo "== $r"; (cd $O/final/$r && nice -n19 ionice -c3 sha256sum -c SHA256SUMS && ls best-epoch=*.ckpt last-epoch=199.ckpt final.json)
done
for l in PFN-Full-200ep-best PFN-NoF0-200ep-best PFN-Baseline-VITS2-200ep-best; do
  echo "== $l"; ls $O/eval/$l/summary.json $O/eval/$l/protocol.json
done
ls $O/eval/compare-*.json
```

Với từng label, kiểm:
- `summary.json`: `num_sentences` 500, `mcd.n` 500, `mcd.n_skipped` 0;
- `protocol.json`: `split` là `test`, và `checkpoint_sha256` trùng dòng `best-epoch=…ckpt` trong `final/<run_id>/SHA256SUMS`;
- `mcd.mean_gt_utmos` ≈ 3,831.

**Thiếu thư mục hay file nào, tên khác bảng trên, hoặc một kiểm tra không đạt thì dừng và báo `BLOCKED_MISSING_INPUT`.** Đừng tự evaluate hay tự chốt thay.

## 3. Chép ra ổ ngoài

1. **Ổ ngoài.** Tác giả cắm ổ; nó hiện trong WSL dưới dạng `/mnt/<x>`.
   - Kiểm chỗ trống bằng `df -h /mnt/<x>`. Cần ít nhất **7 GB**: khoảng 5,3 GB checkpoint, cộng phần dự phòng.
   - Ngày 19/10, cổng USB trên máy này từng reset nhiều lần khi đọc ổ (`UASPStor`, `data-sync.md`). Vì vậy bắt buộc kiểm hash sau khi chép (bước 4).
2. **Tạo archive.** Ghi vào một thư mục mới; `mkdir` sẽ báo lỗi nếu thư mục đã có.
   ```sh
   D=/mnt/<x>/pfn-transfer-20261024 && mkdir "$D" && cd ~/outputs
   for l in PFN-Full-200ep-best PFN-NoF0-200ep-best PFN-Baseline-VITS2-200ep-best; do
     nice -n19 ionice -c3 tar --exclude="$l/wavs" -cf "$D/eval-$l.tar" -C eval "$l"; done
   nice -n19 ionice -c3 tar -cf "$D/eval-compare.tar" -C eval $(cd eval && ls compare-*.json)
   for r in PFN-Full-275ep-s1234 PFN-NoF0-200ep-s1234 PFN-Baseline-VITS2-200ep-s1234; do
     nice -n19 ionice -c3 tar -cf "$D/final-$r.tar" -C final "$r"; done
   (cd "$D" && nice -n19 ionice -c3 sha256sum *.tar > SHA256SUMS.source)
   ```
3. **Kiểm `wavs/` không có trong archive.** Lệnh dưới phải in `0` cho từng file:
   ```sh
   for f in "$D"/eval-PFN-*.tar; do echo "$f: $(tar -tf "$f" | grep -c /wavs/)"; done
   ```
4. **Kiểm lại hash từ ổ.**
   - Tác giả rút ổ an toàn trong Windows rồi cắm lại, để lần đọc này đọc thật từ ổ chứ không đọc từ cache.
   - Sau đó chạy:
     ```sh
     cd /mnt/<x>/pfn-transfer-20261024 && nice -n19 ionice -c3 sha256sum -c SHA256SUMS.source
     ```
   - Tất cả phải `OK`. Có file lỗi thì tạo lại riêng file đó và kiểm lại; không sửa hash bằng tay.

## 4. Ghi chép và bàn giao

- **Ghi vào `docs/camera-ready/runs-MardeusPC.md`**, một mục mới "Export to the PC":
  - giờ chép;
  - các file tar và SHA-256 của chúng (nội dung `SHA256SUMS.source`);
  - kết quả các kiểm tra ở mục 2 và 3.
- **Cập nhật bảng "Final checkpoints of every run"** cho `PFN-Baseline-VITS2-200ep-s1234`: best, last, SHA-256. Hiện bảng còn ghi "pending (training)".
- **Commit** trên nhánh làm việc của máy này, đúng định dạng của `AGENTS.md`. Không push khi chưa được yêu cầu.
- **Báo tác giả ngắn bằng tiếng Việt:**
  - đường dẫn thư mục trên ổ;
  - tổng dung lượng;
  - nội dung `SHA256SUMS.source`.

  Kết thúc bằng một trong các trạng thái: `READY_FOR_TRANSFER`, `BLOCKED_MISSING_INPUT`.

## 5. Phía PC, để tham khảo (không làm trên máy này)

Khi nhận ổ, PC sẽ:
1. chép các file tar vào WSL và kiểm `SHA256SUMS.source`;
2. giải nén vào `$CR_OUTPUTS/eval/<label>/` và `$CR_OUTPUTS/final/<run_id>/` của PC, rồi chạy `sha256sum -c` trên `SHA256SUMS` trong mỗi `final/<run_id>/`;
3. chuyển các file nhận được sang chỉ đọc, và ghi nguồn gốc vào `runs-Daziel.md`;
4. lập bảng 7 model và chạy `evaluate_like_history.py compare`:
   - leave-one-out: `PFN-Full-200ep-best` so với `PFN-NoF0-200ep-best`, `PFN-NoBigVGAN-200ep-best`, `PFN-NoVITS2-200ep-best`;
   - add-one: `PFN-Baseline-200ep-best` so với `PFN-Baseline-BigVGAN-200ep-best`, `PFN-Baseline-VITS2-200ep-best`.
