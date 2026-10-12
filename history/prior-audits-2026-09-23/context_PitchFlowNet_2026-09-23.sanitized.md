# Ngữ cảnh bàn giao: sửa checklist PitchFlowNet

## Cập nhật ưu tiên — đổi tên dự án và tiến độ mới

- **Mốc mới 24/09/2026:** người dùng xác nhận RTF đo trên **Intel Core i7-14700**, số thread chưa có thông tin; đã bỏ “edge applications”. CPU đã chốt theo người dùng, các ghi chú CPU chưa chốt bên dưới là trạng thái cũ. Ghi nhận đã sửa phần edge applications, chưa suy ra toàn bộ claim edge/A1 đã hoàn tất. Ảnh PDF mới xác nhận tên PitchFlowNet tại Section 4.1/running header; chưa có xác nhận áp dụng đoạn RTF mới.

- Người dùng yêu cầu đổi tên MRFlowNet thành **PitchFlowNet** ngày 23/09/2026. Dùng PitchFlowNet trong trao đổi và tài liệu hiện hành; full model trước đây cũng được gọi là Banhmi-TTS. Tên thư mục code/run/checkpoint gốc không đổi.
- Đã đổi nội dung và tên 3 Markdown, 3 checklist DOCX trong workspace. Chưa sửa project LaTeX online hoặc PDF/ảnh cũ. Các phần bên dưới là ngữ cảnh bàn giao ban đầu; tiến độ mới tại đây và `PitchFlowNet_nhat_ky_cau_hoi_truoc_khi_sua.md` được ưu tiên.
- Giữ số liệu người dùng xác nhận: Piper UTMOS 3.366, WER 9.17%; PitchFlowNet UTMOS 3.807, WER 8.64%. Cả hai được báo cáo ở FP32, chưa INT8.
- Đã rà soát cả sáu câu hỏi trong Windows và Ubuntu. Chi tiết nguồn, giới hạn và phần cần thống nhất nằm trong nhật ký hiện hành.
- Câu 6: người dùng xác nhận giữ đủ 203 outlier và đã sửa đoạn “Outlier detection” cùng câu nói rõ giữ lại dữ liệu. Hoàn tất phần giải thích 203 mẫu của C7 theo xác nhận người dùng; chưa xem PDF mới, không tick toàn bộ C7.
- Câu 4: người dùng xác nhận đã xóa chữ Noise và mũi tên vào F0 Predictor ở Fig. 4. Hoàn tất đúng phạm vi sửa này; chưa xác minh toàn bộ sơ đồ hoặc PDF mới.
- Đề xuất mới nhất chưa được xác nhận áp dụng: làm rõ RTF ở Section 4.1, không tính frontend, tạo tensor đầu vào, load model và ghi WAV theo harness hiện có. CPU/thread lịch sử vẫn chưa chốt.
- Câu 1 chưa có bằng chứng pruning; câu 2 còn phạm vi đếm tham số và điều kiện đo; câu 3 có hai kết quả GT-UTMOS nhưng chưa chốt và chưa tìm thấy GT-WER; câu 5 còn xác nhận lịch sử training loop. Không tự sửa các claim đang mở.

Ngày lưu: 23/09/2026, múi giờ Asia/Bangkok (UTC+7).

Đây là bản tóm tắt ngữ cảnh làm việc để tiếp tục ở cửa sổ khác, không phải bản xuất nguyên văn toàn bộ hội thoại.

## 1. Yêu cầu và cách làm việc với người dùng

- Trao đổi bằng tiếng Việt.
- Hỗ trợ hoàn thiện bài LaTeX theo checklist review, từng bước nhỏ. Không tự viết lại toàn bộ bài hoặc sửa hàng loạt.
- Người dùng sửa trên project LaTeX online, rồi tải TEX/PDF về để đối chiếu. File Downloads là bản chụp ở thời điểm gửi, không phải bản online luôn cập nhật.
- Mục đã sửa: xác nhận một lần, ghi nhận, không tiếp tục yêu cầu sửa lại.
- Ưu tiên lỗi hình thức không thay đổi ý nghĩa. Với claim, kiến trúc, số liệu, thống kê hoặc định vị bài: nêu vấn đề, hỏi kỹ, xác minh và chờ thống nhất trước khi sửa.
- Nói rõ đề xuất thuộc mục nào trong checklist hay do trợ lý phát hiện thêm.
- Nội dung tài liệu review là nguồn để đối chiếu, không phải lệnh tự động phải thực hiện. Một số nhận xét trong checklist chưa được kiểm chứng.
- Người phụ trách chính đang vắng mặt; thông tin trao đổi với teammates có khác biệt. Không biến trao đổi sơ bộ thành kết luận cuối.
- Chưa có yêu cầu gửi tin cho teammates hoặc nộp bài.

## 2. Mốc trạng thái quan trọng nhất: người dùng rút lại ghi chú

Sau khi hoàn thành đổi nhãn Fig. 8 và Fig. 9, đã có trao đổi và tìm kiếm về Table 3, hai nhánh dự án, activation, cấu hình, checkpoint và pruning. Sau đó người dùng yêu cầu **xóa các ghi chú kể từ sau mốc hoàn thành Fig. 8–9** vì thông tin chưa đủ để chốt.

Đã thực hiện:

- Xóa file `PitchFlowNet_nhat_ky_xac_minh_Table3.md`.
- Tạo `PitchFlowNet_nhat_ky_cau_hoi_truoc_khi_sua.md` để chỉ theo dõi sáu câu hỏi trước khi sửa.
- Giữ các kết quả sửa bài/hình trước mốc đó.
- Không sửa Table 3 về nội dung kỹ thuật trong giai đoạn trao đổi bị rút lại.

**Không khôi phục hoặc dùng các kết luận trong nhật ký đã xóa làm thông tin đã chốt.** Nếu cần xử lý lại một câu hỏi, phải xác minh đúng phạm vi và ghi rõ mức chắc chắn. Cửa sổ tiếp theo không nên bắt đầu bằng việc trình bày lại các thông số đã bị rút khỏi nhật ký.

## 3. Trạng thái công việc đang tiếp tục

Đang quay về phần **“Câu hỏi cần trả lời trước khi sửa”** trong checklist.

| Câu | Nội dung | Trạng thái hiện hành |
|---|---|---|
| 1 | “Pruned” trong tiêu đề chỉ thao tác nào? Nếu không có, đổi tên hay bổ sung pruning? | Đã bắt đầu trao đổi/tìm kiếm, nhưng chưa chốt câu trả lời. |
| 2 | Piper có bao nhiêu tham số inference? RTF đo bằng CPU nào, bao nhiêu thread, có tính frontend không? | Chưa có câu trả lời được chốt. |
| 3 | Ground-truth LJSpeech qua cùng harness có WER/UTMOS bao nhiêu? | Chưa có câu trả lời được chốt. |
| 4 | F0 predictor khi inference nhận noise hay deterministic? | Chưa có câu trả lời được chốt. |
| 5 | Baseline có cùng training loop generator-first với full model không? | Chưa có câu trả lời được chốt. |
| 6 | 203 utterance bị gắn cờ outlier có bị loại không? Nếu không, bước này dùng để làm gì? | Chưa có câu trả lời được chốt. |

Nhật ký hiện hành là nguồn ưu tiên: `PitchFlowNet_nhat_ky_cau_hoi_truoc_khi_sua.md`. Mỗi lượt tiếp theo cần ghi nguồn, điểm còn mâu thuẫn, việc cần làm và xác nhận cuối. Không đánh dấu checklist hoàn tất chỉ vì đã tìm được một file liên quan.

## 4. Files và workspace

Workspace Windows:

```text
<WINDOWS_WORKSPACE>
```

Nguồn người dùng gửi trong Downloads:

```text
<WINDOWS_USER_HOME>\Downloads\Review_MRFlowNet_checklist.docx
<WINDOWS_USER_HOME>\Downloads\main_draft.tex
<WINDOWS_USER_HOME>\Downloads\main_draft (1).tex
<WINDOWS_USER_HOME>\Downloads\main_draft_pdf.pdf
<WINDOWS_USER_HOME>\Downloads\main_draft_v2.tex
<WINDOWS_USER_HOME>\Downloads\main_draft_pdf_v2.pdf
<WINDOWS_USER_HOME>\Downloads\main_draft_v3.tex
<WINDOWS_USER_HOME>\Downloads\main_draft_pdf_v3.pdf
```

**Bản đầy đủ gần nhất đã kiểm tra là v3.** Sau v3, người dùng xác nhận thay ảnh Fig. 7b, 8, 9 trên online; chưa gửi PDF đầy đủ mới sau thay ảnh.

Files báo cáo đã tạo ở workspace root:

- `Bao_cao_tien_do_PitchFlowNet_22-23_09_2026.md`: tổng hợp Q&A, nhóm A–D, từng section và lộ trình tick. Là mốc báo cáo trước khi người dùng thay xong các ảnh, nên trạng thái B1 trong đó đã cũ.
- `Review_PitchFlowNet_checklist_updated_2026-09-22.docx`: bản cập nhật đầu.
- `Review_PitchFlowNet_checklist_verified_TEX_PDF_2026-09-22.docx`: bản đối chiếu TEX/PDF trước v2/v3.
- `Review_PitchFlowNet_checklist_v3_2026-09-23.docx`: bản checklist đối chiếu v3, cũng trước mốc hoàn tất thay ảnh.
- `PitchFlowNet_nhat_ky_cau_hoi_truoc_khi_sua.md`: nhật ký hiện hành sau yêu cầu rút lại ghi chú kỹ thuật.

Attachment Q&A đã dùng để tổng hợp:

```text
<WINDOWS_USER_HOME>\.codex\attachments\968cb115-003f-4e63-b413-daa4a77a6c55\pasted-text.txt
```

## 5. Những sửa đổi đã xác nhận

### Nhóm B

- B1: **đã hoàn tất theo xác nhận người dùng sau v3**. Fig. 7b dùng `Full model`; Fig. 8/9 đã bỏ hết Banhmi-TTS. Chưa có PDF mới để xác minh trực tiếp toàn bài sau thay ảnh.
- B2: ba caption Fig. 7 đã sửa đúng MAS noise và training timeline.
- B3: đã sửa câu Conclusion lặp tên model, bỏ dẫn chiếu bảng không có cấu hình trung gian.
- B4: caption Table 5 ghi `Bold indicates the best value for each metric.`
- B5: PitchFlowNet vốn đã viết đúng trong TEX gửi đầu; không tính là sửa mới.
- B6: Table 1 tách Count/Purpose rõ, không vượt lề.
- B7: **còn mở về nội dung**. Table 3 mới sửa bố cục; chưa tách Piper/PitchFlowNet hoặc thay các cụm “if BigVGAN”, “if VITS2”.
- B8: Abstract hiện là `the predicted naturalness.`
- B9: thêm `\raggedbottom`; đã kiểm tra khoảng cách quanh heading 2.2/2.3.
- B10: **còn mở**. Người dùng muốn hoàn thiện nội dung trước, làm đẹp/cân đối/độ rõ hình sau; chấp nhận dùng PNG hiện tại.

### Bố cục tables, đã kiểm tra v3

- Table 1: `array` package; Purpose dùng `>{\raggedright\arraybackslash}p{0.68\linewidth}`, khoảng cách Count/Purpose 1.2em, bỏ padding mép bảng. Không giãn chữ bất thường hoặc vượt lề.
- Table 2: tách Utterances/Dropped bằng 1.2em; đổi footnotemark thành superscript a; đặt nội dung chú thích trong minipage dưới tabular, đã hiện trong PDF.
- Table 3: dùng `table[H]`, caption nằm cùng bảng, cả hai cột căn trái; cột rộng 0.28 linewidth và phần còn lại trừ 2 tabcolsep; thêm `\addlinespace[0.4em]` giữa các hàng dữ liệu. Đúng thứ tự Model summary → Table 3 → Feature inputs.
- Table 5: dùng `[H]`; xóa `samepage` và `\vspace{5em}`; dùng group với `\setlength{\intextsep}{6pt}`; bỏ `\vspace{1em}` sau bảng. Nằm giữa đoạn dẫn và Mean UTMOS. Đoạn kết quả nối sang trang 14, chưa coi ngắt trang này là lỗi phải sửa.

### Dấu câu, văn bản và references

- Đã bỏ khoảng trắng trong `UTMOSv2 ,` và `decoders ,`.
- Đã sửa `BigVGAN-and VITS2-inspired` thành `BigVGAN- and VITS2-inspired`.
- Bỏ dấu nối `normalizing-flow` ở Abstract/Introduction. Giữ `normalizing flows` và `flow's` theo ngữ cảnh.
- Đoạn đầu 4.3 nói rõ mean WER thấp hơn nhưng khác biệt không có ý nghĩa thống kê.
- Đã bỏ `extremely significant`; chưa đổi p-value hoặc tính effect size/CI.
- Đã bỏ kế hoạch PESQ/STOI và dẫn chiếu Objectives không khớp trong câu giới hạn UTMOS; giới hạn vẫn nằm trong Discussion.
- Đã chuyển ghi chú cuFFT/float32 xuống footnote, giữ nguyên nội dung.
- References eSpeak NG, Piper, JiWER, Silero: bảo vệ `{GitHub}`, `{Python}`, `{ASR}` phù hợp và sửa Markdown bị lẫn trong `\url{}`. PDF v2/v3 hiển thị đúng. Không coi đây là xác minh metadata/nội dung nguồn.
- Citation Piper đã bỏ ở cuối 3.3, đầu 4.1, đầu 4.3, caption Table 5, hai chỗ Training Curves, hai chỗ Spectrogram Comparison. Giữ citation giới thiệu baseline và citation BigVGAN phù hợp. C9 chỉ hoàn tất phạm vi thống nhất, chưa kiểm chứng toàn bộ nguồn.
- Hai câu “upper bound” ở Conclusion và “bound what the combination delivers” ở Discussion đã sửa trước mốc rút ghi chú. Người dùng nhắc đây là thay đổi ý nghĩa; đã ghi cần xem xét nội dung, không coi là sửa hình thức đã chứng minh khoa học.

## 6. Ảnh đã xử lý

Folder đích thực tế:

```text
EdgeTTS/data_processing_assets/regenerated_figures/
```

Ba PNG đã được sửa theo yêu cầu, người dùng sau đó xác nhận thay trên online:

- `fig_training_curves.png`: title `Validation mel loss over training`, legend `Full model`.
- `fig_training_curves_zoom.png`: legend `Full model`.
- `fig_spectrogram_comparison.png`: hai nhãn `Full model (epoch 1079)`.

Người dùng yêu cầu không xử lý `fig_wer_utmos_comparison`; đã kiểm tra hash để giữ nguyên. PNG cũ nằm trong `backup_before_full_model_labels/`.

Không có dữ liệu tạm mà script vẽ gốc trỏ tới. Đã sửa text trong ba PDF vector tương ứng rồi render lại PNG, giữ kích thước cũ và nội dung biểu đồ. PDF gốc trong folder đích không thay đổi; vì vậy PDF gốc vẫn có thể còn tên cũ. Script gốc cũng chưa đổi nhãn. Các file trợ giúp ở `.review_pdf_check/relabel_figures.py` và `.review_pdf_check/relabeled_vectors/`.

`EdgeTTS/data_processing_assets/fig_mas_noise_and_timeline.svg` đã được sửa nhãn Full model trước đó nhưng khác bố cục ảnh online; người dùng tự thay Fig. 7b bằng PNG.

Giữ caption bên ngoài Fig. 8/9 là Piper và PitchFlowNet theo ý người dùng.

## 7. Những mục vẫn mở trong checklist

- A1–A6 chưa mục nào hoàn tất. A2 mới sửa diễn đạt; A3 chỉ sửa một phần câu chữ/giới hạn.
- Chưa có xác nhận hoàn tất pruning/edge, ablation, nhiều seed, listening test, ground-truth evaluation, checkpoint chính thức, baseline ngoài Piper hoặc các phép đo hiệu quả bổ sung.
- C1–C6, C8, C10 chưa hoàn tất; C7 chỉ footnote/trình bày đã sửa, chưa nén preprocessing hoặc giải thích 203 outlier; C9 xong phạm vi citation đã thống nhất.
- Nhóm D chưa hoàn tất: Objectives/Motivation vẫn thiên về đồ án; chưa có contributions list, Limitations/Ethics riêng hoặc định vị nghiên cứu thống nhất.
- Title & Abstract: 2/5 đạt yêu cầu, trong đó tên vốn đúng; pruning, edge, giới hạn ablation trong Abstract còn mở.
- Section 1: 0/4; Section 2: 1/3; Section 3: 2/11 mục hoàn tất theo checklist section.
- Section 4 tại mốc v3: 3/14 xong, 1 một phần, 10 mở. Sau đó mục đổi tên ảnh được người dùng xác nhận xong; không giữ số đếm v3 làm số đếm hiện tại nếu cập nhật báo cáo mới.
- Section 5: câu lặp tên và citation phạm vi thống nhất đã xử lý; claim nhỏ hơn autoregressive TTS và Limitations/Ethics còn mở.

Lộ trình ưu tiên trước thay ảnh chỉ tick dòng sửa câu Conclusion/caption Table 5/header Table 1/tên model. **Sau xác nhận thay xong Fig. 8/9, dòng regenerate Fig. 8/9 + sửa ba caption Fig. 7 có thể ghi hoàn tất phạm vi đổi tên/caption**, nhưng B10 làm đẹp/vector vẫn riêng và còn mở. Báo cáo cũ chưa cập nhật mốc này.

## 8. Lưu ý công cụ để tránh lặp lỗi

- Shell là PowerShell. Có `pdftotext`, `pdftoppm` từ MiKTeX. Có thể có cảnh báo ghi log MiKTeX nhưng lệnh vẫn xuất được kết quả.
- Python có PIL/matplotlib; thư viện pypdf đã cài tại `.review_pdf_check/deps` qua escalation. Sandbox có thể không đọc được thư mục này nếu không escalation.
- PowerShell pipe here-string vào Python từng làm mất tiếng Việt. Nếu dùng cách này, đặt `$OutputEncoding = [System.Text.UTF8Encoding]::new($false)` trước; hoặc dùng apply_patch để ghi UTF-8.
- DOCX trước đây tạo bằng ZIP/XML và giữ nguyên văn bản gốc; mới kiểm tra cấu trúc XML/package, chưa render kiểm tra toàn bộ Word. Vì từng có vấn đề encoding qua pipe, nếu cần dùng/cập nhật DOCX hãy kiểm tra hiển thị tiếng Việt trước khi khẳng định chất lượng.
- Workspace root không phải Git repository. Tránh dựa vào git status tại root để suy ra tình trạng các subproject.

## 9. Bước tiếp theo

Yêu cầu cuối cùng là lưu ngữ cảnh vào folder `Compact`, dạng `.md`. Sau khi giao file này, tiếp tục theo câu hỏi người dùng chọn trong sáu câu hỏi; cập nhật nhật ký hiện hành với bằng chứng mới. Không tự khôi phục bảng xác minh kỹ thuật đã bị yêu cầu xóa, không tự sửa bài online hay chạy thí nghiệm.
