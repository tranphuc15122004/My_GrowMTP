# Metrics so sánh GrowMTP và Policy-Shift-Aware GrowMTP

Policy-Shift-Aware GrowMTP tái sử dụng DCA, VGM, replay, optimizer và rollout.
Mỗi `comparison_probe_frequency` actor update (mặc định 4), head tạm hoãn học
teacher cũ trong các PPO mini-batch. Sau PPO, selector tính
`S_r = max(advantage_r, 0) * D_r`, với `D_r` là KL(new || old) trung bình trên
**tất cả vị trí draft của tất cả cycle đã ghi nhận của trajectory**. KL dùng
toàn vocabulary, logits thô, temperature 1 đúng convention teacher của SGLang.
Trajectory có advantage không dương có score 0 và không cần đo KL.

Top-25% tính trên toàn nhóm data-parallel gồm trajectory có cycle và advantage/
reward hữu hạn, làm tròn lên và chỉ chọn score dương. Teacher top-k và hidden
state mới thay thế teacher cũ cho **mọi cycle** của trajectory được chọn. Các
trajectory còn lại giữ nguyên teacher đã ghi nhận. Sau đó chỉ chạy **một bước
head DCA/VGM trên cả batch**, trung bình cycle trong mỗi trajectory rồi trung
bình theo số trajectory toàn batch. Trajectory không có cycle đóng góp loss 0.
Không thêm reward/CE loss; target và trạng thái optimizer của target được giữ
nguyên trong bước head. Scheduler chỉ bước theo PPO như trước.

Các bước không refresh vẫn dùng GrowMTP gốc. Đặt fraction 0 giữ luồng GrowMTP
gốc và chỉ đo probe; đặt frequency 0 tắt cả probe và refresh.

## Metric và cách đọc

| Nhóm | Metric chính | Ý nghĩa |
|---|---|---|
| Acceptance thực tế | `rollout/mtp/acceptance_length`, `acceptance_rate`, `acc_rate_step*` | Verification trong rollout. Acceptance length gồm bonus/replacement token: `1 + accepted_draft_tokens / verification_steps`. |
| Tốc độ rollout | `rollout/mtp/generated_tokens_per_second_per_gpu`, `ms_per_generated_token`, `timing_s/gen` | Tốc độ sinh response; khác `perf/throughput`, vốn tính cả prompt và response trên thời gian step. |
| Chi phí | `comparison/time/step_e2e_s`, `step_gpu_hours`, `step_evaluation_s`, `step_probe_s` | Thời gian đo được và GPU-hours, gồm probe/checkpoint trong step và validation bên ngoài step. |
| Chi phí draft | `draft/update_seconds` | Thời gian DCA/VGM backward thường lệ và refresh draft head; MAX giữa các rank. Timer refresh gồm backward, clipping và optimizer step. |
| Chất lượng | `critic/score/mean`, `critic/rewards/mean`, `val-core/*`, `val-aux/*` | Reward train và validation độc lập. So sánh chất lượng cuối, đường học theo step, thời gian và GPU-hours. |
| Advantage | `comparison/advantage/*` | Mean/std, tỷ lệ dương/âm/bằng 0 của **mean advantage có mask của từng trajectory**; count cho trajectory rỗng hoặc không hữu hạn. |
| Policy shift | `draft/probe/kl_new_old_mean`, `kl_new_old_max`, `tv_new_old_mean` | KL(new target \|\| old target), TV trên cùng context; toàn vocabulary tại các vị trí lấy mẫu. Khác KL tới reference của PPO. |
| Lag | `draft/probe/tau_pre_surrogate`, `tau_post_fixed_draft_surrogate`, `delta_lag_surrogate` | Ước lượng acceptance trước/sau đổi target, giữ nguyên phân phối draft cũ. |
| Joint update | `draft/probe/tau_post_joint_surrogate`, `joint_update_gain_surrogate` | Ước lượng sau toàn bộ PPO và head update, bao gồm selective refresh nếu bật. |
| Detector đầy đủ | `draft/shift/kl_new_old_mean`, `measured_trajectories`, `num_positions`, `cache_bytes` | KL trung bình theo trajectory **đã đo** (advantage dương), số trajectory/vị trí và dung lượng cache FP32 tạm. |
| Chọn refresh | `draft/refresh/batch_trajectories`, `candidate_count`, `selected_count`, `selected_fraction`, `score_mean`, `score_max` | Candidate là trajectory đủ điều kiện của toàn batch; selection dùng KL trên mọi cycle, chỉ chọn score dương. |
| Refresh thực tế | `draft/refresh/applied`, `head_step_applied`, `updated_cycles`, `dca_loss`, `grad_norm`, `lr`, `time_s` | `applied=1` khi có teacher mới và optimizer head thực sự bước; `updated_cycles=0` nếu gradient không hữu hạn. `head_step_applied` vẫn có thể bằng 1 khi cả batch học teacher cũ. Loss/norm là của batch teacher hỗn hợp. `time_s` gồm detector trước/sau PPO, tạo teacher mới và head update, chưa gồm probe chẩn đoán. |

Giữ các khóa `actor/*`, `actor/mtp/*`, `target/*`, `draft/*` sẵn có.
`comparison/config/*` ghi GPU count, batch size, data seed, draft depth, sampling,
validation và ngân sách probe để kiểm tra cấu hình hai run.

## Probe trước/sau cập nhật

Probe lưu `p_old`, `q_old` trước mini-batch đầu, rồi tính `p_new`, `q_new` sau
mini-batch cuối và head update trên cùng chuỗi token. Context target được dựng từ token gốc;
MTP dùng đúng cặp `hidden(token_i), embedding(token_{i+1})`.

Ở mỗi depth, `alpha = sum_v min(p(v), q(v))`. Trên đường draft cố định:

```text
tau_surrogate = 1 + sum_k product_{j=1..k} alpha_j
delta_lag_surrogate = tau(p_old, q_old) - tau(p_new, q_old)
joint_update_gain_surrogate = tau(p_new, q_new) - tau(p_new, q_old)
```

Lag giữ dấu âm: policy shift có thể cải thiện acceptance. `tau_surrogate` chưa
phải acceptance length thực tế của rollout mới. Đường draft cố định có thể gồm
cả vị trí sau rejection; `verify_valid_length` ghi phạm vi supervision hợp lệ
của cycle gốc. Dùng metric verification thực tế để kết luận tốc độ/acceptance.

Target dùng full-vocabulary softmax với rollout temperature; draft dùng softmax
logit thô như SGLang MTP. Probe **không áp dụng top-p/top-k truncation** cho target;
`unfiltered_target=1` và `target_temperature` ghi rõ điều này. Với train mặc định
`temperature=1, top_p=1, top_k=-1`, phân phối phù hợp sampling của train. Khi bật
truncation, đây là surrogate của phân phối chưa lọc.

Mặc định probe chẩn đoán mỗi 4 actor update, tối đa **4 cycle trên
toàn nhóm data parallel**, với context không quá 1024 token. Trước khi giới hạn
pool, mỗi trajectory lấy ngẫu nhiên một cycle đủ điều kiện; các rank hợp nhất
candidate rồi cùng replay tập case đó, kể cả rank không có cycle. Candidate được
chọn không thiên theo advantage; advantage chỉ dùng sau khi đo KL để tính score.
Không cắt context. Các count `recorded_cycles`, `eligible_cycles`,
`skipped_context_cycles`, `skipped_nonfinite_cycles`, `num_cycles`, `num_positions`
cho biết coverage **chẩn đoán**. Không có case hợp lệ thì count probe bằng 0 và
không ghi KL probe; refresh vẫn xét đầy đủ trajectory, không bị giới hạn bởi
`comparison_probe_max_cycles` hoặc `comparison_probe_max_context`.
Giới hạn context có thể làm mẫu nghiêng về cycle đầu; tăng ngân sách nếu cần
đánh giá context dài và dùng cùng ngân sách cho hai phương pháp.

Đo phân phối p/q không ghi gradient và khôi phục RNG, train/eval mode cùng thiết
lập sharding; bước teacher hỗn hợp có backward chỉ qua draft head. Hiện hỗ trợ
FSDP2 với sequence parallel size 1 như pipeline GrowMTP hiện tại. Chi phí probe
và refresh luôn nằm trong E2E.

Detector KL đầy đủ phải forward target trước/sau PPO trên mọi cycle của
trajectory có advantage dương. Chỉ subset được chọn tạo lại top-k teacher và
hidden state dùng cho DCA. Detector stream log-prob FP32 cũ vào file tạm ở DP
rank 0 để tránh giữ toàn batch logits trong RAM/GPU; file đóng/xóa khi xong hoặc
khi actor lỗi. Dung lượng xấp xỉ `4 * measured_positions * vocabulary_size`
bytes trong thư mục tạm của hệ thống. Mỗi lần chỉ project logits tại các vị trí
verifier, không tạo logits cho toàn context. Chi phí detector tăng theo số cycle
và context; Top-25% không đảm bảo E2E overhead nhỏ. Cần đo `time_s`, `cache_bytes`
và E2E khi so sánh với baseline.

## Cấu hình và artifact

Preset GrowMTP của `scripts/train.sh` bật probe mỗi 4 step, log trajectory,
validation trước train/cuối train; periodic validation vẫn là `test_freq=50`.
B200 `full` bật probe, validation đầu/cuối với `val_kwargs.n=16`; periodic
validation mặc định tắt (`FULL_TEST_FREQ=-1`). B200 `smoke`/`pilot` mặc định tắt
probe và validation. Bật rõ chúng khi kiểm tra instrumentation trên GPU:

```bash
RUN_MODE=smoke MTP_PROBE_FREQ=1 FINAL_VALIDATION=1 VAL_BEFORE_TRAIN=1 \
  VAL_SAMPLES=1 bash scripts/run_b200_growmtp_lora_server.sh
```

Các biến B200 sau được lưu trong `config/resume.sh`:

| Biến | Mặc định full |
|---|---:|
| `MTP_PROBE_FREQ` | 4; đặt 0 để tắt |
| `MTP_PROBE_MAX_CYCLES` | 4, chỉ giới hạn chẩn đoán |
| `MTP_PROBE_MAX_CONTEXT` | 1024, chỉ giới hạn chẩn đoán |
| `MTP_REFRESH_FRACTION` | 0.25; đặt 0 để chỉ probe, không refresh |
| `COMPARISON_LOG_TRAJECTORIES` | 1 |
| `VAL_BEFORE_TRAIN`, `FINAL_VALIDATION` | 1 |
| `VAL_SAMPLES` | 16 |

Launcher thường nhận Hydra override tương ứng, ví dụ
`++actor_rollout_ref.model.mtp.comparison_probe_max_context=2048`.
Đặt `MTP_REFRESH_FRACTION=0` để chạy GrowMTP không refresh nhưng vẫn đo probe;
probe overhead vẫn được tính vào E2E.
`trainer.final_validation=true` cho phép validation cuối dù `test_freq=-1`.
Nếu dùng test benchmark làm `val_files`, nên chỉ đánh giá đầu/cuối; dùng
validation split riêng khi theo dõi hoặc chọn hyperparameter nhiều lần.

```text
RUN_DIR/
  logs/metrics.jsonl
  artifacts/comparison/trajectories/<step>.jsonl
  artifacts/comparison/probes/<step>.jsonl
  artifacts/comparison/shifts/<step>.jsonl
```

Không đặt `GROWMTP_RUN_DIR` thì root là `trainer.default_local_dir`. Scalar JSONL
vẫn được lưu khi không đặt `GROWMTP_LOG_LEVEL`, với backend console mặc định.
Trajectory log chứa prompt group ID, advantage có mask, score, reward, response
length và accepted/proposed/verification counts. Probe log chứa cycle/rank/
trajectory ID, advantage, alpha từng depth, KL, surrogate, policy-shift score
và `refresh_selected`. File `shifts` ghi score, KL toàn trajectory, số cycle,
`refresh_selected` và `refresh_applied` cho mọi candidate của selector; KL là
`null` và `kl_measured=false` nếu advantage không dương. Các log này không cần
`SAVE_GENERATIONS=1`, không lưu token hay hidden state.

## Tổng hợp và so sánh

```bash
python scripts/compare_growmtp_metrics.py \
  /runs/baseline/logs/metrics.jsonl /runs/improved/logs/metrics.jsonl \
  --output /runs/comparison.json

# --merge dành cho các đoạn log của CÙNG một run đã resume.
python scripts/compare_growmtp_metrics.py segment1.jsonl segment2.jsonl --merge
```

Script đọc cả log cũ, báo coverage và `null` cho metric thiếu. Acceptance dùng
pooled counts khi có; log cũ chỉ có scalar dùng mean theo step, với nhãn chỉ rõ.
Các mục `probe`, `shift`, `refresh` trong báo cáo chứa thống kê và số bản ghi cho
từng metric; các bước không refresh không bị điền giá trị 0 giả.
Khi merge, training row cuối ở mỗi global step thắng; từng lần validation riêng
vẫn được tính chi phí, kể cả nhiều lần tại cùng step.

`common_training_step_e2e_speedup_vs_baseline` so sánh training step chung có
timing, gồm evaluation gắn với step. `run_total_e2e_speedup_vs_baseline` bao gồm
validation riêng và chỉ xuất khi coverage step, timing và lịch đánh giá phù
hợp; nếu không là `null`. GPU-hours speedup được báo riêng, cùng thông tin GPU
count có khớp baseline hay không.

E2E là **thời gian được đo**: step cộng validation. Startup, chuẩn bị model và
công việc ngoài timer chưa được tính. `step_training_s` trừ `draft/probe/time_s`,
bao gồm đo policy shift và refresh teacher/head, chỉ để phân tích overhead; dùng
E2E có probe để so sánh tốc độ chính. Counter `session_*`
khởi động lại khi resume; tổng hợp dùng per-step đã loại trùng, không cộng
các counter này.

Hai phương pháp cần cùng initial checkpoint/head, dataset, seed, batch size,
sampling, draft depth, GPU, giới hạn token và lịch validation/probe. Dùng config
đã resolve để kiểm tra phần không phải scalar. Log cũ không thể khôi phục KL
hay validation chưa từng đo; cần baseline mới để so sánh đầy đủ.

CUDA FSDP2/Ray nhiều rank chưa được chạy trong môi trường local; cần smoke trên
host GPU trước thí nghiệm dài.
