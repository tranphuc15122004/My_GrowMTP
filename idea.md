# Policy-Shift-Aware GrowMTP

## 1. Motivation

GrowMTP train draft head online từ verification signal của target policy hiện tại:

$$
\pi_t
\rightarrow
p_t
\rightarrow
q_t,
$$

trong đó \(p_t\) là distribution của target verifier và \(q_t\) là distribution của draft head.

Vấn đề là sau mỗi RL step, target policy được update:

$$
\pi_t \rightarrow \pi_{t+1}.
$$

Trong khi draft head vừa được tối ưu để match \(p_t\), nó lại được sử dụng ngay với target mới \(p_{t+1}\).

Do đó tồn tại một dạng **draft–target staleness**:

$$
q_t \approx p_t
\qquad
\text{nhưng inference tiếp theo cần}
\qquad
q_t \approx p_{t+1}.
$$

GrowMTP hiện tại không xử lý trực tiếp sự thay đổi này.

---

## 2. Core idea

Thay vì chỉ train drafter để khớp với **current policy**, phương pháp đề xuất làm cho drafter thích nghi với **updated/future policy**.

High-level transition:

$$
\boxed{
\text{Current-policy online distillation}
\rightarrow
\text{Policy-shift-aware online distillation}
}
$$

Ý tưởng trung tâm:

> Không phải mọi rollout state đều cần được retrain bằng target mới. Chỉ những state mà target distribution thực sự thay đổi đáng kể sau RL update mới cần refresh supervision.

---

## 3. Hai nguồn tín hiệu

Phương pháp sử dụng hai loại tín hiệu bổ sung nhau.

### 3.1 Task signal: Advantage / Reward

GRPO đã cung cấp advantage:

$$
A_r
$$

cho mỗi rollout \(r\).

Advantage cho biết trajectory nào policy **được khuyến khích dịch chuyển tới**:

$$
A_r>0
\Rightarrow
\pi_{t+1}(y_r|x)
\text{ có xu hướng tăng}.
$$

Ground-truth answer vì vậy không được dùng trực tiếp dưới dạng teacher-forcing CE.

Thay vào đó:

$$
y^*
\rightarrow
R
\rightarrow
A.
$$

Điều này tránh ép drafter bắt chước một reasoning trajectory reference duy nhất.

---

### 3.2 Distribution-shift signal: KL

Sau khi policy update:

$$
\pi_t\rightarrow\pi_{t+1},
$$

ta đo mức thay đổi thực sự của target:

$$
D_{r,k}
=
D_{\mathrm{KL}}
\left(
p_{t+1,r,k}
\|
p_{t,r,k}
\right).
$$

Advantage trả lời:

> policy **nên** thay đổi ở đâu?

KL trả lời:

> policy **thực sự đã** thay đổi ở đâu?

Hai tín hiệu này không trùng nhau.

---

# 4. Policy-shift score

Với rollout \(r\), định nghĩa average shift:

$$
D_r
=
\frac{1}{K_r}
\sum_{k=1}^{K_r}
D_{\mathrm{KL}}
\left(
p_{t+1,r,k}
\|
p_{t,r,k}
\right).
$$

Sau đó kết hợp với positive advantage:

$$
\boxed{
S_r
=
g(A_r)\,D_r
}
$$

với một lựa chọn đơn giản:

$$
g(A_r)=\max(\hat A_r,0).
$$

Hoặc nếu muốn ổn định hơn:

$$
g(A_r)=\sigma(\beta\hat A_r).
$$

Interpretation:

$$
S_r\text{ cao}
$$

khi:

1. trajectory có ích đối với RL objective;
2. target distribution sau update thực sự khác target cũ.

Đây chính là những samples mà GrowMTP supervision cũ dễ trở nên stale nhất.

---

# 5. Selective Future-Policy Refresh

Không recompute \(p_{t+1}\) cho toàn bộ rollout vì overhead sẽ lớn.

Chỉ chọn:

$$
\mathcal R_{\text{refresh}}
=
\operatorname{TopR}(S_r)
$$

ví dụ top \(25\%\).

Đối với sample không cần refresh:

$$
r\notin\mathcal R_{\text{refresh}},
$$

tiếp tục dùng GrowMTP bình thường:

$$
L_r
=
L_{\mathrm{DCA}}^{\mathrm{VGM}}
(q_r,p_{t,r}).
$$

Đối với sample policy đã shift mạnh:

$$
r\in\mathcal R_{\text{refresh}},
$$

recompute updated verifier signal:

$$
p_{t+1,r}
$$

và train drafter bằng:

$$
\boxed{
L_r
=
L_{\mathrm{DCA}}^{\mathrm{VGM}}
(q_r,p_{t+1,r})
}
$$

Thay vì học teacher cũ:

$$
q_t\rightarrow p_t,
$$

drafter học trực tiếp:

$$
\boxed{
q_t\rightarrow p_{t+1}
}
$$

ở những nơi cần thiết.

---

# 6. Tại sao vẫn giữ DCA?

Không nên thay DCA bằng KL.

Speculative acceptance liên hệ trực tiếp với distribution overlap:

$$
\alpha_k
=
1-\mathrm{TV}(p_k,q_k).
$$

Expected accepted length phụ thuộc theo chuỗi:

$$
E[A]
=
\sum_l
\prod_{k=1}^{l}\alpha_k.
$$

DCA được thiết kế để optimize cấu trúc này.

Vì vậy:

$$
\boxed{
\text{DCA = acceptance objective}
}
$$

trong khi:

$$
\boxed{
\text{KL = policy-shift detector}
}
$$

Đây là phân vai sạch nhất.

---

# 7. Final training objective

Một formulation đơn giản:

$$
\tilde p_r
=
\begin{cases}
p_{t+1,r},
&
r\in\mathcal R_{\text{refresh}}
\\[4pt]
p_{t,r},
&
\text{otherwise}.
\end{cases}
$$

Sau đó:

$$
\boxed{
L_{\text{PS-GrowMTP}}
=
\frac{1}{|\mathcal B|}
\sum_{r\in\mathcal B}
L_{\mathrm{DCA}}^{\mathrm{VGM}}
(q_r,\tilde p_r)
}
$$

Không cần thêm một loss phức tạp.

Điểm mới nằm ở **teacher selection/training protocol**, không nằm chủ yếu ở việc thêm một term vào objective.

---

# 8. Full pipeline

$$
\boxed{
\begin{aligned}
&\textbf{1. Rollout with } \pi_t\\
&\qquad\downarrow\\
&\text{collect draft paths + }p_t+A\\
&\qquad\downarrow\\
&\textbf{2. RL update}\\
&\pi_t\rightarrow\pi_{t+1}\\
&\qquad\downarrow\\
&\textbf{3. Estimate policy shift}\\
&D_{\mathrm{KL}}(p_{t+1}\|p_t)\\
&\qquad\downarrow\\
&\textbf{4. Compute }S=A\times KL\\
&\qquad\downarrow\\
&\textbf{5. Select high-shift trajectories}\\
&\qquad\downarrow\\
&\textbf{6. Refresh }p_{t+1}\text{ only for selected samples}\\
&\qquad\downarrow\\
&\textbf{7. DCA + VGM drafter update}\\
&\qquad\downarrow\\
&q_{t+1}
\end{aligned}
}
$$

---

# 9. Efficient variant

Để không phá end-to-end gain, không nhất thiết refresh ở mọi RL step.

Có thể dùng:

$$
M=4
$$

tức chỉ làm future refresh mỗi 4 steps.

Ví dụ:

```text
step 1: standard GrowMTP
step 2: standard GrowMTP
step 3: standard GrowMTP
step 4: policy-shift refresh

step 5: standard GrowMTP
...
```

Và mỗi refresh chỉ chọn:

$$
r=25\%
$$

samples.

Do đó overhead:

$$
C_{\text{refresh}}
\ll
C_{\text{full recomputation}}.
$$

Đây là phần quan trọng vì contribution cuối cùng phải cải thiện:

$$
\text{net E2E training time},
$$

không chỉ acceptance.

---

# 10. Có nên dùng ground-truth CE nữa không?

Không nên đặt nó vào core method.

Một auxiliary variant có thể thử:

$$
L
=
L_{\mathrm{PS-GrowMTP}}
+
\lambda L_{\mathrm{GT}},
$$

nhưng chỉ để ablation.

Lý do:

$$
\text{correct answer}
\neq
\text{unique correct reasoning trajectory}.
$$

Đặc biệt với Math và Code, nhiều trajectory khác nhau đều đúng.

Direct CE có thể khiến:

$$
q
$$

dịch khỏi:

$$
p_{\text{target}},
$$

làm acceptance giảm.

Vì vậy ground truth nên đi gián tiếp qua:

$$
\boxed{
y^*
\rightarrow reward
\rightarrow advantage
\rightarrow selection
}
$$

thay vì:

$$
y^*
\rightarrow CE(q,y^*).
$$

---

# 11. Core contribution

Contribution không nên được mô tả là:

> We improve the GrowMTP loss.

Mà là:

> **GrowMTP distills the drafter from the current target policy even though that policy is immediately updated by RL. We identify this supervision staleness and introduce policy-shift-aware online draft training that selectively refreshes the draft teacher using the updated target distribution.**

Ba contribution cụ thể:

### C1. Draft-policy staleness

Xác định mismatch:

$$
q_t\approx p_t
\qquad\text{while}\qquad
q_t\text{ serves }p_{t+1}.
$$

### C2. Policy-shift-aware supervision

Dùng:

$$
A_r
$$

và:

$$
KL(p_{t+1}\|p_t)
$$

để xác định supervision nào thực sự stale.

### C3. Selective future-policy refresh

Chỉ recompute updated supervision cho một subset có giá trị cao, giữ overhead thấp.

---

# 12. Metric mới cần report

Ngoài acceptance length:

$$
\tau,
$$

nên đo trực tiếp mechanism.

## Draft-policy lag

Trước policy update:

$$
\tau_{\text{pre}}
=
\tau(q_t,p_t).
$$

Sau update:

$$
\tau_{\text{post}}
=
\tau(q_t,p_{t+1}).
$$

Định nghĩa:

$$
\boxed{
\Delta_{\text{lag}}
=
\tau_{\text{pre}}
-
\tau_{\text{post}}
}
$$

GrowMTP baseline kỳ vọng:

$$
\Delta_{\text{lag}}>0.
$$

Method cần làm:

$$
\boxed{
\Delta_{\text{lag}}\downarrow.
}
$$

Đây là metric rất quan trọng vì nó trực tiếp kiểm chứng claim của method.

---

# 13. Metrics cuối cùng

Cần report đồng thời:

$$
\boxed{
\begin{aligned}
&\tau &&\text{acceptance length}\\
&\alpha_k &&\text{per-position acceptance}\\
&\Delta_{\text{lag}} &&\text{policy-draft staleness}\\
&T_{\text{gen}} &&\text{rollout time}\\
&T_{\text{step}} &&\text{E2E RL step time}\\
&\text{throughput} &&\text{tokens/s/GPU}\\
&\text{reward/task accuracy} &&\text{policy quality}\\
&C_{\text{refresh}} &&\text{extra overhead}
\end{aligned}
}
$$

Objective cuối cùng không phải:

$$
\max\tau
$$

mà là:

$$
\boxed{
\max
\frac{\text{useful accepted tokens}}
{\text{total training-system cost}}
}
$$

---

# 14. Experiment roadmap

## Phase 0 — LoRA baseline

$$
\text{GrowMTP + LoRA}
$$

xác nhận pipeline chạy đúng.

---

## Phase 1 — Cheap probe

So sánh:

$$
\text{DCA}
$$

vs:

$$
A\times DCA.
$$

Chỉ 30–50 steps.

Mục đích không phải contribution mà để kiểm tra task signal có hữu ích hay không.

---

## Phase 2 — Mechanism validation

Log:

$$
A,
\quad
KL(p_{t+1}\|p_t),
\quad
TV(p_t,q),
\quad
\alpha_k.
$$

Kiểm tra:

$$
KL\uparrow
\Rightarrow
\Delta_{\text{lag}}\uparrow?
$$

Nếu không có relation, dừng hướng này.

---

## Phase 3 — Core method

Triển khai:

$$
\boxed{
\text{Selective Policy-Shift Refresh}
}
$$

với:

$$
25\%
$$

samples mỗi:

$$
4
$$

steps.

---

## Phase 4 — Ablation

So sánh:

$$
\begin{aligned}
&\text{GrowMTP}\\
&+\text{Advantage weighting}\\
&+\text{KL-only selection}\\
&+\text{Advantage + KL selection}\\
&+\text{full future refresh}\\
&+\text{selective future refresh}.
\end{aligned}
$$

---

# 15. Hypothesis chính

### H1

RL update tạo ra measurable target-distribution shift:

$$
KL(p_{t+1}\|p_t)>0.
$$

### H2

Policy shift này gây giảm speculative compatibility:

$$
KL(p_{t+1}\|p_t)\uparrow
\Rightarrow
\tau(q_t,p_{t+1})\downarrow.
$$

### H3

Refresh supervision bằng updated policy giảm draft-policy lag:

$$
\Delta_{\text{lag}}^{\text{ours}}
<
\Delta_{\text{lag}}^{\text{GrowMTP}}.
$$

### H4

Selective refresh đạt gần gain của full refresh nhưng với overhead thấp hơn:

$$
Gain_{\text{selective}}
\approx
Gain_{\text{full}}
$$

trong khi:

$$
Cost_{\text{selective}}
\ll
Cost_{\text{full}}.
$$

### H5

Giảm policy-draft lag chuyển thành:

$$
\tau\uparrow
\rightarrow
T_{\text{rollout}}\downarrow
\rightarrow
T_{\text{RL}}\downarrow.
$$

---

# 16. Final high-level statement

Phương pháp có thể được cô đọng thành:

$$
\boxed{
\textbf{Train the drafter for the policy it will serve, not only the policy that generated its supervision.}
}
$$

GrowMTP:

$$
\text{current-policy distillation}.
$$

Phương pháp đề xuất:

$$
\boxed{
\text{policy-shift-aware future-policy distillation}.
}
$$

Reward/advantage cho biết **hướng dịch chuyển có giá trị**, KL đo **mức dịch chuyển thực tế**, selective refresh lấy supervision từ **updated target**, còn DCA tiếp tục đảm bảo objective phù hợp với speculative acceptance.

Đây là phiên bản mà tôi nghĩ nên khóa làm hướng chính để triển khai.
