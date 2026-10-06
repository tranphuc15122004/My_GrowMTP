# Policy-Shift-Aware GrowMTP: hướng cải tiến ít overhead

## Trạng thái và mục tiêu

Có hai phương án cần phân biệt:

1. **Exact-KL selective refresh** là ý tưởng ban đầu, đã được triển khai. Sau PPO, phương án này chạy lại target để đo sự khác biệt giữa policy cũ và mới, chọn trajectory cần refresh, rồi dùng target mới làm teacher cho DCA/VGM.
2. **Rollout-advantage auxiliary loss** đã được tích hợp vào mã với mặc định tắt. Nó giữ nguyên DCA/VGM và dùng token target đã phát ra trong rollout cùng advantage dương làm tín hiệu phụ cho draft head. Đường train chính không cần chạy target sau PPO. Việc ánh xạ token đã được triển khai, nhưng cần smoke/pilot trên môi trường train để xác nhận runtime.

Mục tiêu cuối cùng là giảm **thời gian end-to-end của RL post-training** so với GrowMTP gốc trên cùng phần cứng và cấu hình, đồng thời giữ chất lượng policy. Acceptance length tăng nhưng tổng thời gian không giảm thì chưa đạt mục tiêu.

## 1. Vấn đề cần kiểm chứng

Tại RL step $t$, target $p_t$ sinh rollout và verification signal cho draft $q_\phi$. Policy được cập nhật thành $p_{t+1}$, trong khi draft vừa học từ $p_t$ sẽ được dùng với $p_{t+1}$ ở rollout tiếp theo:

$$
q_{\phi,t+1}\approx p_t,
\qquad
\text{nhưng rollout tiếp theo dùng }p_{t+1}.
$$

Đây là độ trễ một bước, **chưa chắc là bottleneck**. Pilot 8 bước của bản exact-KL chỉ đo một trajectory có advantage dương ở step 4, với KL trung bình khoảng $9.37\times10^{-4}$ nat. Cùng step đó, thời gian diagnostic probe khoảng 783 giây; ở step 8 là khoảng 162 giây. Các số này cho thấy chấm lại target có thể rất tốn thời gian, nhưng chưa chứng minh policy shift gây giảm acceptance đáng kể. GrowMTP gốc 500 bước trên server mất khoảng 12 giờ theo log người dùng cung cấp.

## 2. Giữ nguyên objective acceptance của GrowMTP

Tại vị trí $j$ của verification cycle $c$, gọi $p_{t,c,j}$ và $q_{\phi,c,j}$ là hai phân phối trên **cùng draft-conditioned state**. Xác suất chấp nhận trung bình:

$$
\alpha_{c,j}
=\sum_v\min\{p_{t,c,j}(v),q_{\phi,c,j}(v)\}
=1-\operatorname{TV}(p_{t,c,j},q_{\phi,c,j}).
$$

Với draft depth $K$ và vị trí từ chối đầu tiên $J_c$ ($J_c=K+1$ nếu chấp nhận hết), DCA với Verify-Gated Masking (VGM) là

$$
DCA_c(q_\phi,p_t)
=-\log\left(
  \sum_{\ell=1}^{\min(J_c,K)}
  \prod_{j=1}^{\ell}\alpha_{c,j}
\right).
$$

VGM giữ đến **và gồm** vị trí từ chối đầu tiên. GrowMTP hiện đã triển khai draft-path reconstruction, DCA/VGM và xấp xỉ overlap bằng target top-k cộng residual bin; cần tái sử dụng chúng. Chỉ draft head nhận gradient. DCA gắn trực tiếp với số token có thể được chấp nhận. KL ở phương án cũ là **bộ đo policy shift**, không thay thế DCA.

## 3. Đối chứng exact-KL selective refresh đã triển khai

Trên các trajectory có advantage dương, phương án cũ đo shift sau PPO:

$$
D_r=\frac{1}{N_r}\sum_{(c,j)\in r}
D_{\mathrm{KL}}\left(p_{t+1,c,j}\|p_{t,c,j}\right),
\qquad
S_r=[A_r]_+D_r.
$$

Một số trajectory có $S_r$ cao được học bằng teacher $p_{t+1}$; phần còn lại tiếp tục học bằng $p_t$. Preset hiện tại kích hoạt mỗi 4 step và chọn tối đa 25% tổng số trajectory trong batch, chỉ trong nhóm có score dương.

Điểm tốn kém: trước khi biết trajectory nào được chọn, detector vẫn phải chấm lại target trên **mọi cycle của các trajectory advantage dương**. Giới hạn cycle của diagnostic probe không giới hạn detector này; chọn 25% không đồng nghĩa chỉ trả 25% chi phí target forward. Phương án này là đối chứng để đo shift thực, nhưng chỉ có ích cho tăng tốc nếu gain rollout bù được chi phí detector, probe và refresh.

## 4. Phương án mới: DCA cộng CE từ token rollout có advantage dương

Gọi $y_{c,j}$ là **token target thực sự phát ra** tại draft-conditioned state $s_{c,j}$; $r(c)$ là trajectory chứa cycle $c$. Token này không phải ground-truth answer và không nhất thiết bằng draft proposal. Tại vị trí từ chối đầu tiên của một cycle, $y_{c,j}$ là token sửa từ phân phối residual.

Với speculative rejection sampling đúng, tại một state đã đi tới, token đầu ra có phân phối biên $p_t$:

$$
\Pr(Y=y,\text{accept})=\min\{p_t(y),q(y)\},
\qquad
\Pr(Y=y,\text{reject})=[p_t(y)-q(y)]_+,
$$

$$
\Pr(Y=y)=p_t(y).
$$

Vì vậy ta có thể dùng rollout đã có làm nguồn token target. Đặt advantage không âm có chặn:

$$
w_r=\min\{\max(A_r,0),A_{\max}\}.
$$

Với $L_c=\min(J_c,K)$, CE được **lấy trung bình trong từng cycle**:

$$
CE_c(q_\phi,y)
=-\frac{1}{L_c}\sum_{j=1}^{L_c}
\log q_\phi(y_{c,j}\mid s_{c,j}).
$$

Khi GrowMTP lấy trung bình theo cycle, loss đề xuất là

$$
\boxed{
L_{\mathrm{draft}}
=\frac{1}{|\mathcal C|}\sum_{c\in\mathcal C}
\left[DCA_c(q_\phi,p_t)+\lambda w_{r(c)}CE_c(q_\phi,y)\right].
}
$$

Nếu cấu hình hiện tại lấy trung bình theo trajectory thì **cả hai hạng tử phải dùng cùng phép gộp**. Không chia tiếp cho số trajectory có advantage dương: nếu chỉ có một trajectory dương, phép chia ấy sẽ khuếch đại tín hiệu của nó. Khi mọi $A_r=0$, auxiliary bằng 0 và loss trở về GrowMTP gốc. Target distribution, token rollout, advantage và backbone hidden state đều được detach; objective PPO/GRPO của actor giữ nguyên.

### Trực giác về hướng dịch chuyển

Ở mức logit của một state đơn lẻ, policy gradient theo token rollout $y$ có hướng xấp xỉ

$$
\Delta z^p_v\propto A_r
\left(\mathbf 1[v=y]-p_t(v\mid s)\right).
$$

Gradient của CE lên draft logit là

$$
\frac{\partial[-w_r\log q_\phi(y\mid s)]}{\partial z^q_v}
=w_r\left(q_\phi(v\mid s)-\mathbf 1[v=y]\right).
$$

Nếu $q_\phi\approx p_t$, giảm CE trên trajectory có $A_r>0$ có thể đẩy draft theo hướng policy được khuyến khích di chuyển. Đây chỉ là **proxy của policy shift**, không tái tạo chính xác $p_{t+1}$: PPO clipping, regularization, tham số chung giữa các state và trajectory advantage âm đều ảnh hưởng đến cập nhật target thực. DCA vẫn neo draft vào toàn bộ phân phối target cũ, tránh CE ép quá mạnh vào một token.

## 5. Cân bằng scale của hai loss

Không chọn $\lambda$ bằng cách so hai giá trị scalar. DCA có thể âm khi tổng acceptance-chain lớn hơn 1; CE luôn không âm. Tổng CE theo token cũng không được ghép với DCA trung bình theo cycle, vì batch có nhiều vị trí hợp lệ sẽ vô tình tăng trọng số auxiliary.

Trên **một batch hiệu chuẩn có advantage dương**, đo gradient theo tham số draft head $\phi$:

$$
g_D=\|\nabla_\phi L_{\mathrm{DCA}}\|_2,
\qquad
g_C=\|\nabla_\phi L_{\mathrm{CE,weighted}}\|_2,
\qquad
\rho=\frac{\lambda g_C}{g_D+\epsilon}.
$$

Điểm bắt đầu để thử: $\rho\approx0.1$, tức $\lambda_0\approx0.1g_D/(g_C+\epsilon)$, rồi **giữ cố định $\lambda_0$** trong pilot. Có thể thử thêm $\rho\approx0.2$ nếu tín hiệu phụ quá yếu. Đây là hyperparameter giả thuyết, chưa phải giá trị tối ưu. Không hiệu chuẩn từ batch không có trajectory advantage dương hoặc có $g_C$ gần 0. Cố định $A_{\max}$ trước khi so sánh; mức chặn 2 là một điểm khởi đầu có thể thử với GRPO advantage đã chuẩn hóa.

Đặt $d=\nabla_\phi L_{\mathrm{DCA}}$ và $c=\nabla_\phi L_{\mathrm{CE,weighted}}$. Với SGD, nếu $\lambda\|c\|\leq\rho\|d\|$ và $\rho<1$ thì

$$
d^\top(d+\lambda c)\geq(1-\rho)\|d\|^2>0.
$$

Tức là bước cập nhật vẫn giảm DCA ở bậc một trên batch hiệu chuẩn, dù hai gradient ngược hướng. Đây **không phải bảo đảm cho Adam**, gradient clipping hoặc các batch tiếp theo. Chỉ đo tỷ lệ và cosine gradient trên một số batch trong pilot; không chạy hai backward riêng ở mọi step của full train vì sẽ làm mất lợi thế tốc độ.

## 6. Ràng buộc triển khai tối thiểu

- Giữ nguyên draft-path reconstruction, VGM, target top-k/residual overlap, DCA và optimizer group. CE nên tái sử dụng draft logits đã được chiếu lên vocabulary khi tính DCA. Cần đo chi phí thực tế vì checkpoint/recompute có thể làm tăng overhead.
- Xác nhận $y_{c,j}$ khớp chính xác với $s_{c,j}$, đặc biệt tại vị trí từ chối đầu tiên, cycle cuối và khi response bị cắt. Bonus token khi chấp nhận hết không có draft logit tương ứng trong $K$ vị trí thì không dùng. Không dùng draft proposal thay cho token target phát ra.
- Khi auxiliary bật, signal transport bổ sung $y_{c,j}$ và mask bằng response index $position+j+2-prompt\_length$ với $j$ bắt đầu từ 0. Prompt trong signal đã dịch trái và chứa token rollout đầu tiên làm seed; depth đầu dự đoán token rollout thứ hai. Mask giao với VGM và response mask. Smoke/pilot vẫn cần xác nhận runtime cho các trường hợp accept, reject và truncate.
- Bật/tắt auxiliary bằng config, mặc định tắt để GrowMTP gốc không đổi. Không kết hợp auxiliary với exact-KL refresh trong thí nghiệm đầu tiên: cần tách tác dụng và chi phí của từng phương án.

## 7. So sánh và tiêu chí quyết định

So sánh từ **cùng checkpoint ban đầu**, cùng dữ liệu, seed, effective batch, GPU, giới hạn response và lịch validation:

| Nhánh | Loss draft | Chạy lại target sau PPO |
|---|---|---|
| GrowMTP gốc | DCA/VGM | Không |
| Exact-KL refresh đã có | DCA/VGM với teacher chọn lọc $p_{t+1}$ | Có |
| Đề xuất ít overhead | DCA/VGM $+\lambda w_r CE$ | Không trong train chính |

Trước hết chạy pilot 30–50 step cho GrowMTP gốc và phương án auxiliary; chỉ chạy 500 step nếu pilot cho thấy khả năng cải thiện. Diagnostic KL nếu cần nên chạy cùng lịch ở mọi nhánh hoặc tách khỏi phép đo tốc độ. Không suy ra thời gian full train chỉ từ các step đầu vì draft head còn trong giai đoạn khởi động.

Log tối thiểu: DCA, auxiliary CE, $\lambda$, positive-advantage fraction, tỷ lệ/cosine gradient trên batch đo, acceptance length và $\alpha_j$, rollout time, head-update time, end-to-end step time, reward/accuracy, response clipping và tỷ lệ step có policy-gradient loss khác 0.

Để nghiên cứu policy shift trên một tập diagnostic nhỏ, có thể đo acceptance-chain surrogate trên **cùng fixed states**:

$$
\widetilde\tau(q,p)
=\sum_{\ell=1}^{K}\prod_{j=1}^{\ell}
[1-\operatorname{TV}(q_j,p_j)],
\qquad
\Delta_{\mathrm{lag}}
=\widetilde\tau(q,p_t)-\widetilde\tau(q,p_{t+1}).
$$

Đây là surrogate chẩn đoán tốn thêm target forward, **không phải** acceptance length quan sát trực tiếp trong rollout kế tiếp; không cần đo mỗi train step.

**Điều kiện thành công:** thời gian end-to-end 500 step thấp hơn GrowMTP gốc trong điều kiện tương đương, không giảm chất lượng policy đáng kể và gain rollout lớn hơn chi phí auxiliary. Nếu acceptance không cải thiện hoặc CE thường xuyên xung đột mạnh với DCA, giảm $\lambda$ hoặc bỏ auxiliary. KL pilot nhỏ cũng có thể cho thấy staleness không phải bottleneck; đó là kết quả nghiên cứu hợp lệ.

## 8. Giả thuyết cần kiểm chứng

1. Token rollout có advantage dương cho draft một tín hiệu xấp xỉ hướng policy update mà không chạy lại target.
2. Khi giữ gradient auxiliary nhỏ so với DCA, tín hiệu này cải thiện acceptance trên rollout sau PPO.
3. Gain rollout, nếu có, lớn hơn chi phí loss phụ và tạo ra **net end-to-end speedup** so với GrowMTP gốc.

Không giả định các giả thuyết này đúng. Exact-KL vẫn là đối chứng để đo shift thực; auxiliary là hướng ưu tiên nghiên cứu khi mục tiêu chính là tăng tốc post-training.

## Tài liệu tham khảo

- GrowMTP, công thức DCA/VGM và phân tích độ trễ policy: https://arxiv.org/abs/2609.16648
- PPO, objective policy-gradient có clipping: https://arxiv.org/abs/1707.06347
- GradNorm, động cơ cân loss theo gradient thay vì giá trị scalar: https://proceedings.mlr.press/v80/chen18a.html
