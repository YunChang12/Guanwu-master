# 基于 URDF、关节角和单目 RGB 图像的机械臂基座位姿估计方案

## 1. 目标

给定：

- 机械臂 `URDF` 文件；
- 当前帧关节角 `q`；
- 相机内参 `K`；
- 当前 RGB 图像 `I`；
- 图像中的机械臂可能只被部分拍到，或者存在遮挡；

优化求解机械臂基座坐标系 `B` 在相机坐标系 `C` 下的刚体位姿：

```text
T_C_B = [R_C_B | t_C_B]
```

其中 `t_C_B` 是机械臂基座原点在相机坐标系下的位置，`R_C_B` 是基座坐标系相对于相机坐标系的旋转。

最终输出：

- 优化后的 `T_C_B`；
- 位姿数值结果，例如 `4x4` 齐次矩阵、平移向量、旋转向量或四元数；
- 可视化结果：将 URDF 在估计位姿下渲染到原图上，形成 overlay 对比图。

本方案不针对单独机械臂训练网络。机械臂差异通过 `URDF + q` 输入建模。

## 2. 基本假设

### 2.1 必需输入

```text
I: H x W x 3 RGB 图像
K: 3 x 3 相机内参矩阵
URDF: 包含 link mesh、joint、joint limit 的机械臂模型
q: 当前帧关节角，单位与 URDF 一致
```

### 2.2 可选输入

```text
T0_C_B: 初始基座位姿
M_obs: 已分割出的机械臂可见区域 mask
camera_distortion: 相机畸变参数
robot_mask_prompt: 用于 SAM / Grounded-SAM 的框、点或文本提示
```

### 2.3 坐标系约定

建议统一使用 OpenCV 相机坐标系：

```text
x: 图像右方
y: 图像下方
z: 相机前方
```

机械臂基座坐标系 `B` 使用 URDF 默认 root link 坐标系。

三维点投影：

```text
X_C = R_C_B X_B + t_C_B
u_h = K X_C
u = [u_h.x / u_h.z, u_h.y / u_h.z]
```

所有 mesh 顶点、link transform、joint angle 的单位必须一致。通常 URDF 使用米和弧度。

## 3. 总体算法流程

```text
输入: I, K, URDF, q

1. 读取 URDF
2. 根据 q 做正运动学，得到当前构型下的整机 mesh_B(q)
3. 从 RGB 图像中提取机械臂可见区域 M_obs
4. 生成一个或多个初始位姿 T0_C_B
5. 对每个初始位姿执行 render-and-compare 优化
6. 选择 loss 最低且几何合理的结果
7. 将最优位姿下的机械臂渲染到原图
8. 输出 T_C_B 和可视化图像
```

## 4. 模块设计

### 4.1 URDF 解析与正运动学模块

职责：

- 加载 URDF；
- 读取 link mesh；
- 根据关节角 `q` 计算每个 link 到基座坐标系的变换；
- 合成当前构型下的整机 mesh 或每个 link 的 mesh 列表。

推荐库：

```text
yourdfpy
urdfpy
pytorch_kinematics
trimesh
Pinocchio
```

输出：

```text
meshes_B(q) = [
  {vertices_B, faces, link_name},
  ...
]
```

注意事项：

- 检查 URDF mesh 路径是否能正确解析；
- 检查 mesh scale，很多 URDF 的 mesh scale 可能定义在 `<mesh scale="...">` 中；
- 忽略视觉上不可见或无几何的 link；
- 对 mimic joint、fixed joint 做正确处理；
- 确保 joint angle 顺序与输入 `q` 一致。

### 4.2 图像分割模块

职责：

- 从 RGB 图像中得到机械臂可见区域 mask：

```text
M_obs in {0, 1}^{H x W}
```

可选方法：

```text
1. SAM / SAM2: 使用点、框或第一帧提示
2. Grounded-SAM: 使用文本提示，例如 "robot arm"
3. 颜色阈值: 适合背景简单、机械臂颜色明显的场景
4. 视频跟踪: 第一帧人工或 SAM 分割，后续用跟踪器传播 mask
```

输出建议：

```text
M_obs: 二值 mask
M_obs_soft: 可选软 mask 或置信度 mask
C_obs: 从 M_obs 提取的有效轮廓
```

关键要求：

- mask 只表示图像中实际可见的机械臂部分；
- 不要求覆盖整台机械臂；
- 被画面裁切、遮挡或漏分割的部分不应强行补全。

### 4.3 可微或近似可微渲染模块

职责：

给定候选位姿 `T_C_B`，将当前构型的机械臂 mesh 渲染到图像平面：

```text
M_ren(T): 渲染 silhouette mask
C_ren(T): 渲染轮廓
RGB_ren(T): 可选渲染颜色图
Z_ren(T): 可选渲染深度图，仅用于可见性处理，不依赖真实深度
```

推荐实现：

```text
PyTorch3D
nvdiffrast
Kaolin
OpenDR 风格 silhouette renderer
```

如果第一版不做端到端梯度优化，也可以使用普通 rasterizer：

```text
trimesh + pyrender
Open3D visualizer offscreen
自写 OpenCV polygon rasterization
```

然后用无梯度优化器，例如 CMA-ES、Nelder-Mead、Powell 或 coarse-to-fine search。

## 5. 优化变量

由于关节角 `q` 已知，优化变量只有基座位姿：

```text
theta = [omega_x, omega_y, omega_z, t_x, t_y, t_z]
```

其中：

- `omega` 是 SO(3) 李代数旋转向量；
- `t` 是平移向量；
- `R = exp(omega)`。

从初始位姿 `T0_C_B` 做增量优化更稳定：

```text
T_C_B(theta) = delta_T(theta) * T0_C_B
```

其中 `delta_T` 是待优化的小扰动。

## 6. Loss 设计

由于机械臂可能拍不全，核心原则是：

```text
只强制观测到的机械臂区域能被渲染模型解释；
不要强惩罚渲染出来但图像中没有观测到的区域。
```

因此不建议把标准 IoU、整图 BCE 或双向 Chamfer 作为主 loss。

### 6.1 单向 observed-to-render mask distance loss

令：

```text
M_obs: 观测到的机械臂 mask
M_ren: 当前位姿渲染出的机械臂 mask
D_M_ren: 到 M_ren 的距离变换图
```

定义：

```text
L_obs2ren_mask =
  mean_{p in M_obs} robust(D_M_ren(p))
```

含义：

真实图像中看到的机械臂像素，应该尽可能落在渲染出的机械臂区域附近。

推荐 robust 函数：

```text
robust(x) = sqrt(x^2 + eps^2)
```

或 Huber：

```text
robust(x) =
  0.5 * x^2 / delta, if |x| < delta
  |x| - 0.5 * delta, otherwise
```

优点：

- 不惩罚画面外的机械臂；
- 不惩罚被遮挡或漏分割的渲染区域；
- 适合 partial observation。

风险：

- 如果只使用该项，渲染 mask 过大时可能也能覆盖 `M_obs`，导致解不唯一。

### 6.2 单向 observed-to-render contour Chamfer loss

提取观测 mask 的轮廓：

```text
C_obs = contour(M_obs)
```

提取渲染 mask 的轮廓：

```text
C_ren = contour(M_ren)
```

计算渲染轮廓的距离变换：

```text
D_C_ren = distance_transform(C_ren)
```

定义：

```text
L_obs2ren_contour =
  mean_{p in C_obs_valid} robust(D_C_ren(p))
```

其中 `C_obs_valid` 是去掉图像边界附近假轮廓后的观测轮廓：

```text
C_obs_valid = C_obs ∩ Omega_border_valid
```

边界有效区域：

```text
Omega_border_valid = {p | distance_to_image_border(p) > margin}
```

推荐：

```text
margin = 5 到 20 pixels
```

含义：

真实可见轮廓应该靠近渲染轮廓。由于只做 observed-to-render 单向距离，不惩罚渲染中那些不可见、被遮挡或画面外的轮廓。

优点：

- 对位姿偏移和旋转更敏感；
- 比纯 mask 覆盖更能约束边界对齐。

注意：

- 图像边界上的 mask 截断边缘不是真实机械臂轮廓，必须排除；
- 分割 mask 很粗时，该 loss 权重要降低。

### 6.3 弱反向 render-to-observed regularization

完全单向 loss 可能允许渲染结果过大，只要覆盖观测区域即可。因此可以加入一个弱反向项，但只在可信区域计算。

定义可信可见区域：

```text
Omega_trust =
  image interior
  minus border band
  minus known occluder mask if available
```

定义：

```text
M_ren_trust = M_ren ∩ Omega_trust
D_M_obs = distance_transform(M_obs)

L_ren2obs_weak =
  mean_{p in M_ren_trust} clamp(D_M_obs(p), 0, d_max)
```

推荐：

```text
d_max = 20 到 50 pixels
权重 beta = 0.02 到 0.20
```

这项不是主约束，只用于防止“渲染机械臂无限变大”或严重漂移。

如果遮挡严重，第一版可以关闭该项：

```text
beta = 0
```

### 6.4 位姿先验 loss

如果有初始位姿 `T0_C_B`，建议加弱先验：

```text
L_prior =
  ||t - t0||_2^2 / sigma_t^2
  + ||log(R0^T R)||_2^2 / sigma_R^2
```

推荐：

```text
sigma_t = 0.1 到 0.5 meters
sigma_R = 10 到 30 degrees
```

该项用于避免优化跑到几何上不合理的远处局部最优。

### 6.5 面积比例约束

当单向 loss 约束不足时，可以加弱面积约束：

```text
area_obs = sum(M_obs)
area_ren_inside = sum(M_ren ∩ image)

L_area =
  max(0, area_obs / area_ren_inside - r_max)^2
  + max(0, area_ren_inside / area_obs - r_area_max)^2
```

由于机械臂可能拍不全，面积约束要非常弱。

建议默认只限制极端情况：

```text
r_area_max = 5 到 10
weight_area = 0.001 到 0.01
```

### 6.6 推荐总 loss

第一版纯 RGB、机械臂可能拍不全时，推荐：

```text
L_total =
  1.0  * L_obs2ren_mask
  + 1.0  * L_obs2ren_contour
  + 0.05 * L_ren2obs_weak
  + 0.01 * L_prior
  + 0.001 * L_area
```

如果遮挡明显或 mask 漏分割严重：

```text
L_total =
  1.0  * L_obs2ren_mask
  + 1.0  * L_obs2ren_contour
  + 0.00 * L_ren2obs_weak
  + 0.01 * L_prior
```

如果初始位姿比较准：

```text
提高 L_prior 权重
缩小优化搜索范围
```

如果初始位姿不准：

```text
降低 L_prior 权重
增加多初值搜索
先优化平移和 yaw，再优化完整 6DoF
```

## 7. 初始化策略

优化是否成功很大程度取决于初值。

### 7.1 有人工或历史初值

如果相机和机器人固定，推荐：

```text
第一帧人工给粗略 T0_C_B
后续帧沿用同一个 T_C_B
```

如果是视频序列，可以所有帧共享同一个 `T_C_B` 联合优化。

### 7.2 无初值的单帧粗搜索

在合理空间内采样候选位姿：

```text
t_x, t_y, t_z: 根据相机视野和机械臂可能位置采样
yaw: 0 到 360 degrees 粗采样
roll, pitch: 根据安装方式限制范围
```

流程：

```text
1. 粗采样 100 到 5000 个候选 T
2. 对每个候选低分辨率渲染 mask
3. 用 L_obs2ren_mask + L_obs2ren_contour 排序
4. 取 top-K 候选做连续优化
5. 选择最终 loss 最低的结果
```

### 7.3 利用二维包围框初始化

如果有 `M_obs`：

```text
bbox_obs = bounding_box(M_obs)
```

可以用渲染 bbox 与观测 bbox 的尺度关系粗估深度：

```text
z roughly proportional to projected_model_size / observed_size
```

这只能作为粗初值，不能作为最终结果。

## 8. 优化策略

### 8.1 分阶段优化

推荐 coarse-to-fine：

```text
Stage 1: 低分辨率图像，优化 tx, ty, tz, yaw
Stage 2: 中分辨率图像，优化完整 6DoF
Stage 3: 原分辨率或高分辨率 crop，细化完整 6DoF
```

示例：

```text
resolution pyramid:
  160 x 120
  320 x 240
  640 x 480
```

每个阶段结束后，将当前最优位姿作为下一阶段初值。

### 8.2 优化器选择

如果使用可微渲染器：

```text
Adam: 适合初期粗优化
LBFGS: 适合后期细化
```

如果使用不可微渲染器：

```text
Powell
Nelder-Mead
CMA-ES
coarse grid search + local search
```

推荐第一版：

```text
多初值 coarse search + Powell/Nelder-Mead
```

推荐进阶版：

```text
PyTorch3D/nvdiffrast soft silhouette + Adam + LBFGS
```

### 8.3 参数尺度

旋转和平移的量纲不同，优化时需要归一化：

```text
rotation parameter: radians
translation parameter: meters
```

建议将平移扰动缩放：

```text
delta = [d_rx, d_ry, d_rz, d_tx / s_t, d_ty / s_t, d_tz / s_t]
s_t = 0.1 到 0.5 meters
```

否则优化器可能偏向调整某一类参数。

## 9. 多帧扩展

如果有视频，且相机和机械臂基座相对固定，强烈建议多帧联合优化。

输入：

```text
I_t, q_t, M_obs_t, t = 1...N
```

共享变量：

```text
T_C_B
```

总 loss：

```text
L_video(T_C_B) =
  sum_t w_t * L_frame(I_t, M_obs_t, q_t, T_C_B)
```

其中 `w_t` 可以根据 mask 面积和分割质量设置：

```text
w_t = clamp(area(M_obs_t) / median_area, 0.2, 2.0)
```

优点：

- 单帧被裁切或遮挡时，多帧可以互补；
- 不同关节构型会提供更多几何约束；
- 比单帧更不容易陷入错误位姿。

建议选择帧：

```text
1. 机械臂可见面积较大的帧
2. 关节构型差异明显的帧
3. 遮挡较少的帧
4. 图像模糊较少的帧
```

## 10. 可视化输出

优化完成后，需要生成至少三类可视化。

### 10.1 RGB overlay

将估计位姿下的机械臂渲染到原图：

```text
overlay = alpha_blend(I, rendered_robot_color, alpha=0.4)
```

建议：

```text
渲染机械臂: 半透明青色或绿色
原图: 保持原始颜色
观测 mask 边界: 黄色
渲染 mask 边界: 红色
```

### 10.2 Mask 对比图

输出四宫格：

```text
1. 原始 RGB
2. 观测 mask M_obs
3. 渲染 mask M_ren
4. overlay: M_obs 与 M_ren
```

颜色建议：

```text
观测 mask: 绿色
渲染 mask: 红色
重叠区域: 黄色
```

### 10.3 误差热力图

可选输出：

```text
D_M_ren over M_obs
D_C_ren over C_obs_valid
```

这有助于定位优化失败原因，例如初值错误、mask 漏分割、URDF mesh 尺寸错误。

## 11. 结果评估指标

没有真实位姿标注时，可以用以下指标做自检：

```text
obs2ren_mask_distance: mean distance from M_obs pixels to M_ren
obs2ren_contour_distance: mean distance from C_obs_valid to C_ren
weak_ren2obs_distance: mean distance from trusted M_ren to M_obs
overlap_ratio: area(M_obs ∩ M_ren) / area(M_obs)
area_ratio: area(M_ren ∩ image) / area(M_obs)
```

推荐验收标准：

```text
overlap_ratio 越接近 1 越好
obs2ren_contour_distance 应在几像素到十几像素量级
area_ratio 不应极端大，例如不应长期大于 5 到 10
overlay 中 link 轮廓应与真实图像大体一致
```

如果有标定真值：

```text
translation_error = ||t_est - t_gt||
rotation_error = angle(R_gt^T R_est)
ADD / ADD-S on robot mesh vertices
```

## 12. 伪代码

```python
def estimate_robot_base_pose(
    image_rgb,
    camera_K,
    urdf_path,
    joint_angles,
    init_poses=None,
    robot_mask=None,
):
    robot_model = load_urdf(urdf_path)
    meshes_B = forward_kinematics_meshes(robot_model, joint_angles)

    if robot_mask is None:
        robot_mask = segment_robot(image_rgb)

    obs_contour = extract_contour(robot_mask)
    obs_contour_valid = remove_border_contour(
        obs_contour,
        image_shape=image_rgb.shape[:2],
        margin=10,
    )

    if init_poses is None:
        init_poses = generate_pose_candidates(
            image_rgb=image_rgb,
            mask=robot_mask,
            camera_K=camera_K,
            meshes_B=meshes_B,
        )

    best_result = None

    for T0_C_B in init_poses:
        T_C_B = optimize_pose(
            init_pose=T0_C_B,
            meshes_B=meshes_B,
            camera_K=camera_K,
            robot_mask=robot_mask,
            obs_contour_valid=obs_contour_valid,
            image_shape=image_rgb.shape[:2],
        )

        rendered = render_robot(
            meshes_B=meshes_B,
            camera_K=camera_K,
            T_C_B=T_C_B,
            image_shape=image_rgb.shape[:2],
        )

        score = compute_partial_observation_loss(
            M_obs=robot_mask,
            C_obs_valid=obs_contour_valid,
            M_ren=rendered.mask,
            C_ren=rendered.contour,
            T_C_B=T_C_B,
            T0_C_B=T0_C_B,
        )

        if best_result is None or score < best_result.score:
            best_result = PoseResult(
                T_C_B=T_C_B,
                score=score,
                rendered=rendered,
            )

    overlay = draw_overlay(
        image_rgb=image_rgb,
        rendered_mask=best_result.rendered.mask,
        observed_mask=robot_mask,
        rendered_contour=best_result.rendered.contour,
        observed_contour=obs_contour_valid,
    )

    return best_result.T_C_B, best_result.score, overlay
```

Loss 伪代码：

```python
def compute_partial_observation_loss(
    M_obs,
    C_obs_valid,
    M_ren,
    C_ren,
    T_C_B,
    T0_C_B,
):
    D_M_ren = distance_transform(1 - M_ren)
    D_C_ren = distance_transform(1 - C_ren)

    L_obs2ren_mask = robust_mean(D_M_ren[M_obs > 0])
    L_obs2ren_contour = robust_mean(D_C_ren[C_obs_valid > 0])

    D_M_obs = distance_transform(1 - M_obs)
    trusted_region = make_trusted_region(M_obs.shape, border_margin=10)
    M_ren_trust = M_ren & trusted_region

    if sum(M_ren_trust) > 0:
        L_ren2obs_weak = robust_mean(
            clip(D_M_obs[M_ren_trust > 0], 0, 30)
        )
    else:
        L_ren2obs_weak = 0.0

    L_prior = pose_prior_loss(T_C_B, T0_C_B)
    L_area = weak_area_loss(M_obs, M_ren)

    L_total = (
        1.0 * L_obs2ren_mask
        + 1.0 * L_obs2ren_contour
        + 0.05 * L_ren2obs_weak
        + 0.01 * L_prior
        + 0.001 * L_area
    )

    return L_total
```

## 13. 工程实现建议

### 13.1 第一版最小可行实现

```text
URDF/FK: yourdfpy 或 urdfpy + trimesh
分割: 手工 mask 或 SAM
渲染: pyrender / trimesh / OpenCV polygon rasterization
优化: scipy.optimize.minimize(method="Powell")
可视化: OpenCV alpha blending
```

优点：

- 实现快；
- 不需要可微渲染环境；
- 适合验证 loss 和流程是否可行。

缺点：

- 速度较慢；
- 优化精度依赖多初值；
- 梯度不可用。

### 13.2 进阶实现

```text
URDF/FK: pytorch_kinematics
渲染: PyTorch3D 或 nvdiffrast
优化: Adam + LBFGS
批处理: 多初值、多帧并行
```

优点：

- 可微；
- 支持多帧联合优化；
- 更容易 GPU 加速。

缺点：

- 工程复杂度更高；
- 需要处理 soft rasterization 的数值稳定性。

## 14. 常见失败模式与排查

### 14.1 渲染结果整体尺度不对

可能原因：

```text
URDF mesh scale 错误
相机内参错误
图像 resize 后 K 没同步缩放
单位混用，例如毫米和米混用
```

排查：

```text
渲染已知 T 的机械臂，检查投影尺寸是否合理
打印 mesh bounding box 尺寸
检查 link transform 单位
```

### 14.2 位姿收敛到覆盖整个 mask 的错误解

可能原因：

```text
只用了单向 obs2ren loss
缺少弱反向项或面积约束
初始深度太近，渲染投影过大
```

解决：

```text
加入 L_ren2obs_weak
加入 L_area
限制 z 范围
增加 pose prior
```

### 14.3 轮廓被图像边界吸引

可能原因：

```text
机械臂被裁切，mask 在图像边界产生假轮廓
```

解决：

```text
删除距离图像边界 margin 内的观测轮廓
margin 设置为 5 到 20 pixels
```

### 14.4 某些 link 对不上

可能原因：

```text
joint angle 顺序错误
joint angle 单位错误
mimic joint 未处理
URDF visual mesh 与真实机械臂外观不同
```

排查：

```text
单独渲染每个 link
对比不同 q 下的机械臂姿态是否与图像一致
输出 link 坐标系和 joint axis 可视化
```

## 15. 推荐开发顺序

```text
1. 实现 URDF 加载和给定 q 的机械臂渲染
2. 手工给一个 T_C_B，确认 overlay 的投影方向和坐标系正确
3. 加入图像 mask 读取和轮廓提取
4. 实现 obs2ren mask distance loss
5. 实现 obs2ren contour Chamfer loss，并排除图像边界轮廓
6. 实现单初值优化
7. 加入多初值 coarse search
8. 加入弱反向项和面积约束
9. 输出 overlay、mask 对比和误差热力图
10. 扩展到多帧联合优化
```

## 16. 最终推荐配置

对于“无深度、机械臂可能拍不全”的场景，建议默认配置：

```text
主 loss:
  L_obs2ren_mask
  L_obs2ren_contour

辅助 loss:
  L_ren2obs_weak, 权重 0.05
  L_prior, 权重 0.01
  L_area, 权重 0.001

边界处理:
  忽略距离图像边界 10 px 内的观测轮廓

优化:
  多初值 coarse search
  coarse-to-fine 分辨率金字塔
  第一版用 Powell/Nelder-Mead
  进阶版用 soft silhouette renderer + Adam/LBFGS

可视化:
  原图 + 半透明渲染机械臂
  观测轮廓和渲染轮廓同时显示
  输出 mask overlap 四宫格
```

这套方案的核心优势是：它不需要为每个机械臂单独训练；只要 URDF、关节角、相机内参和图像 mask 可用，就可以通过 render-and-compare 优化得到机械臂基座在相机坐标系下的位姿。同时，loss 设计避免了对图像外、被遮挡或未观测到的机械臂部分进行强惩罚。
