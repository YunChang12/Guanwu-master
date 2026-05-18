# Pose Optimizer 使用说明

这个目录把姿态优化脚本整理成了“统一入口 + 多个算法版本 + 配置文件”的形式。日常运行建议从项目根目录 `E:\QingYan` 执行：

```powershell
python -m process.pose_optimizer.cli
```

它会根据 `--variant` 或配置文件选择具体策略，再把参数转发给对应的实现模块。

## 1. 这个工具做什么

输入一个样本目录，例如：

```text
sample_dir/
  image.jpg
  mask.png
  object_*.glb
  task.json
```

优化器会读取图像、目标 mask、GLB 模型、相机参数和 `task.json` 中已有的 `corrected_pose`，重新搜索一个更合理的 3D 位姿和统一缩放。目标是让 GLB 在图像上的投影轮廓尽量贴合 `mask.png`，同时让投影 bbox 尽量接近 `task.json` 里的 bbox。

主要输出在 `output_dir` 中：

- `task_with_optimized_corrected_pose.json`：写入新 `corrected_pose` 的任务文件。
- `optimization_report.json`：完整指标、参数、输出路径和版本元数据。
- `optimization_history.csv`：局部搜索过程。
- `01_alignment_overview.png`、`02_pose_inspection.png`、`03_model_reference.png`：核心可视化拼图。
- `04_temporal_edge_debug.png`：`temporal_fast` 开启边缘辅助时的调试图，包含渲染边缘、真实图像边缘、距离场和 temporal prior overlay。
- `05_grabcut_debug.png`：`edge_contour_fast` 开启 GrabCut 时的调试图，包含原始 mask、GrabCut 结果、合并 mask 和差异对比。

## 2. 工程结构

```text
process/
  optimize_pose_uniform_scale.py          # baseline 兼容旧入口
  optimize_pose_uniform_scale_fast.py     # fast 兼容旧入口
  pose_optimizer/
    cli.py                                # 推荐统一入口
    config.py                             # 简单配置读取和配置转命令行参数
    variants.py                           # baseline、fast、temporal_fast、edge_contour_fast 版本注册表
    configs/
      baseline.yaml                       # baseline 默认参数
      fast.yaml                           # fast 默认参数
      fast_quick.yaml                     # fast GPU 快速实验参数
      temporal_fast_quick.yaml            # 跨帧稳定快速实验参数
      edge_contour_fast_quick.yaml        # GrabCut mask 增强快速实验参数
    strategies/
      baseline.py                         # 稳定基线实现
      fast.py                             # 加速实验实现
      temporal_fast.py                    # 跨帧稳定版 fast 实现
      edge_contour_fast.py                # GrabCut mask 增强 + 跨帧稳定版实现
    renderers/                            # 后续拆渲染后端的预留目录
    README.md
```

当前可选版本：

| 版本        | 适合场景               | 核心特点 |
| ---        | ---                    |     --- |
| `baseline` | 稳定对照、确认结果可靠性 | 参数较少，默认 `triangle_fill` / `pyrender` / `auto` 渲染后端。 |
| `fast`     | 日常加速、GPU/PyTorch3D 实验、性能 profiling | 在 baseline 基础上增加 bbox 预筛选、ROI IoU、float32、GPU batch 预筛选、PyTorch3D 渲染和验证参数。 |
| `temporal_fast` | 同一目标连续帧优化、减少后续帧 pose jump | 复用 fast 流程，并加入前序帧 pose prior、截断可见区域处理和边缘辅助评分。 |
| `edge_contour_fast` | mask 质量不高时用 GrabCut 增强 mask、连续帧优化 | 在 temporal_fast 基础上加入 GrabCut 重新分割，与原始 mask 取并集后评分。可选开关 `--enable_grabcut` / `--no-enable_grabcut`。 |

查看已注册版本：

```powershell
python -m process.pose_optimizer.cli --list_variants
```

旧入口仍然能用，但只是薄包装：

```powershell
python process\optimize_pose_uniform_scale.py
python process\optimize_pose_uniform_scale_fast.py
```

新实验建议统一走 `python -m process.pose_optimizer.cli`，因为统一入口会在 `optimization_report.json` 里额外记录 `variant`、配置文件、转发参数、Git commit 和工作区是否 dirty。

工程里还有一个批量脚本 `process/batch_optimize_pose_uniform_scale.py`，它默认逐个样本启动 `process/optimize_pose_uniform_scale_fast.py`。这个批量脚本仍是旧包装器风格，目前没有暴露 `fast` 版本里的所有 PyTorch3D/GPU batch 参数；如果要做最快 GPU 批处理，建议先用本文的单样本命令验证，再扩展批量脚本或写循环调用统一入口。

## 3. 先选运行命令

下面命令都假设当前目录是：

```powershell
cd E:\QingYan
```

样本路径使用：

```text
E:\QingYan\pose_matching_tasks\pose_matching_tasks\obj_000001@000001
```

### 稳定基线

用于和 fast 版本对比，或者先确认整条优化流程是否可靠。

```powershell
python -m process.pose_optimizer.cli `
  --variant baseline `
  --sample_dir "E:\QingYan\pose_matching_tasks\pose_matching_tasks\obj_000001@000006" `
  --output_dir "outputs\obj_000001@000006_baseline"
```

### 默认 fast

默认仍然使用 `triangle_fill` 渲染后端，速度通常比 baseline 更适合日常实验。

```powershell
python -m process.pose_optimizer.cli `
  --variant fast `
  --sample_dir "E:\QingYan\pose_matching_tasks\pose_matching_tasks\obj_000001@000001" `
  --output_dir "outputs\obj_000001@000001_fast"
```

### fast + 耗时统计

用于看每个阶段耗时。正式批处理追求速度时可以不加 `--profile_timings`，因为 profiling 会引入额外同步。

```powershell
python -m process.pose_optimizer.cli `
  --variant fast `
  --sample_dir "E:\QingYan\pose_matching_tasks\pose_matching_tasks\obj_000001@000001" `
  --output_dir "outputs\obj_000001@000001_fast_profile" `
  --profile_timings
```

### 只启用 GPU batch 预筛选

这个开关只把“初始候选位姿的投影和 bbox 预筛选”放到 Torch/CUDA 上批量跑。最终 mask 渲染仍然是默认的 `triangle_fill`，不是 PyTorch3D 渲染。

```powershell
python -m process.pose_optimizer.cli `
  --variant fast `
  --sample_dir "E:\QingYan\pose_matching_tasks\pose_matching_tasks\obj_000001@000001" `
  --output_dir "outputs\obj_000001@000001_fast_gpu_prefilter" `
  --enable_batch_gpu_eval `
  --batch_gpu_size 32 `
  --profile_timings
```

### 只启用 PyTorch3D 渲染

这个命令把最终 silhouette/mask 渲染后端切到 PyTorch3D，并使用 CUDA。它不会自动启用 GPU batch 预筛选。

```powershell
python -m process.pose_optimizer.cli `
  --variant fast `
  --sample_dir "E:\QingYan\pose_matching_tasks\pose_matching_tasks\obj_000001@000001" `
  --output_dir "outputs\obj_000001@000001_fast_pytorch3d" `
  --render_backend pytorch3d `
  --device cuda `
  --profile_timings
```

### 最快 GPU 实验模板

这个模板同时开启 GPU batch 预筛选、PyTorch3D/CUDA 渲染和 float32，是当前 `fast` 版本里 GPU 路径开得最满的单样本命令。实际最快配置仍取决于显卡、模型面数、图像尺寸和 PyTorch3D 安装情况；建议用 `--profile_timings` 做一次对比。

```powershell
python -m process.pose_optimizer.cli `
  --variant fast `
  --sample_dir "E:\QingYan\pose_matching_tasks\pose_matching_tasks\obj_000001@000001" `
  --output_dir "outputs\obj_000001@000001_fast_gpu_pytorch3d" `
  --enable_batch_gpu_eval `
  --batch_gpu_size 32 `
  --render_backend pytorch3d `
  --device cuda `
  --fast_float32 `
  --profile_timings
```

如果要记录耗时，在最后加：

```powershell
  --profile_timings
```

### 快速 GPU 配置

`fast_quick.yaml` 是面向日常快速实验的 PyTorch3D/GPU 配置。它默认开启 GPU batch 预筛选、PyTorch3D/CUDA 渲染和 float32，并把初始 yaw 搜索、精修候选数和局部搜索轮数调得更激进一些，以减少渲染评分次数。

```powershell
python -m process.pose_optimizer.cli `
  --config process\pose_optimizer\configs\fast_quick.yaml `
  --sample_dir "E:\QingYan\pose_matching_tasks\pose_matching_tasks\obj_000001@000002" `
  --output_dir "outputs\obj_000001@000002_fast_quick" `
  --profile_timings
```

如果正式批量处理只追求速度，可以去掉 `--profile_timings`：

```powershell
python -m process.pose_optimizer.cli `
  --config process\pose_optimizer\configs\fast_quick.yaml `
  --sample_dir "E:\QingYan\pose_matching_tasks\pose_matching_tasks\obj_000001@000001" `
  --output_dir "outputs\obj_000001@000001_fast_quick"
```

这个配置相比完整 GPU 模板通常更快，但会少做一些局部精修；如果要最高质量或做最终对照，仍建议使用上一节的完整 GPU 实验模板。

### 跨帧稳定快速配置

`temporal_fast_quick.yaml` 在 `fast_quick.yaml` 的基础上加入了前序帧 pose prior、截断帧可见区域评分和边缘辅助评分。它会从当前 `output_dir` 的父目录里自动搜索同一 `object_id` 的前序输出，例如优先找 `obj_000001@000002_temporal_fast_quick`，再找 `obj_000001@000002_fast_quick` 等目录。

```powershell
python -m process.pose_optimizer.cli `
  --config process\pose_optimizer\configs\temporal_fast_quick.yaml `
  --sample_dir "E:\QingYan\pose_matching_tasks\pose_matching_tasks\obj_000001@000003" `
  --output_dir "outputs\obj_000001@000003_temporal_fast_quick" `
  --profile_timings
```

也可以显式指定 variant，默认会加载已注册的 `temporal_fast_quick.yaml`：

```powershell
python -m process.pose_optimizer.cli `
  --variant temporal_fast `
  --sample_dir "E:\QingYan\pose_matching_tasks\pose_matching_tasks\obj_000001@000004" `
  --output_dir "outputs\obj_000001@000004_temporal_fast_quick" `
  --profile_timings
```

如果没有找到前序结果，`temporal_fast` 会打印提示并退化为普通 fast 风格评分；报告中的 `temporal`、`partial_visibility` 和 `edge_assist` 字段会记录这三类增强是否实际生效。

### edge_contour_fast 带 GrabCut mask 增强

`edge_contour_fast_quick.yaml` 在 `temporal_fast_quick.yaml` 的基础上加入了 GrabCut mask 增强。当提供的 `mask.png` 质量不高（例如没有完全覆盖车辆轮廓）时，GrabCut 会利用真实图像在 bbox 区域内重新分割前景，然后与原始 mask 取并集，得到更完整的目标 mask 用于评分。

```powershell
python -m process.pose_optimizer.cli `
  --config process\pose_optimizer\configs\edge_contour_fast_quick.yaml `
  --sample_dir "E:\QingYan\pose_matching_tasks\pose_matching_tasks\obj_000001@000001" `
  --output_dir "outputs\obj_000001@000001_edge_contour_fast_quick" `
  --profile_timings
```

也可以显式指定 variant：

```powershell
python -m process.pose_optimizer.cli `
  --variant edge_contour_fast `
  --sample_dir "E:\QingYan\pose_matching_tasks\pose_matching_tasks\obj_000002@000001" `
  --output_dir "outputs\obj_000002@000001_edge_contour_fast_quick" `
  --profile_timings
```

如果不需要 GrabCut 增强（只使用原始 mask），可以在 yaml 中设 `enable_grabcut: false` 或命令行加 `--no-enable_grabcut`。此时 `edge_contour_fast` 退化为 `temporal_fast`，但仍保留时序 prior、截断处理和边缘辅助。

连续帧批量运行示例（obj_000001 帧 001-008）：

```powershell
foreach ($i in 1..8) {
  $f = "{0:D6}" -f $i
  python -m process.pose_optimizer.cli `
    --config process\pose_optimizer\configs\edge_contour_fast_quick.yaml `
    --sample_dir "E:\QingYan\pose_matching_tasks\pose_matching_tasks\obj_000001@$f" `
    --output_dir "outputs\obj_000001@${f}_edge_contour_fast_quick" `
    --profile_timings
}
```

`temporal_search_output_suffixes` 默认搜索顺序为 `_edge_contour_fast_quick,_temporal_fast_quick,_fast_quick,_fast,_baseline`，因此 `edge_contour_fast` 在连续帧优化时会优先找同版本的前序结果。

### edge_contour_fast 的 GrabCut 增强流程

1. 读取 `mask.png`，优先用白色前景外接框对齐 `bbox_xyxy` 放回原图坐标；当前景框不适合对齐时，再回退到 `crop.jpg` template matching 或 bbox resize。
2. 以 `bbox_xyxy` 外扩 `grabcut_margin` 像素作为 GrabCut 前景框，原始 mask 初始化为 probable foreground。
3. 运行 `cv2.grabCut()` 得到 GrabCut 分割结果。
4. 将 GrabCut mask 与原始 mask 取并集（`mask_merge_mode: union`），得到增强后的目标 mask。
5. 安全回退：GrabCut 异常、结果面积过小（< 原始 20%）或过大（> bbox 面积 95%）时自动回退到原始 mask。

输出 `05_grabcut_debug.png` 中各面板含义：

| 面板 | 颜色 | 含义 |
| --- | --- | --- |
| `original mask` | 绿色半透明 | 原始 `mask.png` 放回原图后的覆盖区域。 |
| `grabcut result` | 黄色 = 重叠，橙色 = GrabCut 新增 | GrabCut 分割结果与原始 mask 的关系。 |
| `merged mask` | 橙黄色半透明 | 取并集后最终用于评分的 mask。 |
| `diff` | 黄色 = 重叠，绿色 = 新增，红色 = 移除 | 增强前后 mask 的像素级差异。 |

`optimization_report.json` 中 `grabcut` 字段：

| 字段 | 含义 |
| --- | --- |
| `grabcut.enabled` | 是否启用 GrabCut 增强。 |
| `grabcut.grabcut_succeeded` | GrabCut 是否成功执行。 |
| `grabcut.fallback_used` | 是否回退到原始 mask。 |
| `grabcut.fallback_reason` | 回退原因（异常、面积过小、面积过大等）。 |
| `grabcut.original_mask_area_px` | 原始 mask 面积。 |
| `grabcut.grabcut_mask_area_px` | GrabCut 分割面积。 |
| `grabcut.merged_mask_area_px` | 合并后面积。 |
| `grabcut.area_change_ratio` | 合并面积 / 原始面积。 |

### temporal_fast 算法更新记录

本次新增 `process.pose_optimizer.strategies.temporal_fast`，目标是解决连续帧车辆优化时容易出现的 yaw、depth、scale 突变问题，同时处理后续帧车辆接近图像边界时的部分可见问题，并用图像边缘给 silhouette 评分补充一个更稳定的辅助约束。

`temporal_fast` 没有推翻 `fast` 的主流程。它仍然复用 fast 的 coarse candidate search、bbox prefilter、ROI IoU、PyTorch3D/CUDA 渲染、top-K refinement、三阶段局部搜索和报告输出，只在候选初始化和评分层增加三类信息：

1. 跨帧 prior：从当前 `output_dir` 的父目录向前搜索同一 `object_id` 的历史结果，优先读取最近一帧的 `task_with_optimized_corrected_pose.json`。找到 prior 后，它会作为一个明确的 `temporal_prior` 初始候选进入 top-K 精修，而不只是最后加惩罚。
2. 时序连续性评分：对当前候选与 prior 的平移、深度、旋转、yaw/pitch/roll 和 uniform scale 变化计算归一化二次损失，再转成 `temporal_score = exp(-temporal_loss)`。最终分数会在几何分数基础上加入小权重的 temporal 奖励，默认 `temporal_weight = 0.15`。
3. 截断与边缘辅助：先检测 mask/bbox 是否接触图像边界。如果是截断帧，就降低边界带附近 rendered mask 外溢的惩罚；同时从真实 `image.jpg` 提取 Canny 边缘，对 rendered silhouette 边缘查询距离场，得到 `edge_score`，默认小权重 `edge_weight = 0.08`。

整体评分可以理解为：

```text
base_geometry_score = soft_mask_iou - bbox_weight * (1 - bbox_iou)
final_score = base_geometry_score
            + temporal_weight * temporal_score
            + edge_weight * edge_score

如果检测到截断，则用 adjusted_mask_score 修正 base_geometry_score。
```

这次对 `obj_000001@000003` 的实测输出目录是：

```text
E:\QingYan\outputs\obj_000001@000003_temporal_fast_quick
```

关键结果：

| 指标 | 数值 |
| --- | --- |
| prior 来源 | `obj_000001@000002_fast_quick\task_with_optimized_corrected_pose.json` |
| `best_started_from_temporal_seed` | `true` |
| `mask_iou` | `0.950126` |
| `bbox_iou` | `0.991424` |
| `bbox_center_error_px` | `0.352744` |
| `base_geometry_score` | `0.826623` |
| `final_score` | `1.030206` |
| `temporal_score` / `temporal_loss` | `0.986211` / `0.013885` |
| `delta_translation_norm` | `0.014135 m` |
| `delta_depth` | `0.008750 m` |
| `delta_rotation_deg` / `delta_yaw_deg` | `1.656324 deg` / `0.500000 deg` |
| `delta_scale_log` | `-0.007500` |
| `is_truncated` | `false` |
| `edge_score` / `edge_mean_distance_px` | `0.695638` / `1.683172 px` |

这说明第 3 帧确实找到了第 2 帧 prior，并且最终结果从 temporal seed 出发，优化后的位姿变化很小：平移约 `1.4 cm`，深度变化约 `0.9 cm`，整体旋转约 `1.66°`，yaw 约 `0.5°`，scale 也基本稳定。当前第 3 帧没有被判定为截断帧，所以 partial visibility 逻辑只记录状态，没有替换 mask score；边缘辅助可用，并参与最终评分。

`04_temporal_edge_debug.png` 中各面板含义：

| 面板 | 含义 |
| --- | --- |
| `edge overlay` | 当前图像上的边缘调试叠加。绿色是真实图像边缘，红色是优化后渲染 silhouette 边缘。两者贴近说明边缘辅助约束有效。 |
| `rendered edge` | 从最终 rendered mask 提取出的轮廓边缘。 |
| `image edge` | 从真实图像 ROI 中提取出的 Canny 边缘。 |
| `edge distance` | 真实边缘的距离场。颜色越靠近车身轮廓的低距离区域，渲染边缘评分越高。 |
| `temporal prior` | 上一帧优化位姿直接投到当前帧的 overlay，用于直观看 prior 是否合理。 |

### 检查 PyTorch3D 渲染是否对齐

用于比较同一最终位姿下 `triangle_fill` 和 PyTorch3D 的 mask。输出会包含 `render_check_triangle_fill.png`、`render_check_pytorch3d.png` 和 `render_check_pytorch3d_overlay.png`。

```powershell
python -m process.pose_optimizer.cli `
  --variant fast `
  --sample_dir "E:\QingYan\pose_matching_tasks\pose_matching_tasks\obj_000001@000001" `
  --output_dir "outputs\obj_000001@000001_fast_pytorch3d_check" `
  --render_backend pytorch3d `
  --device cuda `
  --validate_pytorch3d_alignment `
  --profile_timings
```

## 4. GPU 和渲染后端的关系

这几个参数很容易混在一起，可以按下面理解：

| 参数                          | 真正影响的阶段               | 说明  |
| ---                           | ---                         | --- |
| `--enable_batch_gpu_eval`     | 初始候选预筛选               | 用 Torch/CUDA 批量投影候选位姿，计算 bbox 指标并过滤明显不靠谱的候选。 |
| `--render_backend pytorch3d`  | 最终 mask 渲染和局部精修评分 | 使用 PyTorch3D rasterizer 生成 silhouette/mask，会影响优化评分和最终结果。 |
| `--device cuda`               | Torch 相关路径              | 给 PyTorch3D 和 GPU batch 预筛选选择设备；单独写它不会自动启用 GPU 功能。 |
| `--batch_gpu_size 32`         | GPU batch 预筛选批大小       | 批越大可能越快，但更吃显存；CUDA OOM 时脚本会自动减半重试。 |
| `--fast_float32`              | fast evaluator 数值缓冲      | 降低内存和计算开销，结果可能和 float64 有细微差异。 |

所以：

- 只写 `--enable_batch_gpu_eval`：GPU 用在初始候选预筛选，渲染还是默认 `triangle_fill`。
- 只写 `--render_backend pytorch3d --device cuda`：GPU 用在 PyTorch3D 渲染，不自动启用 batch 预筛选。
- 两组参数都写：预筛选和渲染都走 GPU 相关路径。

`--render_backend pytorch3d` 需要当前 Python 环境已经安装可用的 PyTorch 和 PyTorch3D，并且 CUDA 配置正确。如果 CUDA 不可用，显式 `--device cuda` 会报错；`--render_backend auto` 在部分路径中可以退回 CPU 可用的后端。

## 5. 配置文件和参数覆盖规则

默认配置：

- `--variant baseline` 加载 `process/pose_optimizer/configs/baseline.yaml`
- `--variant fast` 加载 `process/pose_optimizer/configs/fast.yaml`
- `--variant temporal_fast` 加载 `process/pose_optimizer/configs/temporal_fast_quick.yaml`
- `--variant edge_contour_fast` 加载 `process/pose_optimizer/configs/edge_contour_fast_quick.yaml`

实际转发参数的顺序是：

```text
默认配置文件参数 + 命令行额外参数
```

后面的命令行参数会覆盖前面的配置。例如：

```powershell
python -m process.pose_optimizer.cli `
  --variant fast `
  --sample_dir "E:\QingYan\pose_matching_tasks\pose_matching_tasks\obj_000001@000001" `
  --output_dir "outputs\override_example" `
  --top_k_candidates 12 `
  --profile_timings
```

即使 `fast.yaml` 里是 `top_k_candidates: 8`，这次运行也会使用 `12`。

自定义配置：

```powershell
Copy-Item `
  process\pose_optimizer\configs\fast.yaml `
  process\pose_optimizer\configs\fast_debug.yaml
```

```powershell
python -m process.pose_optimizer.cli `
  --config process\pose_optimizer\configs\fast_debug.yaml `
  --sample_dir "E:\QingYan\pose_matching_tasks\pose_matching_tasks\obj_000001@000001" `
  --output_dir "outputs\obj_000001@000001_fast_debug"
```

预览最终会转发哪些参数，不实际运行：

```powershell
python -m process.pose_optimizer.cli `
  --variant fast `
  --dry_run `
  --sample_dir "E:\QingYan\pose_matching_tasks\pose_matching_tasks\obj_000001@000001" `
  --output_dir "outputs\dry_run"
```

配置文件只支持简单的扁平 `key: value` 写法，也支持 JSON。YAML 示例：

```yaml
variant: fast
render_backend: triangle_fill
device: cuda
enable_batch_gpu_eval: false
batch_gpu_size: 32
top_k_candidates: 8
init_scale_factors: "0.5,0.7,1.0,1.3,1.6"
```

布尔参数规则：

- `true` 会转成 `--参数名`。
- `false` 通常不会转发。
- `include_corrected_seed: false` 会转成 `--no-include_corrected_seed`。
- `enable_bbox_prefilter: false` 会转成 `--no-enable_bbox_prefilter`。

## 6. 参数速查

### 统一入口参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--variant` | 由配置决定，否则 `baseline` | 选择算法版本：`baseline`、`fast`、`temporal_fast` 或 `edge_contour_fast`。 |
| `--config` | 对应版本默认配置 | 指定自定义 YAML/JSON 配置文件。 |
| `--list_variants` | 关闭 | 列出已注册版本后退出。 |
| `--dry_run` | 关闭 | 只打印解析后的版本和转发参数，不运行优化。 |
| `--no_default_config` | 关闭 | 不加载版本默认配置，只使用命令行显式参数。 |

### 输入输出和渲染

| 参数 | 默认值 | 版本 | 作用 |
| --- | --- | --- | --- |
| `--sample_dir` | 示例样本路径 | baseline / fast / temporal_fast | 输入样本目录。 |
| `--output_dir` | `outputs/<sample>_pose_optimized_uniform_scale` | baseline / fast / temporal_fast | 输出目录。相对路径会基于当前工作目录解析。 |
| `--render_backend` | `triangle_fill` | baseline / fast / temporal_fast | 渲染后端。baseline 支持 `triangle_fill`、`pyrender`、`auto`；fast 和 temporal_fast 额外支持 `pytorch3d`。 |
| `--device` | `cuda` | fast / temporal_fast | Torch/PyTorch3D 使用的设备，例如 `cuda`、`cuda:0`、`cpu`。 |
| `--pytorch3d_faces_per_pixel` | `1` | fast / temporal_fast | PyTorch3D rasterizer 每个像素保留的面数量；silhouette 通常保持 `1`。 |
| `--pytorch3d_cull_backfaces` | `false` | fast / temporal_fast | PyTorch3D 是否剔除背面；可能更快，但可能改变 mask。 |

### GPU、profiling 和验证

| 参数 | 默认值 | 版本 | 作用 |
| --- | --- | --- | --- |
| `--enable_batch_gpu_eval` | 关闭 | fast | 用 Torch/CUDA 批量投影初始候选并做 bbox 预筛选。 |
| `--batch_gpu_size` | `32` | fast | GPU batch 初始批大小；显存不足时自动减半重试。 |
| `--fast_float32` | 关闭 | fast | evaluator 侧使用 float32 缓冲和 soft mask。 |
| `--profile_timings` | 关闭 | fast | 收集并打印耗时统计，也写入 `optimization_report.json`。 |
| `--validate_render_backends` | 关闭 | fast | 保存 `triangle_fill` 与 `pyrender` 的最终位姿 mask 和 overlay。 |
| `--validate_pytorch3d_alignment` | 关闭 | fast | 保存 `triangle_fill` 与 PyTorch3D 的最终位姿 mask 和 overlay。 |
| `--save_full_history` | 关闭 | fast | 保存完整局部搜索试探历史；便于调试，但输出更大。 |

`temporal_fast` 继承 `fast` 的 GPU、profiling、候选搜索、bbox 预筛选和局部精修参数；下表中标为 `fast` 的参数一般也可用于 `temporal_fast`。

### 初始候选搜索

| 参数 | 默认值 | 版本 | 作用 |
| --- | --- | --- | --- |
| `--include_corrected_seed` / `--no-include_corrected_seed` | 开启 | baseline / fast | 是否把 `task.json` 中已有 `corrected_pose` 也作为初始化候选。 |
| `--world_up_axis` | `y` | baseline / fast | 世界坐标中作为竖直方向的轴，用于构造 upright 初始化假设。 |
| `--proxy_face_count` | `1800` | baseline / fast | 生成快速评估代理 mesh 的目标面数。越小越快但越粗。 |
| `--top_k_candidates` | `8` | baseline / fast | 粗搜索后保留多少个候选。越大越稳但越慢。 |
| `--refine_top_k` | `3` | baseline / fast | 进入局部精修的候选数量。越大越稳但越慢。 |
| `--init_yaw_step_deg` | `15.0` | baseline / fast | 初始 yaw 搜索步长。越小搜索越密、越慢。 |
| `--init_scale_factors` | `0.5,0.7,1.0,1.3,1.6` | baseline / fast | 粗搜索尝试的 uniform scale 倍数。 |
| `--init_depth_factors` | `0.8,1.0,1.2` | baseline / fast | 围绕 bbox 推导深度尝试的深度倍数。 |

### 评分和预筛选

| 参数 | 默认值 | 版本 | 作用 |
| --- | --- | --- | --- |
| `--bbox_weight` | `0.1` | baseline / fast | bbox IoU 在总分中的约束权重；调大可减少 silhouette-only 漂移。 |
| `--enable_bbox_prefilter` / `--no-enable_bbox_prefilter` | 开启 | fast | 渲染前跳过明显不合理的 bbox 候选。 |
| `--prefilter_bbox_iou_min` | `0.05` | fast | bbox 预筛选的最小 IoU 阈值。 |
| `--prefilter_center_factor` | `1.8` | fast | bbox 中心距离阈值，按目标 bbox 对角线倍数计算。 |
| `--prefilter_size_ratio_min` | `0.35` | fast | 投影 bbox 相对目标 bbox 的最小尺寸比例。 |
| `--prefilter_size_ratio_max` | `3.0` | fast | 投影 bbox 相对目标 bbox 的最大尺寸比例。 |
| `--roi_iou_margin` | `30` | fast | 计算 mask IoU 时，在目标和投影 bbox 并集外扩的像素边距。 |
| `--disable_roi_iou` | 关闭 | fast | 关闭 ROI 限制，改用整张图计算 IoU。 |

优化器内部主要分数：

```text
score = soft_mask_iou - bbox_weight * (1 - bbox_iou)
```

`temporal_fast` 额外参数：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--temporal_enabled` / `--no-temporal_enabled` | 开启 | 是否启用历史帧 prior 搜索和时序连续性评分。 |
| `--temporal_lookback` | `5` | 最多向前搜索多少帧历史输出。 |
| `--temporal_seed_enabled` / `--no-temporal_seed_enabled` | 开启 | 是否把历史帧 pose 作为初始化候选加入 top-K。 |
| `--temporal_weight` | `0.15` | temporal score 加到最终分数中的权重。 |
| `--temporal_translation_sigma` / `--temporal_depth_sigma` | `0.35` / `0.45` | 平移和深度突变惩罚的尺度。 |
| `--temporal_rotation_sigma_deg` / `--temporal_yaw_sigma_deg` | `20.0` / `15.0` | 旋转和 yaw 突变惩罚的尺度。 |
| `--temporal_scale_sigma` | `0.12` | uniform scale 对数变化惩罚的尺度。 |
| `--temporal_search_output_suffixes` | `_temporal_fast_quick,_fast_quick,_fast,_baseline` | 搜索前序输出目录时尝试的后缀顺序。 |
| `--partial_visibility_enabled` / `--no-partial_visibility_enabled` | 开启 | 是否启用截断检测和可见区域 mask score 修正。 |
| `--ignore_truncated_border_band_px` | `8` | 截断边界附近忽略或降低惩罚的像素带宽。 |
| `--edge_score_enabled` / `--no-edge_score_enabled` | 开启 | 是否启用图像边缘辅助评分。 |
| `--edge_weight` | `0.08` | edge score 加到最终分数中的权重。 |
| `--edge_canny_low` / `--edge_canny_high` | `50` / `150` | Canny 边缘提取阈值。 |
| `--edge_distance_sigma_px` | `4.0` | rendered edge 到真实边缘距离转换成分数的尺度。 |
| `--edge_roi_margin` | `20` | 边缘评分 ROI 在目标 bbox 外扩的像素边距。 |

### 局部精修

| 参数 | 默认值 | 版本 | 作用 |
| --- | --- | --- | --- |
| `--stage1_iters` | `10` | baseline / fast | coarse 阶段最大迭代次数。 |
| `--stage2_iters` | `8` | baseline / fast | rotation 阶段最大迭代次数。 |
| `--stage3_iters` | `14` | baseline / fast | fine 阶段最大迭代次数。 |
| `--step_decay` | `0.5` | baseline / fast | 当前步长无提升时的衰减系数。 |
| `--max_translation_delta` | `0.8` | baseline / fast | 局部搜索中平移相对初始值的最大偏移。 |
| `--max_rotation_delta_deg` | `45.0` | baseline / fast | 局部搜索中旋转相对初始值的最大角度。 |
| `--scale_min_factor` | `0.5` | baseline / fast | uniform scale 相对初始 scale 的最小倍数。 |
| `--scale_max_factor` | `2.2` | baseline / fast | uniform scale 相对初始 scale 的最大倍数。 |
| `--early_stop_mask_iou` | `0.90` | baseline / fast | mask IoU 达到该值时可提前停止。 |
| `--early_stop_bbox_iou` | `0.85` | baseline / fast | bbox IoU 达到该值时配合 mask IoU 用于提前停止。 |

## 7. 输出怎么看

重点文件：

| 文件 | 用途 |
| --- | --- |
| `optimization_report.json` | 最完整的机器可读报告。统一入口运行时会额外写入 `run_metadata`。 |
| `task_with_optimized_corrected_pose.json` | 可继续用于后续验证或处理的新 task。 |
| `optimization_history.csv` | 每次局部搜索试探的参数、分数和指标。 |
| `01_alignment_overview.png` | mask/bbox 对齐总览。 |
| `02_pose_inspection.png` | 图像上的近景投影检查。 |
| `03_model_reference.png` | GLB 模型自身形状和参考视图。 |
| `04_temporal_edge_debug.png` | `temporal_fast` 的边缘和前序位姿调试图。 |
| `render_check_*.png` | 只有开启验证参数时生成，用于比较不同渲染后端。 |

`optimization_report.json` 中常看的字段：

| 字段 | 含义 |
| --- | --- |
| `metrics.mask_iou` | 最终渲染 mask 与目标 mask 的 IoU，越高越贴合。 |
| `metrics.bbox_iou` | 投影 bbox 与标注 bbox 的 IoU，越高越贴合。 |
| `metrics.bbox_center_error_px` | 投影 bbox 中心与标注 bbox 中心的像素距离，越小越好。 |
| `render_backend` | 本次实际使用或偏好的渲染后端。 |
| `best_candidate_rank` | 最终最优结果来自初始候选列表中的排名。 |
| `profiling` | 开启 `--profile_timings` 后出现，记录投影、渲染、IoU 等耗时。 |
| `temporal` | `temporal_fast` 的前序帧 prior、时序变化量和 temporal score。 |
| `partial_visibility` | `temporal_fast` 的截断检测和可见区域 mask score 记录。 |
| `edge_assist` | `temporal_fast` 的边缘辅助评分、平均边缘距离和 ROI。 |
| `run_metadata.variant` | 统一入口记录的版本名。 |
| `run_metadata.forwarded_args` | 配置文件和命令行合并后实际转发给策略模块的参数。 |
| `run_metadata.git_commit` / `git_dirty` | 运行时的代码版本信息。 |

输出目录建议带上样本名和实验配置：

```text
outputs/obj_000001@000001_baseline
outputs/obj_000001@000001_fast
outputs/obj_000001@000001_fast_profile
outputs/obj_000001@000001_fast_gpu_prefilter
outputs/obj_000001@000001_fast_pytorch3d
outputs/obj_000001@000001_fast_gpu_pytorch3d
outputs/obj_000001@000003_temporal_fast_quick
```

## 8. 常见问题

### 我应该优先跑哪个版本？

先跑 `baseline` 做可靠性对照，再跑默认 `fast` 看速度和结果是否一致。确认 PyTorch3D 渲染对齐后，再尝试“最快 GPU 实验模板”。

### `--device cuda` 是否等于开启 GPU？

不是。`--device cuda` 只是告诉 Torch 路径使用 CUDA。是否真的走 GPU，取决于是否同时开启 `--enable_batch_gpu_eval` 或 `--render_backend pytorch3d`。

### PyTorch3D 报错怎么办？

先确认当前 Python 环境能 `import torch`、`import pytorch3d`，并且 `torch.cuda.is_available()` 为 `True`。如果只想继续优化，不依赖 PyTorch3D，可以退回：

```powershell
--render_backend triangle_fill
```

### CUDA OOM 怎么办？

先减小：

```powershell
--batch_gpu_size 16
```

如果仍然 OOM，可以继续降到 `8` 或 `4`，或者去掉 `--enable_batch_gpu_eval`。PyTorch3D 渲染 OOM 时，也可以降低 mesh 复杂度相关参数，例如 `--proxy_face_count`，但最终 full mesh 渲染仍可能受原始模型复杂度影响。

### 为什么 mask IoU 高但 3D 位姿看起来仍可能不对？

单张图像的 2D silhouette 对 3D 姿态约束有限。不同深度、旋转和 scale 组合可能产生相似轮廓。遇到这种情况，需要结合 bbox、车辆朝向、真实尺寸、地面约束或多视角信息判断。

### 还能用旧脚本吗？

可以：

```powershell
python process\optimize_pose_uniform_scale.py --sample_dir <sample>
python process\optimize_pose_uniform_scale_fast.py --sample_dir <sample>
```

但推荐使用统一入口，因为它会自动记录版本和配置元数据，之后更容易复现实验。

## 9. 新增算法版本

假设要新增 `depth` 版本：

1. 新建策略文件：

```text
process/pose_optimizer/strategies/depth.py
```

策略文件至少需要提供：

```python
def parse_args() -> argparse.Namespace:
    ...

def optimize_sample(args: argparse.Namespace) -> dict[str, Any]:
    ...
```

2. 新建默认配置：

```text
process/pose_optimizer/configs/depth.yaml
```

示例：

```yaml
variant: depth
render_backend: triangle_fill
top_k_candidates: 8
refine_top_k: 3
profile_timings: true
```

3. 在 `process/pose_optimizer/variants.py` 注册：

```python
"depth": Variant(
    name="depth",
    module_name="process.pose_optimizer.strategies.depth",
    config_path=PACKAGE_DIR / "configs" / "depth.yaml",
    description="Depth-guided pose optimizer experiment.",
),
```

4. 检查注册结果：

```powershell
python -m process.pose_optimizer.cli --list_variants
```

5. 运行：

```powershell
python -m process.pose_optimizer.cli `
  --variant depth `
  --sample_dir "E:\QingYan\pose_matching_tasks\pose_matching_tasks\obj_000001@000001" `
  --output_dir "outputs\obj_000001@000001_depth"
```

## 10. 推荐工作流

稳定代码放在 `main` 分支。做新实验时开分支：

```powershell
git checkout -b exp/depth-guided-pose
```

一次可复现实验完成后提交：

```powershell
git add process
git commit -m "feat: add depth-guided pose optimizer variant"
```

如果结果重要，可以打 tag：

```powershell
git tag pose-opt-v0.3-depth-guided
```

经验规则：

```text
历史快照       -> git commit / git tag
临时实验       -> git branch
算法变体       -> configs/*.yaml + variants.py
参数变化       -> configs/*.yaml 或命令行覆盖
输出结果       -> outputs/<样本名>_<版本名>_<关键参数>
```
