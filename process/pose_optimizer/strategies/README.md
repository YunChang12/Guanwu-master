# Strategies

这个目录放姿态优化器的核心算法实现。统一入口 `process.pose_optimizer.cli` 会根据 `variants.py` 中注册的版本导入这里的模块。

当前版本：

- `baseline.py`：稳定基线版本，原始兼容入口是 `process/optimize_pose_uniform_scale.py`。
- `fast.py`：加速实验版本，原始兼容入口是 `process/optimize_pose_uniform_scale_fast.py`，包含 bbox 预筛选、ROI IoU、profiling、GPU batch 预筛选和 PyTorch3D 相关路径。
- `temporal_fast.py`：跨帧稳定版，复用 `fast.py` 的候选搜索和渲染路径，并加入前序帧 pose prior、截断可见区域评分和边缘辅助评分。

每个策略模块至少需要提供：

```python
def parse_args() -> argparse.Namespace:
    ...

def optimize_sample(args: argparse.Namespace) -> dict[str, Any]:
    ...
```

新增算法版本时，优先把核心实现放在这里，再在 `process/pose_optimizer/variants.py` 和 `process/pose_optimizer/configs/` 中注册对应版本和默认配置。
