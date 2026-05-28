"""Registry for pose optimizer variants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Variant:
    name: str
    module_name: str
    config_path: Path
    description: str


VARIANTS: dict[str, Variant] = {
    "baseline": Variant(
        name="baseline",
        module_name="process.pose_optimizer.strategies.baseline",
        config_path=PACKAGE_DIR / "configs" / "baseline.yaml",
        description="Stable single-file uniform-scale optimizer.",
    ),
    "fast": Variant(
        name="fast",
        module_name="process.pose_optimizer.strategies.fast",
        config_path=PACKAGE_DIR / "configs" / "fast.yaml",
        description="Accelerated optimizer with bbox prefilter, profiling, and optional PyTorch3D/GPU paths.",
    ),
    "temporal_fast": Variant(
        name="temporal_fast",
        module_name="process.pose_optimizer.strategies.temporal_fast",
        config_path=PACKAGE_DIR / "configs" / "temporal_fast_quick.yaml",
        description=(
            "Temporal fast optimizer with previous-frame pose prior, partial visibility handling, "
            "and edge-assisted scoring."
        ),
    ),
    "edge_contour_fast": Variant(
        name="edge_contour_fast",
        module_name="process.pose_optimizer.strategies.edge_contour_fast",
        config_path=PACKAGE_DIR / "configs" / "edge_contour_fast_quick.yaml",
        description=(
            "Edge contour fast optimizer with GrabCut mask enhancement, "
            "temporal prior, and edge-assisted scoring."
        ),
    ),
    "generic_appearance_temporal": Variant(
        name="generic_appearance_temporal",
        module_name="process.pose_optimizer.strategies.generic_appearance_temporal",
        config_path=PACKAGE_DIR / "configs" / "generic_appearance_temporal.yaml",
        description=(
            "Generic optimizer using mask/bbox/contour, image appearance, depth consistency, "
            "and SE(3) temporal smoothness without default road/vehicle hard priors."
        ),
    ),
}
