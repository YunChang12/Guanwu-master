from __future__ import annotations

from guanwu.video.model_backend.config import ProviderSettings
from guanwu.video.model_backend.providers.sam3 import build_sam3_provider
from guanwu.video.model_backend.providers.sam3d import build_sam3d_provider
from guanwu.video.model_backend.providers.vlm import build_vlm_provider


class ProviderRegistry:
    def __init__(self, settings: ProviderSettings) -> None:
        self.sam3 = build_sam3_provider(settings.sam3)
        self.sam3d = build_sam3d_provider(settings.sam3d)
        self.vlm = build_vlm_provider(settings.vlm)

    def checks(self) -> list[dict]:
        checks: list[dict] = []
        for name, provider in (("sam3", self.sam3), ("sam3d", self.sam3d), ("vlm", self.vlm)):
            ok = True
            detail = "ready"
            if getattr(provider, "mode", "disabled") == "disabled":
                ok = False
                detail = "disabled"
            checks.append({"name": name, "mode": getattr(provider, "mode", "unknown"), "ok": ok, "detail": detail})
        return checks
