from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

from guanwu.video.model_backend.config import SAM3DProviderConfig
from guanwu.video.model_backend.schemas import FrameDetectionsModel
from guanwu.video.model_backend.providers.base import post_json


class DisabledSAM3DProvider:
    mode = "disabled"

    def reconstruct_object_meshes(self, detections: FrameDetectionsModel, objects: list[dict[str, Any]]) -> dict[str, dict]:
        _ = (detections, objects)
        return {}


class EmbeddedSAM3DProvider:
    mode = "embedded"

    def __init__(self, cfg: SAM3DProviderConfig) -> None:
        self.backend = cfg.backend.strip().lower()
        self.output_dir = Path(cfg.output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.object_command = cfg.object_command.strip()
        self.body_command = cfg.body_command.strip()
        self.min_mesh_quality = float(cfg.min_mesh_quality)

    def reconstruct_object_meshes(self, detections: FrameDetectionsModel, objects: list[dict[str, Any]]) -> dict[str, dict]:
        segment_map = {inst.track_id: self._normalize_segment_kind(inst.segment_kind) for inst in detections.instances}
        if self._normalize_segment_kind and self.backend == "command":
            return self._command_reconstruct(detections, objects, segment_map)
        raise NotImplementedError("SAM3D embedded backend only supports 'command' backend mode. 'mock' mode has been removed.")

    def _normalize_segment_kind(self, segment_kind: str) -> str:
        return "body" if str(segment_kind).strip().lower() == "body" else "object"



    def _command_reconstruct(
        self,
        detections: FrameDetectionsModel,
        objects: list[dict[str, Any]],
        segment_map: dict[str, str],
    ) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for obj in objects:
            track_id = str(obj.get("track_id", ""))
            object_id = str(obj.get("object_id", track_id or "obj_unknown"))
            segment_kind = segment_map.get(track_id, "object")
            command = self.object_command if segment_kind == "object" else self.body_command
            if not command:
                raise RuntimeError(f"Missing SAM3D command for segment_kind={segment_kind}")

            payload = {
                "frame_idx": detections.frame_idx,
                "timestamp": detections.timestamp,
                "image_b64": detections.image_b64,
                "object": obj,
                "segment_kind": segment_kind,
                "output_dir": str(self.output_dir),
            }
            proc = subprocess.run(
                shlex.split(command),
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"SAM3D command failed for {object_id}: {proc.stderr.strip()}")

            try:
                data = json.loads(proc.stdout)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"SAM3D command output is not JSON for {object_id}") from exc
            if not isinstance(data, dict):
                raise RuntimeError(f"SAM3D command output must be JSON object for {object_id}")

            data.setdefault("instance_id", object_id)
            data.setdefault("segment_kind", segment_kind)
            data.setdefault("source_command", command)
            out[object_id] = data
        return out

    def _get_scale_3d(self, obj: dict[str, Any]) -> list[float]:
        geom = obj.get("geometry") if isinstance(obj.get("geometry"), dict) else {}
        raw = geom.get("scale_3d") if isinstance(geom, dict) else None
        if isinstance(raw, list) and len(raw) >= 3:
            return [float(raw[0]), float(raw[1]), float(raw[2])]
        return [0.1, 0.1, 0.1]




class HttpSAM3DProvider:
    def __init__(
        self,
        base_url: str,
        timeout_sec: float,
        mode: str,
        output_dir: str = "sam3d_meshes",
        min_mesh_quality: float = 0.6,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.mode = mode
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_mesh_quality = min_mesh_quality
        self._log = __import__("logging").getLogger("spwm_agent.sam3d")

    def _http(self, method: str, path: str, *, json: dict | None = None) -> dict:
        """Simple HTTP helper using urllib (no extra deps)."""
        import json as _json
        import urllib.request
        import urllib.error

        url = self.base_url + path
        body = _json.dumps(json).encode() if json is not None else None
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                return _json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode(errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {url}: {body_text}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"HTTP request failed for {url}: {exc.reason}") from exc

    def _download_file(self, download_url: str, dest: Path) -> None:
        """Download a file from the sam3d service's /files/{id} endpoint."""
        import urllib.request
        url = self.base_url + download_url if download_url.startswith("/") else download_url
        urllib.request.urlretrieve(url, str(dest))  # noqa: S310

    def reconstruct_object_meshes(self, detections: FrameDetectionsModel, objects: list[dict[str, Any]]) -> dict[str, dict]:
        """Call the sam3d HTTP service per-object and return mesh paths keyed by object_id."""
        if not detections.image_b64:
            return {}

        out: dict[str, dict] = {}
        for obj in objects:
            object_id = str(obj.get("object_id", obj.get("track_id", "obj_unknown")))
            segment_kind = str(obj.get("segment_kind", "object")).strip().lower()
            if segment_kind not in {"object", "body"}:
                segment_kind = "object"

            # Build per-object mask from bbox (full-frame fallback = all-ones 1×1)
            mask = self._bbox_to_mask(obj, detections)

            try:
                if segment_kind == "body":
                    endpoint = "/predict/body"
                    payload: dict[str, Any] = {"image_base64": detections.image_b64}
                else:
                    endpoint = "/predict/objects"
                    payload = {"image_base64": detections.image_b64, "mask": mask}

                resp = self._http("POST", endpoint, json=payload)

                # Download the first available mesh file
                files = resp.get("files") or []
                if not files:
                    self._log.warning("SAM3D %s returned no files for %s", endpoint, object_id)
                    continue

                # Prefer .ply, then .glb
                chosen = next((f for f in files if f.get("format") == "ply"), files[0])
                ext = chosen.get("format", "bin")
                dest = self.output_dir / f"{object_id}_{resp.get('request_id', 'mesh')}.{ext}"
                self._download_file(chosen["download_url"], dest)

                out[object_id] = {
                    "instance_id": object_id,
                    "mesh_path": str(dest),
                    "quality": self.min_mesh_quality,
                    "segment_kind": segment_kind,
                    "source": "sam3d_http",
                }
                self._log.info("SAM3D mesh saved for %s → %s", object_id, dest)

            except Exception as exc:
                self._log.warning("SAM3D HTTP reconstruct failed for %s: %s. Skipping.", object_id, exc)

        return out

    def _bbox_to_mask(self, obj: dict[str, Any], detections: FrameDetectionsModel) -> list[list[float]]:
        """Convert a bounding-box to a proper binary mask matching image dimensions."""
        import base64
        from io import BytesIO
        from PIL import Image
        import numpy as np

        if not detections.image_b64:
            return [[1.0]]

        img_data = detections.image_b64
        if "," in img_data:
            img_data = img_data.split(",", 1)[1]
        img = Image.open(BytesIO(base64.b64decode(img_data)))
        w, h = img.size

        mask = np.zeros((h, w), dtype=np.float32)
        geom = obj.get("geometry")
        bbox = (geom.get("bbox_2d") if isinstance(geom, dict) else None) or []
        if len(bbox) >= 4:
            y1 = max(0, int(round(float(bbox[1]))))
            y2 = min(h, int(round(float(bbox[3]))))
            x1 = max(0, int(round(float(bbox[0]))))
            x2 = min(w, int(round(float(bbox[2]))))
            mask[y1:y2, x1:x2] = 1.0
        else:
            mask[:] = 1.0

        return mask.tolist()



def build_sam3d_provider(cfg: SAM3DProviderConfig):
    mode = cfg.mode.strip().lower()
    if mode == "http":
        return HttpSAM3DProvider(
            cfg.service.base_url,
            cfg.service.timeout_sec,
            mode="http",
            output_dir=cfg.output_dir,
            min_mesh_quality=cfg.min_mesh_quality,
        )
    if mode == "embedded":
        return EmbeddedSAM3DProvider(cfg)
    return DisabledSAM3DProvider()
