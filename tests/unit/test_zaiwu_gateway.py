from __future__ import annotations

from pathlib import Path
import tarfile

import httpx

from guanwu.core.config import StorageConfig, WorkspaceConfig
from guanwu.video.clients.zaiwu import (
    ZaiwuGatewayClient,
    ZaiwuGroundedSAM2Detector,
    ZaiwuSAM3DAdapter,
    ZaiwuWildGSAdapter,
    normalize_service_id,
)
from guanwu.video.core.schema import ObjectNode
from guanwu.video.core.types import DetectedInstance, FrameDetections
from guanwu.video.executor import VideoProjectExecutor


class _FakeGateway(ZaiwuGatewayClient):
    def __init__(self) -> None:
        super().__init__(
            gateway_url="http://zaiwu.local:8181",
            job_poll_interval_sec=0.01,
            auto_start_workers=True,
        )
        self.start_calls = 0
        self.worker_calls = 0
        self.job_calls = 0

    def _request_json(  # type: ignore[override]
        self,
        method: str,
        path: str,
        *,
        json_payload: dict | None = None,
        timeout_sec: float | None = None,
    ) -> dict:
        if method == "GET" and path == "/api/v1/workers":
            self.worker_calls += 1
            if self.worker_calls == 1:
                return {"items": [], "summary": {}}
            return {
                "api": {
                    "upload_endpoint": "/upload",
                    "download_template": "/download/{file_id}",
                    "artifact_metadata_template": "/api/v1/artifacts/{artifact_id}",
                    "job_status_template": "/api/v1/jobs/{job_id}",
                },
                "items": [
                    {
                        "service_id": "services.grounding_dino_sam2",
                        "run_group": "services",
                        "status": "running",
                        "ready": True,
                        "ready_port": 19007,
                    }
                ],
                "summary": {},
            }
        if method == "POST" and path == "/api/v1/workers/actions/start":
            self.start_calls += 1
            return {"ok": True, "status": "started", "service_id": json_payload["service_id"]}
        if method == "POST" and path == "/api/v1/jobs":
            return {"status": "queued", "spec": {"job_id": "job-123"}}
        if method == "GET" and path == "/api/v1/jobs/job-123":
            self.job_calls += 1
            if self.job_calls < 3:
                return {"status": "running", "spec": {"job_id": "job-123"}}
            return {
                "status": "succeeded",
                "spec": {"job_id": "job-123"},
                "result": {"output_file_id": "outputs/depth.npy"},
            }
        raise AssertionError(f"Unexpected request: {method} {path}")


class _FlakyPollGateway(_FakeGateway):
    def __init__(self) -> None:
        super().__init__()
        self._poll_failures = 0

    def get_job(self, job_id: str) -> dict:  # type: ignore[override]
        if self._poll_failures == 0:
            self._poll_failures += 1
            raise httpx.ConnectError("[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol")
        return super().get_job(job_id)


class _UnknownJobOnceGateway(_FakeGateway):
    def __init__(self) -> None:
        super().__init__()
        self._poll_failures = 0

    def get_job(self, job_id: str) -> dict:  # type: ignore[override]
        if self._poll_failures == 0:
            self._poll_failures += 1
            raise RuntimeError(
                "Zaiwu gateway error 404 for GET /api/v1/jobs/job-123: {'error': 'Unknown job: job-123'}"
            )
        return super().get_job(job_id)


class _FakeGatewayRoutes(ZaiwuGatewayClient):
    def __init__(self) -> None:
        super().__init__(
            gateway_url="http://zaiwu.local:8181",
            job_poll_interval_sec=0.01,
            auto_start_workers=False,
        )
        self.upload_paths: list[str] = []
        self.download_paths: list[str] = []
        self.artifact_paths: list[str] = []

    def _request_json(  # type: ignore[override]
        self,
        method: str,
        path: str,
        *,
        json_payload: dict | None = None,
        timeout_sec: float | None = None,
    ) -> dict:
        if method == "GET" and path == "/api/v1/workers":
            return {
                "api": {
                    "upload_endpoint": "/gateway-upload",
                    "download_template": "/gateway-download/{file_id}",
                    "artifact_metadata_template": "/api/v1/files/{artifact_id}",
                    "job_status_template": "/api/v1/jobs/{job_id}",
                },
                "items": [],
                "summary": {},
            }
        if method == "GET" and path == "/api/v1/files/outputs%2Fresult.json":
            self.artifact_paths.append(path)
            return {"artifact_id": "outputs/result.json", "name": "result.json"}
        raise AssertionError(f"Unexpected request: {method} {path}")

    def _upload_via_gateway(  # type: ignore[override]
        self,
        path: str,
        *,
        content: bytes,
        timeout_sec: float | None = None,
    ) -> dict:
        self.upload_paths.append(path)
        return {"file_id": "uploads/demo.mp4"}

    def _download_via_gateway(  # type: ignore[override]
        self,
        path: str,
        *,
        timeout_sec: float | None = None,
    ) -> bytes:
        self.download_paths.append(path)
        return b"payload"


class _FakeJobGateway(ZaiwuGatewayClient):
    def __init__(self) -> None:
        super().__init__(
            gateway_url="http://zaiwu.local:8181",
            job_poll_interval_sec=0.01,
            auto_start_workers=False,
        )
        self.uploads: list[tuple[str, str]] = []
        self.jobs: list[tuple[str, str, dict]] = []
        self.downloads: list[tuple[str, str]] = []

    def ensure_service(self, service_id: str, *, timeout_sec: float = 60.0):  # type: ignore[override]
        return self._service_cache.setdefault(
            service_id,
            type("Endpoint", (), {"base_url": "http://zaiwu.local:19000", "sse_url": "http://zaiwu.local:19000/sse"})(),
        )

    def upload_file(self, service_id: str, path: str | Path) -> str:  # type: ignore[override]
        self.uploads.append((service_id, Path(path).name))
        return f"upload://{service_id}/{Path(path).name}"

    def run_service_job(  # type: ignore[override]
        self,
        service_id: str,
        operation: str,
        payload: dict,
        *,
        requested_by: str = "guanwu",
        execution_labels: dict[str, str] | None = None,
        timeout_sec: float | None = None,
    ) -> dict:
        self.jobs.append((service_id, operation, payload))
        if service_id == "services.grounding_dino_sam2" and operation == "gsam2_parse_video":
            return {
                "video_file_id": payload["video_file_id"],
                "frame_count": 1,
                "frames": [
                    {
                        "frame_idx": 1,
                        "timestamp": 0.0,
                        "instances": [
                            {
                                "mask_ref": "mask://frame_00001/trk_1",
                                "bbox": [10, 20, 30, 40],
                                "track_id": "trk_1",
                                "concept_label": "cup",
                                "segment_kind": "object",
                                "score": 0.9,
                                "mask_rle": "{\"counts\":\"abc\",\"size\":[8,8]}",
                            }
                        ],
                        "image_b64": "ZmFrZQ==",
                    }
                ],
            }
        if service_id == "services.sam3d" and operation == "reconstruct_objects":
            return {
                "request_id": "sam3d-job",
                "quality": 0.75,
                "files": [{"format": "ply", "file_id": "outputs/object.ply"}],
            }
        raise AssertionError(f"Unexpected service job: {service_id}.{operation} payload={payload}")

    def download_bytes(self, service_id: str, file_id: str) -> bytes:  # type: ignore[override]
        self.downloads.append((service_id, file_id))
        if file_id.endswith(".ply"):
            return b"ply\nformat ascii 1.0\nend_header\n"
        raise AssertionError(f"Unexpected download: {service_id} {file_id}")


class _FakeArtifactJobGateway(_FakeJobGateway):
    def run_service_job(  # type: ignore[override]
        self,
        service_id: str,
        operation: str,
        payload: dict,
        *,
        requested_by: str = "guanwu",
        execution_labels: dict[str, str] | None = None,
        timeout_sec: float | None = None,
    ) -> dict:
        self.jobs.append((service_id, operation, payload))
        if service_id == "services.grounding_dino_sam2" and operation == "gsam2_parse_video":
            return {"output_file_id": "outputs/frame_batches.json"}
        return super().run_service_job(
            service_id,
            operation,
            payload,
            requested_by=requested_by,
            execution_labels=execution_labels,
            timeout_sec=timeout_sec,
        )

    def download_bytes(self, service_id: str, file_id: str) -> bytes:  # type: ignore[override]
        self.downloads.append((service_id, file_id))
        if file_id == "outputs/frame_batches.json":
            return (
                b'{"frame_count":1,"frames":[{"frame_idx":1,"timestamp":0.0,"instances":'
                b'[{"mask_ref":"mask://frame_00001/trk_1","bbox":[10,20,30,40],'
                b'"track_id":"trk_1","concept_label":"cup","segment_kind":"object","score":0.9,'
                b'"mask_rle":"{\\"counts\\":\\"abc\\",\\"size\\":[8,8]}"}],"image_b64":"ZmFrZQ=="}]}'
            )
        return super().download_bytes(service_id, file_id)


class _FakeWildGSGateway(_FakeJobGateway):
    def run_service_job(  # type: ignore[override]
        self,
        service_id: str,
        operation: str,
        payload: dict,
        *,
        requested_by: str = "guanwu",
        execution_labels: dict[str, str] | None = None,
        timeout_sec: float | None = None,
    ) -> dict:
        self.jobs.append((service_id, operation, payload))
        if service_id == "services.wildgs_slam" and operation == "wildgs_run_slam":
            return {
                "camera_poses_file_id": "outputs/camera_poses.jsonl",
                "depth_maps_file_id": "outputs/depth_maps.tar.gz",
                "num_frames": 2,
                "slam_quality": 0.9,
            }
        return super().run_service_job(
            service_id,
            operation,
            payload,
            requested_by=requested_by,
            execution_labels=execution_labels,
            timeout_sec=timeout_sec,
        )

    def download_bytes(self, service_id: str, file_id: str) -> bytes:  # type: ignore[override]
        self.downloads.append((service_id, file_id))
        if file_id == "outputs/camera_poses.jsonl":
            return b'{"frame":0}\n{"frame":1}\n'
        if file_id == "outputs/depth_maps.tar.gz":
            import io

            payload = io.BytesIO()
            with tarfile.open(fileobj=payload, mode="w:gz") as handle:
                data = b"fake"
                info = tarfile.TarInfo("depth_maps/00000.npy")
                info.size = len(data)
                handle.addfile(info, io.BytesIO(data))
            return payload.getvalue()
        return super().download_bytes(service_id, file_id)


def test_zaiwu_gateway_resolves_service_sse_url_after_start() -> None:
    client = _FakeGateway()

    sse_url = client.service_sse_url("services.grounding_dino_sam2")

    assert sse_url == "http://zaiwu.local:19007/sse"
    assert client.start_calls == 1


def test_zaiwu_wildgs_adapter_requests_dense_depth_export(tmp_path) -> None:
    gateway = _FakeWildGSGateway()
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    adapter = ZaiwuWildGSAdapter(
        gateway,
        service_id="services.wildgs_slam",
        output_root=str(tmp_path / "wildgs"),
    )

    adapter.run_slam(
        video_path=str(video_path),
        fps_override=15.0,
        export_depth_every_frame=True,
        depth_export_stride=1,
        pose_export_stride=1,
        extract_every_input_frame=True,
        frame_stride=1,
    )

    assert gateway.jobs
    _service_id, operation, payload = gateway.jobs[-1]
    assert operation == "wildgs_run_slam"
    assert payload["export_depth_every_frame"] is True
    assert payload["depth_export_stride"] == 1
    assert payload["pose_export_stride"] == 1
    assert payload["extract_every_input_frame"] is True
    assert payload["frame_stride"] == 1


def test_zaiwu_gateway_runs_job_until_success() -> None:
    client = _FakeGateway()

    result = client.run_job(
        handler="services.depth_anything3.estimate_from_video",
        payload={"video_file_id": "uploads/demo.mp4"},
    )

    assert result["output_file_id"] == "outputs/depth.npy"
    assert client.job_calls >= 3


def test_zaiwu_gateway_retries_transient_polling_errors() -> None:
    client = _FlakyPollGateway()

    result = client.run_job(
        handler="services.depth_anything3.estimate_from_video",
        payload={"video_file_id": "uploads/demo.mp4"},
    )

    assert result["output_file_id"] == "outputs/depth.npy"
    assert client.job_calls >= 3


def test_zaiwu_gateway_retries_transient_unknown_job_polling_error() -> None:
    client = _UnknownJobOnceGateway()

    result = client.run_job(
        handler="services.wildgs_slam.wildgs_run_slam",
        payload={"video_file_id": "uploads/demo.mp4"},
    )

    assert result["output_file_id"] == "outputs/depth.npy"
    assert client.job_calls >= 3


def test_zaiwu_gateway_uses_advertised_file_routes(tmp_path: Path) -> None:
    client = _FakeGatewayRoutes()
    payload = tmp_path / "demo.mp4"
    payload.write_bytes(b"demo")

    file_id = client.upload_file("services.grounding_dino_sam2", payload)
    data = client.download_bytes("services.grounding_dino_sam2", "outputs/result.json")
    artifact = client.get_artifact("outputs/result.json")

    assert file_id == "uploads/demo.mp4"
    assert data == b"payload"
    assert artifact["artifact_id"] == "outputs/result.json"
    assert client.upload_paths == ["/gateway-upload?filename=demo.mp4"]
    assert client.download_paths == ["/gateway-download/outputs%2Fresult.json"]
    assert client.artifact_paths == ["/api/v1/files/outputs%2Fresult.json"]


def test_zaiwu_gateway_does_not_treat_stopped_worker_as_ready() -> None:
    client = ZaiwuGatewayClient(
        gateway_url="http://zaiwu.local:8181",
        auto_start_workers=False,
    )
    client._request_json = lambda method, path, **kwargs: {  # type: ignore[method-assign]
        "items": [
            {
                "service_id": "services.sam3d",
                "run_group": "services",
                "status": "stopped",
                "ready_port": 19003,
            }
        ],
        "summary": {},
    }

    endpoint = client.get_ready_service("services.sam3d")

    assert endpoint is None


def test_video_project_init_applies_workspace_zaiwu_settings(tmp_path: Path) -> None:
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"demo")

    workspace = WorkspaceConfig(
        workspace_root=str(tmp_path),
        storage=StorageConfig(
            raw_root=str(tmp_path / "raw"),
            staging_root=str(tmp_path / "staging"),
            canonical_root=str(tmp_path / "canonical"),
            export_root=str(tmp_path / "exports"),
            catalog_path=str(tmp_path / "catalog" / "catalog.duckdb"),
            project_root=str(tmp_path / "projects"),
        ),
    )
    workspace.video_pipeline.provider_mode = "zaiwu"
    workspace.video_pipeline.zaiwu_gateway_url = "http://zaiwu.example:8181"
    workspace.video_pipeline.object_detection_backend = "seg2track_sam2"
    workspace.video_pipeline.zaiwu_auto_start_workers = False

    context = VideoProjectExecutor.init_project(
        video=str(video_path),
        out_dir=tmp_path / "projects" / "video" / "demo",
        workspace=workspace,
    )

    assert context.config.project.provider_mode == "zaiwu"
    assert context.config.settings.zaiwu.enabled is True
    assert context.config.settings.zaiwu.gateway_url == "http://zaiwu.example:8181"
    assert context.config.settings.zaiwu.object_detection_backend == "seg2track_sam2"
    assert context.config.settings.zaiwu.auto_start_workers is False
    assert context.config.settings.zaiwu.depth_service == "services.depth_anything3"


def test_zaiwu_grounded_detector_prefetches_video_jobs_and_normalizes_track_ids(tmp_path: Path) -> None:
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"demo")
    gateway = _FakeJobGateway()
    detector = ZaiwuGroundedSAM2Detector(
        gateway,
        service_id="services.grounding_dino_sam2",
        video_source=str(video_path),
    )
    detector.set_object_detection_prompts(["cup"])

    detector.prefetch_video()
    batch = detector.detect_objects_in_frame(1, 0.0)

    assert gateway.uploads == [("services.grounding_dino_sam2", "demo.mp4")]
    assert gateway.jobs[0][1] == "gsam2_parse_video"
    assert batch.instances[0].object_id == "trk_1"
    assert batch.instances[0].concept_label == "cup"
    assert batch.image_b64 == "ZmFrZQ=="


def test_zaiwu_grounded_detector_prefetches_artifact_backed_video_jobs(tmp_path: Path) -> None:
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"demo")
    gateway = _FakeArtifactJobGateway()
    detector = ZaiwuGroundedSAM2Detector(
        gateway,
        service_id="services.grounding_dino_sam2",
        video_source=str(video_path),
    )
    detector.set_object_detection_prompts(["cup"])

    detector.prefetch_video()
    batch = detector.detect_objects_in_frame(1, 0.0)

    assert gateway.uploads == [("services.grounding_dino_sam2", "demo.mp4")]
    assert gateway.jobs[0][1] == "gsam2_parse_video"
    assert ("services.grounding_dino_sam2", "outputs/frame_batches.json") in gateway.downloads
    assert batch.instances[0].object_id == "trk_1"
    assert batch.image_b64 == "ZmFrZQ=="


def test_zaiwu_sam3d_adapter_materializes_job_artifacts(tmp_path: Path) -> None:
    gateway = _FakeJobGateway()
    adapter = ZaiwuSAM3DAdapter(
        gateway,
        service_id="services.sam3d",
        materialization_root=str(tmp_path),
        materialization_mode="copy",
    )
    detections = FrameDetections(
        frame_idx=3,
        timestamp=0.2,
        image_b64="ZmFrZQ==",
        instances=[
            DetectedInstance(
                mask_ref="mask://frame_00003/trk_1",
                bbox=[1.0, 2.0, 5.0, 7.0],
                object_id="trk_1",
                concept_label="cup",
                segment_kind="object",
                score=0.95,
            )
        ],
    )
    best_frames = {"trk_1": (detections, detections.instances[0])}
    objects = [ObjectNode(object_id="trk_1", label="cup", segment_kind="object")]

    result = adapter.reconstruct_object_meshes(best_frames, objects)

    mesh_path = Path(result["trk_1"]["mesh_path"])
    assert mesh_path.exists()
    assert mesh_path.suffix == ".ply"
    assert gateway.jobs[0][1] == "reconstruct_objects"
    assert gateway.downloads == [("services.sam3d", "outputs/object.ply")]


def test_zaiwu_wildgs_requires_camera_poses_artifact(tmp_path: Path) -> None:
    class _MissingCameraPosesGateway(_FakeJobGateway):
        def run_service_job(  # type: ignore[override]
            self,
            service_id: str,
            operation: str,
            payload: dict,
            *,
            requested_by: str = "guanwu",
            execution_labels: dict[str, str] | None = None,
            timeout_sec: float | None = None,
        ) -> dict:
            self.jobs.append((service_id, operation, payload))
            if service_id == "services.wildgs_slam" and operation == "wildgs_run_slam":
                return {"depth_maps_file_id": "outputs/depth_maps.tar.gz"}
            return super().run_service_job(
                service_id,
                operation,
                payload,
                requested_by=requested_by,
                execution_labels=execution_labels,
                timeout_sec=timeout_sec,
            )

    import pytest

    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"demo")
    adapter = ZaiwuWildGSAdapter(
        _MissingCameraPosesGateway(),
        service_id="services.wildgs_slam",
        output_root=str(tmp_path),
    )

    with pytest.raises(RuntimeError, match="missing required camera_poses_file_id"):
        adapter.run_slam(video_path=str(video_path))
    debug_payload = (tmp_path / "exports" / "wildgs_slam_result.json").read_text(encoding="utf-8")
    assert "outputs/depth_maps.tar.gz" in debug_payload


def test_normalize_service_id_accepts_legacy_mcps_prefix() -> None:
    assert normalize_service_id("mcps.sam3d") == "services.sam3d"
    assert normalize_service_id("services.sam3d") == "services.sam3d"
