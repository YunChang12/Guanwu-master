"""Validation engine for normalized data bundles."""
from __future__ import annotations

import logging
from typing import Any

from guanwu.core.transforms import is_valid_transform
from guanwu.schemas.bundles import JobContext, NormalizeBundle, ValidationIssue, ValidationReport
from guanwu.schemas.enums import GeometryLevel

logger = logging.getLogger("guanwu")


def validate_bundle(bundle: NormalizeBundle, ctx: JobContext) -> ValidationReport:
    """Run all validation checks on a normalized bundle."""
    issues: list[ValidationIssue] = []

    issues.extend(_validate_dataset_record(bundle))
    issues.extend(_validate_scenes(bundle))
    issues.extend(_validate_assets(bundle))
    issues.extend(_validate_episodes(bundle))
    issues.extend(_validate_sensors(bundle))
    issues.extend(_validate_frames(bundle))
    issues.extend(_validate_instances(bundle))
    issues.extend(_validate_track_states(bundle))
    issues.extend(_validate_licenses(bundle))
    issues.extend(_validate_provenance(bundle))

    num_errors = sum(1 for i in issues if i.severity == "error")
    num_warnings = sum(1 for i in issues if i.severity == "warning")

    return ValidationReport(
        dataset_id=bundle.dataset_id,
        passed=num_errors == 0,
        issues=issues,
        num_errors=num_errors,
        num_warnings=num_warnings,
    )


def _validate_dataset_record(bundle: NormalizeBundle) -> list[ValidationIssue]:
    issues = []
    if bundle.dataset_record is None:
        issues.append(ValidationIssue(
            severity="error",
            check_name="dataset_record_exists",
            message="No dataset record in bundle",
        ))
    return issues


def _validate_scenes(bundle: NormalizeBundle) -> list[ValidationIssue]:
    issues = []
    seen_uids = set()
    for scene in bundle.scenes:
        if scene.scene_uid in seen_uids:
            issues.append(ValidationIssue(
                severity="error",
                check_name="scene_uid_unique",
                record_type="scene",
                record_id=scene.scene_uid,
                message=f"Duplicate scene_uid: {scene.scene_uid}",
            ))
        seen_uids.add(scene.scene_uid)

        if scene.world_units != "meters":
            issues.append(ValidationIssue(
                severity="error",
                check_name="scene_units",
                record_type="scene",
                record_id=scene.scene_uid,
                message=f"world_units must be 'meters', got '{scene.world_units}'",
            ))

        if scene.canonical_up_axis != "Z":
            issues.append(ValidationIssue(
                severity="error",
                check_name="scene_up_axis",
                record_type="scene",
                record_id=scene.scene_uid,
                message=f"canonical_up_axis must be 'Z', got '{scene.canonical_up_axis}'",
            ))
    return issues


def _validate_assets(bundle: NormalizeBundle) -> list[ValidationIssue]:
    issues = []
    seen_uids = set()
    for asset in bundle.assets:
        if asset.asset_uid in seen_uids:
            issues.append(ValidationIssue(
                severity="error",
                check_name="asset_uid_unique",
                record_type="asset",
                record_id=asset.asset_uid,
                message=f"Duplicate asset_uid: {asset.asset_uid}",
            ))
        seen_uids.add(asset.asset_uid)

        if asset.is_articulated and asset.geometry_level not in (
            GeometryLevel.G5_ARTICULATED_MESH.value,
            GeometryLevel.G5_ARTICULATED_MESH,
        ):
            issues.append(ValidationIssue(
                severity="warning",
                check_name="asset_articulation_level",
                record_type="asset",
                record_id=asset.asset_uid,
                message="Articulated asset should have geometry_level G5_ARTICULATED_MESH",
            ))
    return issues


def _validate_episodes(bundle: NormalizeBundle) -> list[ValidationIssue]:
    issues = []
    seen_uids = set()
    scene_uids = {scene.scene_uid for scene in bundle.scenes}
    for episode in bundle.episodes:
        if episode.episode_uid in seen_uids:
            issues.append(ValidationIssue(
                severity="error",
                check_name="episode_uid_unique",
                record_type="episode",
                record_id=episode.episode_uid,
                message=f"Duplicate episode_uid: {episode.episode_uid}",
            ))
        seen_uids.add(episode.episode_uid)
        if episode.scene_uid and episode.scene_uid not in scene_uids:
            issues.append(ValidationIssue(
                severity="warning",
                check_name="episode_scene_ref",
                record_type="episode",
                record_id=episode.episode_uid,
                message=f"Episode references unknown scene: {episode.scene_uid}",
            ))
    return issues


def _validate_sensors(bundle: NormalizeBundle) -> list[ValidationIssue]:
    issues = []
    for sensor in bundle.sensors:
        if sensor.T_sensor_from_parent is not None:
            if not is_valid_transform(sensor.T_sensor_from_parent):
                issues.append(ValidationIssue(
                    severity="warning",
                    check_name="sensor_transform_valid",
                    record_type="sensor",
                    record_id=sensor.sensor_uid,
                    message="Invalid T_sensor_from_parent transform matrix",
                ))
    return issues


def _validate_frames(bundle: NormalizeBundle) -> list[ValidationIssue]:
    issues = []
    sensor_frames: dict[str, list[int]] = {}
    episode_uids = {episode.episode_uid for episode in bundle.episodes}
    for frame in bundle.frames:
        sensor_frames.setdefault(frame.sensor_uid, []).append(frame.timestamp_ns)
        if frame.episode_uid and frame.episode_uid not in episode_uids:
            issues.append(ValidationIssue(
                severity="warning",
                check_name="frame_episode_ref",
                record_type="frame",
                record_id=frame.frame_uid,
                message=f"Frame references unknown episode: {frame.episode_uid}",
            ))

        if frame.T_world_from_sensor is not None:
            if not is_valid_transform(frame.T_world_from_sensor):
                issues.append(ValidationIssue(
                    severity="warning",
                    check_name="frame_transform_valid",
                    record_type="frame",
                    record_id=frame.frame_uid,
                    message="Invalid T_world_from_sensor transform matrix",
                ))

    # Check timestamp monotonicity per sensor
    for sensor_uid, timestamps in sensor_frames.items():
        for i in range(1, len(timestamps)):
            if timestamps[i] < timestamps[i - 1]:
                issues.append(ValidationIssue(
                    severity="warning",
                    check_name="timestamp_monotonicity",
                    record_type="frame",
                    record_id=sensor_uid,
                    message=f"Non-monotonic timestamps for sensor {sensor_uid}",
                ))
                break

    return issues


def _validate_instances(bundle: NormalizeBundle) -> list[ValidationIssue]:
    issues = []
    asset_uids = {a.asset_uid for a in bundle.assets}
    episode_uids = {episode.episode_uid for episode in bundle.episodes}
    scene_uids = {scene.scene_uid for scene in bundle.scenes}
    for inst in bundle.instances:
        if inst.asset_uid and inst.asset_uid not in asset_uids:
            issues.append(ValidationIssue(
                severity="warning",
                check_name="instance_asset_ref",
                record_type="instance",
                record_id=inst.instance_uid,
                message=f"Instance references unknown asset: {inst.asset_uid}",
            ))
        if inst.episode_uid and inst.episode_uid not in episode_uids:
            issues.append(ValidationIssue(
                severity="warning",
                check_name="instance_episode_ref",
                record_type="instance",
                record_id=inst.instance_uid,
                message=f"Instance references unknown episode: {inst.episode_uid}",
            ))
        if inst.scene_uid and inst.scene_uid not in scene_uids:
            issues.append(ValidationIssue(
                severity="warning",
                check_name="instance_scene_ref",
                record_type="instance",
                record_id=inst.instance_uid,
                message=f"Instance references unknown scene: {inst.scene_uid}",
            ))
    return issues


def _validate_track_states(bundle: NormalizeBundle) -> list[ValidationIssue]:
    issues = []
    instance_uids = {i.instance_uid for i in bundle.instances}
    for ts in bundle.track_states:
        if ts.instance_uid not in instance_uids:
            issues.append(ValidationIssue(
                severity="warning",
                check_name="track_instance_ref",
                record_type="track_state",
                record_id=ts.track_uid,
                message=f"Track references unknown instance: {ts.instance_uid}",
            ))
        if ts.T_world_from_object is not None and not is_valid_transform(ts.T_world_from_object):
            issues.append(ValidationIssue(
                severity="warning",
                check_name="track_transform_valid",
                record_type="track_state",
                record_id=ts.track_uid,
                message="Invalid T_world_from_object transform matrix",
            ))
    return issues


def _validate_licenses(bundle: NormalizeBundle) -> list[ValidationIssue]:
    issues = []
    if not bundle.licenses:
        issues.append(ValidationIssue(
            severity="warning",
            check_name="license_exists",
            message="No license records in bundle",
        ))
    return issues


def _validate_provenance(bundle: NormalizeBundle) -> list[ValidationIssue]:
    issues = []
    if not bundle.provenance:
        issues.append(ValidationIssue(
            severity="warning",
            check_name="provenance_exists",
            message="No provenance records in bundle",
        ))
    return issues
