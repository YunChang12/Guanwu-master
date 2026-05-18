"""Simulation dataset registry manager."""

from __future__ import annotations

from guanwu.adapters.base import get_adapter, list_adapters

_SIM_DATASET_INFO: dict[str, dict] = {
    "scannetpp": {
        "name": "ScanNet++",
        "description": "High-fidelity indoor scene reconstructions with DSLR and iPhone scans",
        "url": "https://scannetpp.mlsg.cit.tum.de/scannetpp/",
        "geometry_level_max": "G4_EXACT_MESH",
        "access_mode": "gated",
        "license": "ScanNet++ Terms of Use",
        "group": "p0",
    },
    "arkitscenes": {
        "name": "ARKitScenes",
        "description": "Apple ARKit RGB-D indoor scene understanding dataset",
        "url": "https://github.com/apple/ARKitScenes",
        "geometry_level_max": "G4_EXACT_MESH",
        "access_mode": "public",
        "license": "CC BY-NC-SA 4.0",
        "group": "p0",
    },
    "objaverse_xl": {
        "name": "Objaverse-XL",
        "description": "Large-scale 3D object dataset with 10M+ objects",
        "url": "https://objaverse.allenai.org/",
        "geometry_level_max": "G4_EXACT_MESH",
        "access_mode": "mixed",
        "license": "Mixed (per-object)",
        "group": "p0",
    },
    "partnet_mobility": {
        "name": "PartNet-Mobility",
        "description": "Articulated object assets with joint annotations",
        "url": "https://sapien.ucsd.edu/",
        "geometry_level_max": "G5_ARTICULATED_MESH",
        "access_mode": "public",
        "license": "PartNet-Mobility Terms",
        "group": "p0",
    },
    "maniskill3": {
        "name": "ManiSkill 3",
        "description": "Simulation scenes, trajectories, and env states for manipulation",
        "url": "https://maniskill.readthedocs.io/",
        "geometry_level_max": "G5_ARTICULATED_MESH",
        "access_mode": "public",
        "license": "Apache 2.0",
        "group": "p0",
    },
    "procthor_10k": {
        "name": "ProcTHOR-10K",
        "description": "10K procedurally generated indoor houses for embodied AI",
        "url": "https://procthor.allenai.org/",
        "geometry_level_max": "G1_BBOX",
        "access_mode": "public",
        "license": "Apache 2.0",
        "group": "p2",
    },
    "argoverse2_sensor": {
        "name": "Argoverse 2 Sensor",
        "description": "Dynamic driving multi-sensor dataset with LiDAR and cameras",
        "url": "https://argoverse.github.io/user-guide/",
        "geometry_level_max": "G2_POINT_OBS",
        "access_mode": "public",
        "license": "CC BY-NC-SA 4.0",
        "group": "p1",
    },
    "waymo_open": {
        "name": "Waymo Open Dataset",
        "description": "Dynamic driving multi-sensor dataset",
        "url": "https://waymo.com/open/",
        "geometry_level_max": "G2_POINT_OBS",
        "access_mode": "gated",
        "license": "Waymo Dataset License",
        "group": "p1",
    },
    "hoi4d": {
        "name": "HOI4D",
        "description": "Egocentric RGB-D hand-object interaction dataset",
        "url": "https://hoi4d.github.io/",
        "geometry_level_max": "G3_PROXY_MESH",
        "access_mode": "gated",
        "license": "HOI4D Terms",
        "group": "p1",
    },
    "ego_exo4d": {
        "name": "Ego-Exo4D",
        "description": "Multi-view egocentric and exocentric video dataset",
        "url": "https://docs.ego-exo4d-data.org/",
        "geometry_level_max": "G2_POINT_OBS",
        "access_mode": "gated",
        "license": "Ego-Exo4D License",
        "group": "p1",
    },
}


def list_datasets(group: str | None = None) -> list[dict]:
    adapters = list_adapters()
    results = []
    for dataset_id, info in sorted(_SIM_DATASET_INFO.items()):
        if group and info.get("group") != group:
            continue
        results.append({
            "dataset_id": dataset_id,
            "has_adapter": dataset_id in adapters,
            **info,
        })
    return results


def show_dataset(dataset_id: str) -> dict | None:
    info = _SIM_DATASET_INFO.get(dataset_id)
    if not info:
        return None
    adapters = list_adapters()
    result = {
        "dataset_id": dataset_id,
        "has_adapter": dataset_id in adapters,
        **info,
    }
    if dataset_id in adapters:
        adapter = get_adapter(dataset_id)
        result["adapter_version"] = adapter.version
        result["capabilities"] = adapter.capabilities()
    return result


def get_datasets_by_group(group: str) -> list[str]:
    return [
        dataset_id
        for dataset_id, info in _SIM_DATASET_INFO.items()
        if info.get("group") == group
    ]
