"""Logical registry for natural video ingestion."""

NATURAL_VIDEO_DATASET_ID = "natural_video"


def list_sources() -> list[dict]:
    return [{
        "dataset_id": NATURAL_VIDEO_DATASET_ID,
        "name": "Natural Video",
        "description": "Single natural-scene video parsed into proxy geometry and USDC outputs",
        "source_type": "generator",
        "geometry_level_max": "G3_PROXY_MESH",
    }]


def show_source(dataset_id: str) -> dict | None:
    for source in list_sources():
        if source["dataset_id"] == dataset_id:
            return source
    return None
