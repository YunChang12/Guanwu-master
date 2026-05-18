from __future__ import annotations

from enum import Enum


class CanonicalCategory(str, Enum):
    FURNITURE = "furniture"
    APPLIANCE = "appliance"
    CONTAINER = "container"
    TABLEWARE = "tableware"
    ELECTRONICS = "electronics"
    STRUCTURE = "structure"
    DOOR_WINDOW = "door_window"
    VEHICLE = "vehicle"
    HUMAN = "human"
    ROBOT = "robot"
    ANIMAL = "animal"
    VEGETATION = "vegetation"
    ROAD_ELEMENT = "road_element"
    TOOL = "tool"
    SPORTS_EQUIPMENT = "sports_equipment"
    UNKNOWN = "unknown"


# Common mappings across datasets
_COMMON_MAPPINGS: dict[str, CanonicalCategory] = {
    "chair": CanonicalCategory.FURNITURE,
    "table": CanonicalCategory.FURNITURE,
    "desk": CanonicalCategory.FURNITURE,
    "sofa": CanonicalCategory.FURNITURE,
    "couch": CanonicalCategory.FURNITURE,
    "bed": CanonicalCategory.FURNITURE,
    "shelf": CanonicalCategory.FURNITURE,
    "cabinet": CanonicalCategory.FURNITURE,
    "bookshelf": CanonicalCategory.FURNITURE,
    "dresser": CanonicalCategory.FURNITURE,
    "nightstand": CanonicalCategory.FURNITURE,
    "door": CanonicalCategory.DOOR_WINDOW,
    "window": CanonicalCategory.DOOR_WINDOW,
    "wall": CanonicalCategory.STRUCTURE,
    "floor": CanonicalCategory.STRUCTURE,
    "ceiling": CanonicalCategory.STRUCTURE,
    "stairs": CanonicalCategory.STRUCTURE,
    "lamp": CanonicalCategory.ELECTRONICS,
    "tv": CanonicalCategory.ELECTRONICS,
    "monitor": CanonicalCategory.ELECTRONICS,
    "computer": CanonicalCategory.ELECTRONICS,
    "phone": CanonicalCategory.ELECTRONICS,
    "refrigerator": CanonicalCategory.APPLIANCE,
    "fridge": CanonicalCategory.APPLIANCE,
    "microwave": CanonicalCategory.APPLIANCE,
    "oven": CanonicalCategory.APPLIANCE,
    "washer": CanonicalCategory.APPLIANCE,
    "dishwasher": CanonicalCategory.APPLIANCE,
    "toilet": CanonicalCategory.APPLIANCE,
    "sink": CanonicalCategory.APPLIANCE,
    "bathtub": CanonicalCategory.APPLIANCE,
    "car": CanonicalCategory.VEHICLE,
    "truck": CanonicalCategory.VEHICLE,
    "bus": CanonicalCategory.VEHICLE,
    "bicycle": CanonicalCategory.VEHICLE,
    "motorcycle": CanonicalCategory.VEHICLE,
    "person": CanonicalCategory.HUMAN,
    "human": CanonicalCategory.HUMAN,
    "pedestrian": CanonicalCategory.HUMAN,
    "plant": CanonicalCategory.VEGETATION,
    "tree": CanonicalCategory.VEGETATION,
    "cup": CanonicalCategory.TABLEWARE,
    "bowl": CanonicalCategory.TABLEWARE,
    "plate": CanonicalCategory.TABLEWARE,
    "bottle": CanonicalCategory.CONTAINER,
    "box": CanonicalCategory.CONTAINER,
    "basket": CanonicalCategory.CONTAINER,
    "bag": CanonicalCategory.CONTAINER,
    "robot": CanonicalCategory.ROBOT,
    "sign": CanonicalCategory.ROAD_ELEMENT,
    "traffic_light": CanonicalCategory.ROAD_ELEMENT,
    "traffic_sign": CanonicalCategory.ROAD_ELEMENT,
}


def map_category(
    source_category: str | None, dataset_id: str | None = None
) -> tuple[str, str | None]:
    """Map a source category to canonical (category, supercategory).

    Returns (canonical_category, canonical_supercategory).
    Source category is preserved separately - this only provides the canonical mapping.
    """
    if source_category is None:
        return (CanonicalCategory.UNKNOWN.value, None)

    normalized = source_category.lower().strip().replace(" ", "_")

    if normalized in _COMMON_MAPPINGS:
        cat = _COMMON_MAPPINGS[normalized]
        return (cat.value, None)

    # Try partial matches
    for key, cat in _COMMON_MAPPINGS.items():
        if key in normalized or normalized in key:
            return (cat.value, None)

    return (CanonicalCategory.UNKNOWN.value, None)
