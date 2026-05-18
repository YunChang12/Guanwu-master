from __future__ import annotations

from enum import Enum


class GeometryLevel(str, Enum):
    G0_NONE = "G0_NONE"
    G1_BBOX = "G1_BBOX"
    G2_POINT_OBS = "G2_POINT_OBS"
    G3_PROXY_MESH = "G3_PROXY_MESH"
    G4_EXACT_MESH = "G4_EXACT_MESH"
    G5_ARTICULATED_MESH = "G5_ARTICULATED_MESH"
    G6_DEFORMABLE_MESH = "G6_DEFORMABLE_MESH"


class SceneKind(str, Enum):
    INDOOR_STATIC = "indoor_static"
    INDOOR_DYNAMIC = "indoor_dynamic"
    OUTDOOR_DRIVING = "outdoor_driving"
    EGOCENTRIC = "egocentric"
    SYNTHETIC_MANIPULATION = "synthetic_manipulation"
    OBJECT_ONLY = "object_only"
    MIXED = "mixed"


class SensorType(str, Enum):
    CAMERA = "camera"
    DEPTH_CAMERA = "depth_camera"
    LIDAR = "lidar"
    IMU = "imu"
    MICROPHONE = "microphone"
    OTHER = "other"


class AccessMode(str, Enum):
    PUBLIC = "public"
    GATED = "gated"
    MANUAL = "manual"
    MIXED = "mixed"


class SourceType(str, Enum):
    OFFICIAL_DOWNLOAD = "official_download"
    SDK = "sdk"
    LOCAL_FOLDER = "local_folder"
    GENERATOR = "generator"
    MANIFEST = "manifest"


class RecordScope(str, Enum):
    DATASET = "dataset"
    SCENE = "scene"
    ASSET = "asset"
    EPISODE = "episode"
    FILE = "file"


class PipelineStage(str, Enum):
    INVENTORY = "inventory"
    FETCH = "fetch"
    PARSE = "parse"
    NORMALIZE = "normalize"
    DERIVED = "derived"
    VALIDATE = "validate"
    CATALOG = "catalog"
    EXPORT = "export"
