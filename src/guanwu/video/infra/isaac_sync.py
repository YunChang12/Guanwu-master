from __future__ import annotations

import math
from pathlib import Path
import re

from pydantic import BaseModel, Field

from guanwu.video.core.schema import ObjectNode

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    HAS_USD = True
except Exception:
    HAS_USD = False


def _valid_vec3(value: object) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        out = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in out):
        return None
    return out


def _valid_quat_xyzw(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        out = (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in out):
        return None
    norm = math.sqrt(sum(v * v for v in out))
    if norm <= 1e-8:
        return None
    return tuple(v / norm for v in out)  # type: ignore[return-value]


class SyncReport(BaseModel):
    backend: str
    stage_path: str | None = None
    created: int = 0
    updated: int = 0
    removed: int = 0
    active_prims: int = 0
    prim_paths: dict[str, str] = Field(default_factory=dict)


class IsaacSyncAgent:
    """Isaac/OpenUSD sync layer with memory fallback when pxr is unavailable."""

    def __init__(self, stage_path: str = "data/demo_scene.usd", auto_save: bool = True) -> None:
        self._prim_map: dict[str, str] = {}
        self._state_cache: dict[str, dict] = {}
        self._stage_path = stage_path
        self._auto_save = auto_save
        self._backend = "usd" if HAS_USD else "memory"
        self._stage = None
        self._used_prim_tokens: set[str] = set()

        if self._backend == "usd":
            self._stage = self._init_stage(stage_path)

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def stage_path(self) -> str | None:
        return self._stage_path if self._backend == "usd" else None

    def sync(self, objects: list[ObjectNode]) -> SyncReport:
        active_ids = {obj.object_id for obj in objects}
        created = 0
        updated = 0

        for obj in objects:
            if obj.object_id not in self._prim_map:
                self._prim_map[obj.object_id] = self._make_prim_path(obj)
                created += 1
            else:
                updated += 1

            if self._backend == "usd":
                self._sync_usd_object(obj, self._prim_map[obj.object_id])
            else:
                self._sync_memory_object(obj, self._prim_map[obj.object_id])

        removed_ids = [obj_id for obj_id in self._prim_map if obj_id not in active_ids]
        for obj_id in removed_ids:
            prim_path = self._prim_map[obj_id]
            if self._backend == "usd":
                self._deactivate_usd_prim(prim_path)
            self._state_cache.pop(obj_id, None)
            del self._prim_map[obj_id]

        if self._backend == "usd" and self._auto_save and self._stage is not None:
            self._stage.GetRootLayer().Save()

        return SyncReport(
            backend=self._backend,
            stage_path=self.stage_path,
            created=created,
            updated=updated,
            removed=len(removed_ids),
            active_prims=len(self._prim_map),
            prim_paths=dict(self._prim_map),
        )

    def get_prim_path(self, object_id: str) -> str | None:
        return self._prim_map.get(object_id)

    def get_cached_state(self, object_id: str) -> dict | None:
        return self._state_cache.get(object_id)

    def _make_prim_path(self, obj: ObjectNode) -> str:
        raw = f"{obj.label}_{obj.object_id}"
        token = self._make_unique_token(raw, prefix="obj")
        return f"/World/Objects/{token}"

    def _sanitize_token(self, value: str, prefix: str = "p") -> str:
        token = re.sub(r"[^A-Za-z0-9_]", "_", str(value or ""))
        token = re.sub(r"_+", "_", token).strip("_")
        if not token:
            token = prefix
        if not (token[0].isalpha() or token[0] == "_"):
            token = f"{prefix}_{token}"
        return token

    def _make_unique_token(self, value: str, prefix: str = "p") -> str:
        base = self._sanitize_token(value, prefix=prefix)
        candidate = base
        idx = 1
        while candidate in self._used_prim_tokens:
            idx += 1
            candidate = f"{base}_{idx}"
        self._used_prim_tokens.add(candidate)
        return candidate

    def _sync_memory_object(self, obj: ObjectNode, prim_path: str) -> None:
        is_active, render_visibility = self._lifecycle_flags(obj.state.visibility)
        position = _valid_vec3(obj.geometry.pose_3d.position)
        orientation = _valid_quat_xyzw(obj.geometry.pose_3d.orientation_quat)
        scale = _valid_vec3(obj.geometry.scale_3d)
        self._state_cache[obj.object_id] = {
            "prim_path": prim_path,
            "label": obj.label,
            "shape_proxy": obj.geometry.shape_proxy,
            "position": list(position) if position is not None else None,
            "orientation": list(orientation) if orientation is not None else None,
            "scale": list(scale) if scale is not None else None,
            "geometry_available": position is not None and scale is not None,
            "prim_active": is_active,
            "render_visibility": render_visibility,
            "is_dynamic": obj.physics.is_dynamic,
            "velocity_linear": obj.physics.velocity_linear,
            "velocity_angular": obj.physics.velocity_angular,
            "cog": {
                "object_id": obj.object_id,
                "confidence": obj.confidence,
                "visibility": obj.state.visibility,
                "interaction_state": obj.state.interaction_state,
                "affordance_graspable": obj.affordance.graspable,
            },
        }

    def _init_stage(self, stage_path: str):
        path = Path(stage_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Always start fresh — stale prims from a previous run can leave
        # deactivated specs that cause DefinePrim failures on child paths.
        if path.exists():
            path.unlink()
        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.Xform.Define(stage, "/World")
        UsdGeom.Xform.Define(stage, "/World/Objects")
        return stage

    def _sync_usd_object(self, obj: ObjectNode, prim_path: str) -> None:
        assert self._stage is not None

        xform = UsdGeom.Xform.Define(self._stage, prim_path)
        prim = xform.GetPrim()
        self._apply_lifecycle_visibility(prim, obj.state.visibility)

        position = _valid_vec3(obj.geometry.pose_3d.position)
        orientation = _valid_quat_xyzw(obj.geometry.pose_3d.orientation_quat)
        scale = _valid_vec3(obj.geometry.scale_3d)

        ordered_ops = []
        if position is None:
            prim.RemoveProperty("xformOp:translate")
        else:
            translate_op = self._ensure_translate_op(xform)
            translate_op.Set(Gf.Vec3d(*position))
            ordered_ops.append(translate_op)

        if orientation is None:
            prim.RemoveProperty("xformOp:orient")
        else:
            orient_op = self._ensure_orient_op(xform)
            qx, qy, qz, qw = orientation
            orient_op.Set(Gf.Quatf(float(qw), Gf.Vec3f(float(qx), float(qy), float(qz))))
            ordered_ops.append(orient_op)

        if scale is None:
            prim.RemoveProperty("xformOp:scale")
        else:
            scale_op = self._ensure_scale_op(xform)
            scale_op.Set(Gf.Vec3f(*scale))
            ordered_ops.append(scale_op)
        xform.SetXformOpOrder(ordered_ops)

        # Skip collider setup when the prim is inactive (visibility=lost)
        # — USD cannot define children under an inactive parent. Also skip it
        # when metric scale is unavailable, so USD does not contain a unit box
        # pretending to be physical geometry.
        if prim.IsActive() and scale is not None:
            collider_prim = self._ensure_shape_proxy(prim_path, obj.geometry.shape_proxy)
            UsdPhysics.CollisionAPI.Apply(collider_prim)
        elif self._stage is not None:
            self._stage.RemovePrim(f"{prim_path}/Collider")

        if obj.physics.is_dynamic and position is not None and scale is not None:
            UsdPhysics.RigidBodyAPI.Apply(prim)
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            if obj.physics.mass is not None:
                mass_api.CreateMassAttr().Set(float(obj.physics.mass))

        self._set_cog_attr(prim, "cog:object_id", Sdf.ValueTypeNames.String, obj.object_id)
        self._set_cog_attr(prim, "cog:label", Sdf.ValueTypeNames.String, obj.label)
        self._set_cog_attr(prim, "cog:confidence", Sdf.ValueTypeNames.Float, float(obj.confidence))
        self._set_cog_attr(
            prim,
            "cog:affordance:graspable",
            Sdf.ValueTypeNames.Bool,
            bool(obj.affordance.graspable),
        )
        self._set_cog_attr(prim, "cog:state:visibility", Sdf.ValueTypeNames.String, obj.state.visibility)
        self._set_cog_attr(
            prim,
            "cog:state:interaction_state",
            Sdf.ValueTypeNames.String,
            obj.state.interaction_state,
        )

    def _deactivate_usd_prim(self, prim_path: str) -> None:
        assert self._stage is not None
        prim = self._stage.GetPrimAtPath(prim_path)
        if prim and prim.IsValid():
            self._stage.RemovePrim(prim_path)

    def _set_cog_attr(self, prim, name, type_name, value) -> None:
        attr = prim.GetAttribute(name)
        if not attr or not attr.IsValid():
            attr = prim.CreateAttribute(name, type_name, custom=True)
        attr.Set(value)

    def _ensure_translate_op(self, xform):
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                return op
        return xform.AddTranslateOp()

    def _ensure_orient_op(self, xform):
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                return op
        return xform.AddOrientOp()

    def _ensure_scale_op(self, xform):
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                return op
        return xform.AddScaleOp()

    def _apply_lifecycle_visibility(self, prim, visibility: str) -> None:
        imageable = UsdGeom.Imageable(prim)
        is_active, render_visibility = self._lifecycle_flags(visibility)
        prim.SetActive(is_active)
        if render_visibility == "invisible":
            imageable.MakeInvisible()
        else:
            imageable.MakeVisible()

    def _lifecycle_flags(self, visibility: str) -> tuple[bool, str]:
        if visibility == "lost":
            return False, "invisible"
        if visibility == "occluded":
            return True, "invisible"
        return True, "inherited"

    def _ensure_shape_proxy(self, prim_path: str, shape_proxy: str):
        assert self._stage is not None
        collider_path = f"{prim_path}/Collider"
        geom_path = f"{collider_path}/Geom"
        shape_to_type = {
            "box": "Cube",
            "sphere": "Sphere",
            "capsule": "Capsule",
            "cylinder": "Cylinder",
            "mesh": "Cube",
        }
        expected_type = shape_to_type.get(shape_proxy, "Cube")

        collider_prim = self._stage.GetPrimAtPath(collider_path)
        if not collider_prim or not collider_prim.IsValid():
            collider_xform = UsdGeom.Xform.Define(self._stage, collider_path)
        else:
            collider_xform = UsdGeom.Xform(collider_prim)

        existing_geom = self._stage.GetPrimAtPath(geom_path)
        if existing_geom and existing_geom.IsValid() and existing_geom.GetTypeName() != expected_type:
            self._stage.RemovePrim(geom_path)

        if expected_type == "Sphere":
            geom = UsdGeom.Sphere.Define(self._stage, geom_path)
            geom.CreateRadiusAttr(0.5)
        elif expected_type == "Capsule":
            geom = UsdGeom.Capsule.Define(self._stage, geom_path)
            geom.CreateRadiusAttr(0.35)
            geom.CreateHeightAttr(1.0)
        elif expected_type == "Cylinder":
            geom = UsdGeom.Cylinder.Define(self._stage, geom_path)
            geom.CreateRadiusAttr(0.4)
            geom.CreateHeightAttr(1.0)
        else:
            geom = UsdGeom.Cube.Define(self._stage, geom_path)
            geom.CreateSizeAttr(1.0)

        UsdGeom.Imageable(collider_xform.GetPrim()).MakeVisible()
        UsdGeom.Imageable(geom.GetPrim()).MakeVisible()
        return geom.GetPrim()
