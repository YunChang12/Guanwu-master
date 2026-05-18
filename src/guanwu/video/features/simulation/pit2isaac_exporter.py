from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from guanwu.video.core.schema import ObjectNode, RelationEdge
from guanwu.video.features.simulation.usd_coordinate_convention import (
    build_coordinate_report,
    build_world_to_usd_basis,
    convert_cv_camera_pose_to_usd,
    convert_world_points_to_usd,
    convert_world_pose_to_usd,
)

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

    HAS_USD = True
except Exception:
    HAS_USD = False

try:
    import trimesh as _trimesh

    HAS_TRIMESH = True
except Exception:
    HAS_TRIMESH = False

VALID_MODES = {"replay", "physics_ready", "hybrid"}
PHYSICS_MODES = {"physics_ready", "hybrid"}


@dataclass
class ExportResult:
    mode: str
    usd_backend: str
    usd_path: str
    physics_priors_json: str
    asset_mapping_json: str
    conversion_report_json: str
    object_count: int
    relation_count: int
    usdz_path: str | None = None


class PIT2IsaacExporter:
    def __init__(
        self,
        mode: str,
        output_root: str,
        usd_path: str,
        physics_priors_json: str,
        asset_mapping_json: str,
        conversion_report_json: str,
        use_category_assets: bool = True,
        fallback_visual: str = "primitive",
        collision_strategy: str = "primitive",
        min_geom_quality: float = 0.5,
        output_format: str = "usdc",
    ) -> None:
        self.mode = mode
        self.output_format = output_format
        self.output_root = Path(output_root)
        self.usd_path = Path(usd_path)
        self.physics_priors_json = Path(physics_priors_json)
        self.asset_mapping_json = Path(asset_mapping_json)
        self.conversion_report_json = Path(conversion_report_json)
        self.use_category_assets = use_category_assets
        self.fallback_visual = fallback_visual
        self.collision_strategy = collision_strategy
        self.min_geom_quality = min_geom_quality

    def export(
        self,
        objects: list[ObjectNode],
        relations: list[RelationEdge],
        pit_snapshot: dict,
    ) -> ExportResult:
        self._ensure_dirs()
        self._validate_mode()
        self._require_mandatory_prereqs(objects, pit_snapshot)

        if self.mode == "physics_ready" and not bool(pit_snapshot.get("metric_enabled", False)):
            raise ValueError("physics_ready export requires metric_enabled=true in PIT snapshot")

        asset_plan = {obj.object_id: self._resolve_asset(obj, pit_snapshot) for obj in objects}
        physics_priors = {
            obj.object_id: self._estimate_physics_prior(obj, pit_snapshot.get("vlm_priors", {}).get(obj.object_id))
            for obj in objects
        }
        physics_final = {obj.object_id: self._calibrate_physics(physics_priors[obj.object_id], obj) for obj in objects}
        relation_tracks = self._build_relation_tracks(relations, pit_snapshot)

        usd_backend = "usd" if HAS_USD else "json_fallback"
        coordinate_report = None
        if HAS_USD:
            coordinate_report = self._write_usd(objects, relations, pit_snapshot, asset_plan, physics_final, relation_tracks)
        else:
            self._write_fallback_scene_json(objects, relations, pit_snapshot, asset_plan, physics_final, relation_tracks)

        self.physics_priors_json.write_text(
            json.dumps({"physics_priors": physics_priors, "physics_final": physics_final}, indent=2),
            encoding="utf-8",
        )
        self.asset_mapping_json.write_text(
            json.dumps({"asset_mapping": asset_plan}, indent=2),
            encoding="utf-8",
        )

        report = {
            "mode": self.mode,
            "usd_backend": usd_backend,
            "object_count": len(objects),
            "relation_count": len(relations),
            "validation": self._validate(objects, relations, pit_snapshot, asset_plan, physics_final),
        }
        if coordinate_report is not None:
            report["coordinates"] = coordinate_report
        self.conversion_report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

        usdz_path = None
        if self.output_format == "usdz" and HAS_USD:
            usdz_path = str(self._package_usdz())

        return ExportResult(
            mode=self.mode,
            usd_backend=usd_backend,
            usd_path=str(self.usd_path),
            physics_priors_json=str(self.physics_priors_json),
            asset_mapping_json=str(self.asset_mapping_json),
            conversion_report_json=str(self.conversion_report_json),
            object_count=len(objects),
            relation_count=len(relations),
            usdz_path=usdz_path,
        )

    def _ensure_dirs(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.usd_path.parent.mkdir(parents=True, exist_ok=True)
        self.physics_priors_json.parent.mkdir(parents=True, exist_ok=True)
        self.asset_mapping_json.parent.mkdir(parents=True, exist_ok=True)
        self.conversion_report_json.parent.mkdir(parents=True, exist_ok=True)

    def _package_usdz(self) -> Path:
        usdz_path = self.usd_path.with_suffix(".usdz")
        success = UsdUtils.CreateNewUsdzPackage(
            Sdf.AssetPath(str(self.usd_path)), str(usdz_path),
        )
        if not success:
            raise RuntimeError(f"Failed to create USDZ package at {usdz_path}")
        return usdz_path

    def _validate_mode(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"Unsupported export mode: {self.mode}. expected one of {sorted(VALID_MODES)}")

    def _resolve_asset(self, obj: ObjectNode, pit_snapshot: dict) -> dict:
        sam3d_entry = (pit_snapshot.get("sam3d_meshes") or {}).get(obj.object_id)
        has_metric_scale = _valid_vec3(obj.geometry.scale_3d) is not None

        if not isinstance(sam3d_entry, dict) or not sam3d_entry.get("mesh_path"):
            collision_type, collision_params = self._build_collision_plan(obj)
            return {
                "instance_id": obj.object_id,
                "visual_source": self.fallback_visual if has_metric_scale else "none",
                "visual_asset_path": None,
                "visual_asset_exists": False,
                "collision_type": collision_type,
                "collision_params": collision_params,
                "scale_policy": "fit_geometry" if has_metric_scale else "unavailable",
                "pose_alignment_policy": "identity",
                "sam3d_quality": 0.0,
            }

        mesh_path = str(sam3d_entry["mesh_path"])
        mesh_quality = float(sam3d_entry.get("quality", 0.0))
        visual_source = "sam3d_mesh"
        visual_asset_path: str | None = mesh_path
        visual_asset_exists = self._path_exists(mesh_path)

        glb_sibling = Path(mesh_path).with_suffix(".glb")
        visual_glb_path: str | None = str(glb_sibling) if self._path_exists(str(glb_sibling)) else None

        collision_type, collision_params = self._build_collision_plan(obj)

        return {
            "instance_id": obj.object_id,
            "visual_source": visual_source,
            "visual_asset_path": visual_asset_path,
            "visual_asset_exists": visual_asset_exists,
            "visual_glb_path": visual_glb_path,
            "collision_type": collision_type,
            "collision_params": collision_params,
            "scale_policy": "fit_sam3d_mesh",
            "pose_alignment_policy": "align_principal_axis",
            "sam3d_quality": mesh_quality,
        }

    def _estimate_physics_prior(self, obj: ObjectNode, vlm_prior: dict | None) -> dict:
        label = obj.label.lower()
        has_vlm_prior = isinstance(vlm_prior, dict)
        vlm_prior = _normalize_vlm_prior(vlm_prior) if has_vlm_prior else {}
        confidence = vlm_prior.get("confidence")
        confidence_value = None
        if isinstance(confidence, (float, int)):
            confidence_value = float(max(0.0, min(1.0, confidence)))
        is_movable = vlm_prior.get("is_movable")

        return {
            "instance_id": obj.object_id,
            "class_name": str(vlm_prior.get("class_name", label)),
            "material_candidates": vlm_prior.get("material_candidates") if isinstance(vlm_prior.get("material_candidates"), list) else [],
            "mass_range_kg": _safe_range(vlm_prior.get("mass_range_kg")),
            "static_friction_range": _safe_range(vlm_prior.get("static_friction_range")),
            "dynamic_friction_range": _safe_range(vlm_prior.get("dynamic_friction_range")),
            "restitution_range": _safe_range(vlm_prior.get("restitution_range")),
            "measured_mass_kg": _positive_float(vlm_prior.get("measured_mass_kg")),
            "measured_static_friction": _bounded_float(vlm_prior.get("measured_static_friction"), 0.0, 1.5),
            "measured_dynamic_friction": _bounded_float(vlm_prior.get("measured_dynamic_friction"), 0.0, 1.5),
            "measured_restitution": _bounded_float(vlm_prior.get("measured_restitution"), 0.0, 1.0),
            "confidence": confidence_value,
            "rationale": str(vlm_prior["rationale"]) if isinstance(vlm_prior.get("rationale"), str) else None,
            "is_movable": bool(is_movable) if isinstance(is_movable, bool) else None,
            "source": "vlm_prior" if has_vlm_prior else None,
        }

    def _calibrate_physics(self, prior: dict, obj: ObjectNode) -> dict:
        mass = _positive_float(obj.physics.mass)
        if mass is None:
            mass = prior.get("measured_mass_kg")
        sf = _bounded_float(obj.physics.friction, 0.0, 1.5)
        if sf is None:
            sf = prior.get("measured_static_friction")
        df = prior.get("measured_dynamic_friction")
        restitution = _bounded_float(obj.physics.restitution, 0.0, 1.0)
        if restitution is None:
            restitution = prior.get("measured_restitution")
        source = "measured" if any(v is not None for v in (mass, sf, df, restitution)) else None

        return {
            "instance_id": obj.object_id,
            "mass_kg": round(float(mass), 4) if mass is not None else None,
            "center_of_mass_local": None,
            "static_friction": round(float(sf), 4) if sf is not None else None,
            "dynamic_friction": round(float(df), 4) if df is not None else None,
            "restitution": round(float(restitution), 4) if restitution is not None else None,
            "linear_damping": None,
            "angular_damping": None,
            "physics_quality": 1.0 if source == "measured" else None,
            "source": source,
            "is_movable": prior.get("is_movable") if prior.get("is_movable") is not None else bool(obj.physics.is_dynamic),
        }

    def _build_collision_plan(self, obj: ObjectNode) -> tuple[str, dict]:
        scale = _valid_vec3(obj.geometry.scale_3d)
        if scale is None:
            return "none", {}
        sx, sy, sz = [max(0.001, float(v)) for v in scale]

        if self.collision_strategy == "convex_hull":
            return "convex_hull", {"points": _aabb_corners(sx, sy, sz)}

        if self.collision_strategy == "convex_decomp":
            return "convex_decomp", {
                "parts": [
                    {"type": "box", "size": [sx * 0.5, sy, sz], "offset": [-sx * 0.25, 0.0, 0.0]},
                    {"type": "box", "size": [sx * 0.5, sy, sz], "offset": [sx * 0.25, 0.0, 0.0]},
                ]
            }

        collision_type = self._collision_type_for_label(obj.label.lower())
        collision_params = {
            "size": [sx, sy, sz],
            "radius": max(0.01, 0.5 * min(sx, sy)),
            "height": max(0.02, sz),
        }
        return collision_type, collision_params

    def _write_usd(
        self,
        objects: list[ObjectNode],
        relations: list[RelationEdge],
        pit_snapshot: dict,
        asset_plan: dict,
        physics_final: dict,
        relation_tracks: dict[str, list[dict]],
    ) -> dict:
        stage = Usd.Stage.CreateNew(str(self.usd_path))
        stage.SetTimeCodesPerSecond(1.0)

        # Compute timeline range from trajectory data
        all_ts: list[float] = []
        for track in pit_snapshot.get("object_trajectories", {}).values():
            all_ts.extend(float(r["timestamp_sec"]) for r in track if "timestamp_sec" in r)
        for c in pit_snapshot.get("camera_trajectory", []):
            if "timestamp_sec" in c:
                all_ts.append(float(c["timestamp_sec"]))
        if all_ts:
            stage.SetStartTimeCode(min(all_ts))
            stage.SetEndTimeCode(max(all_ts))

        bg_points = None
        bg_recon = pit_snapshot.get("background_reconstruction") or {}
        points_path = bg_recon.get("points_path", "")
        if points_path and Path(points_path).exists() and HAS_TRIMESH:
            try:
                cloud = _trimesh.load(points_path)
                if hasattr(cloud, "vertices"):
                    bg_points = np.asarray(cloud.vertices, dtype=np.float64)
            except Exception:
                bg_points = None

        coord_convention = build_world_to_usd_basis(
            scene_up=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            camera_track=pit_snapshot.get("camera_trajectory", []),
            bg_points=bg_points,
        )

        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, "/World")
        UsdGeom.Xform.Define(stage, "/World/Objects")
        UsdGeom.Xform.Define(stage, "/World/Cameras")
        UsdGeom.Xform.Define(stage, "/World/PIT")
        UsdGeom.Xform.Define(stage, "/World/PIT/Relations")

        # Camera trajectory
        cam = UsdGeom.Camera.Define(stage, "/World/Cameras/PITCamera")
        cam_xf = UsdGeom.Xformable(cam.GetPrim())
        cam_translate = _ensure_op(cam_xf, UsdGeom.XformOp.TypeTranslate)
        cam_orient = _ensure_op(cam_xf, UsdGeom.XformOp.TypeOrient)
        for c in pit_snapshot.get("camera_trajectory", []):
            t = float(c.get("timestamp_sec", 0.0))
            T_world_from_cam = np.eye(4, dtype=np.float64)
            T_world_from_cam[:3, :3] = np.asarray(c.get("R", np.eye(3)), dtype=np.float64).reshape(3, 3)
            T_world_from_cam[:3, 3] = np.asarray(c.get("t", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
            T_usd = convert_cv_camera_pose_to_usd(T_world_from_cam, coord_convention)
            cam_translate.Set(Gf.Vec3d(float(T_usd[0, 3]), float(T_usd[1, 3]), float(T_usd[2, 3])), t)
            quat = _quat_from_rotation(T_usd[:3, :3].tolist())
            cam_orient.Set(Gf.Quatf(float(quat[3]), Gf.Vec3f(float(quat[0]), float(quat[1]), float(quat[2]))), t)

        object_traj = pit_snapshot.get("object_trajectories", {})
        apply_physics = self.mode in PHYSICS_MODES
        object_path_by_id: dict[str, str] = {}
        used_object_tokens: set[str] = set()
        for obj in objects:
            token = _make_unique_usd_token(obj.object_id, used_object_tokens, prefix="obj")
            object_path_by_id[obj.object_id] = f"/World/Objects/{token}"

        for obj in objects:
            obj_path = object_path_by_id[obj.object_id]
            xform = UsdGeom.Xform.Define(stage, obj_path)
            prim = xform.GetPrim()

            self._set_attr(prim, "pit:instance_id", Sdf.ValueTypeNames.String, obj.object_id)
            self._set_attr(prim, "pit:class_name", Sdf.ValueTypeNames.String, obj.label)
            self._set_attr(prim, "pit:geom_quality", Sdf.ValueTypeNames.Float, float(obj.confidence))
            self._set_attr(prim, "pit:source", Sdf.ValueTypeNames.String, "PIT+SAM3D+VLM")

            vis = UsdGeom.Xform.Define(stage, f"{obj_path}/Visual").GetPrim()
            col_xform = UsdGeom.Xform.Define(stage, f"{obj_path}/Collision")
            UsdGeom.Imageable(col_xform.GetPrim()).MakeInvisible()
            col = col_xform.GetPrim()

            vis_plan = asset_plan[obj.object_id]
            self._set_attr(vis, "pit:visual_source", Sdf.ValueTypeNames.String, vis_plan["visual_source"])
            if vis_plan.get("visual_asset_path"):
                self._set_attr(vis, "pit:visual_asset_path", Sdf.ValueTypeNames.String, vis_plan["visual_asset_path"])

            # Embed mesh geometry into the Visual prim
            mesh_embedded = False
            if vis_plan.get("visual_asset_exists") and HAS_TRIMESH:
                load_path = vis_plan.get("visual_glb_path") or vis_plan["visual_asset_path"]
                mesh_embedded = _load_mesh_as_usd(stage, f"{obj_path}/Visual/Mesh", load_path, obj.label)
            if not mesh_embedded and _valid_vec3(obj.geometry.scale_3d) is not None:
                _create_primitive_visual(stage, f"{obj_path}/Visual/Mesh", obj)
            elif not mesh_embedded:
                self._set_attr(vis, "pit:visual_disabled", Sdf.ValueTypeNames.Bool, True)

            if apply_physics and vis_plan["collision_type"] != "none":
                # Collision primitive / hull / decomposition
                col_type = vis_plan["collision_type"]
                col_params = vis_plan.get("collision_params", {})
                if col_type == "cylinder":
                    g = UsdGeom.Cylinder.Define(stage, f"{obj_path}/Collision/Geom")
                    g.CreateRadiusAttr(float(col_params["radius"]))
                    g.CreateHeightAttr(float(col_params["height"]))
                    UsdPhysics.CollisionAPI.Apply(g.GetPrim())
                elif col_type == "capsule":
                    g = UsdGeom.Capsule.Define(stage, f"{obj_path}/Collision/Geom")
                    g.CreateRadiusAttr(float(col_params["radius"]))
                    g.CreateHeightAttr(float(col_params["height"]))
                    UsdPhysics.CollisionAPI.Apply(g.GetPrim())
                elif col_type == "sphere":
                    g = UsdGeom.Sphere.Define(stage, f"{obj_path}/Collision/Geom")
                    g.CreateRadiusAttr(float(col_params["radius"]))
                    UsdPhysics.CollisionAPI.Apply(g.GetPrim())
                elif col_type == "convex_hull":
                    mesh = UsdGeom.Mesh.Define(stage, f"{obj_path}/Collision/Hull")
                    pts = [Gf.Vec3f(*p) for p in col_params.get("points", [])]
                    if pts:
                        mesh.CreatePointsAttr(pts)
                        mesh.CreateFaceVertexCountsAttr([4, 4, 4, 4, 4, 4])
                        mesh.CreateFaceVertexIndicesAttr([
                            0,
                            1,
                            3,
                            2,
                            4,
                            5,
                            7,
                            6,
                            0,
                            1,
                            5,
                            4,
                            2,
                            3,
                            7,
                            6,
                            0,
                            2,
                            6,
                            4,
                            1,
                            3,
                            7,
                            5,
                        ])
                    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
                elif col_type == "convex_decomp":
                    for idx, part in enumerate(col_params.get("parts", [])):
                        p = UsdGeom.Cube.Define(stage, f"{obj_path}/Collision/Part_{idx}")
                        p.CreateSizeAttr(1.0)
                        pxf = UsdGeom.Xformable(p.GetPrim())
                        pop = _ensure_op(pxf, UsdGeom.XformOp.TypeTranslate)
                        sc = _ensure_op(pxf, UsdGeom.XformOp.TypeScale)
                        ox, oy, oz = part.get("offset", [0.0, 0.0, 0.0])
                        sx, sy, sz = part.get("size", [1.0, 1.0, 1.0])
                        pop.Set(Gf.Vec3d(float(ox), float(oy), float(oz)))
                        sc.Set(Gf.Vec3f(float(sx), float(sy), float(sz)))
                        UsdPhysics.CollisionAPI.Apply(p.GetPrim())
                elif col_type == "box":
                    g = UsdGeom.Cube.Define(stage, f"{obj_path}/Collision/Geom")
                    g.CreateSizeAttr(1.0)
                    UsdPhysics.CollisionAPI.Apply(g.GetPrim())
                else:
                    self._set_attr(col, "pit:collision_disabled", Sdf.ValueTypeNames.Bool, True)

                physics = physics_final[obj.object_id]
                if (
                    physics.get("is_movable", True)
                    and physics.get("mass_kg") is not None
                    and _valid_vec3(obj.geometry.pose_3d.position) is not None
                    and _valid_vec3(obj.geometry.scale_3d) is not None
                ):
                    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
                    _ = rb
                    mass_api = UsdPhysics.MassAPI.Apply(prim)
                    mass_api.CreateMassAttr().Set(float(physics["mass_kg"]))
                for attr_name, value in (
                    ("pit:static_friction", physics.get("static_friction")),
                    ("pit:dynamic_friction", physics.get("dynamic_friction")),
                    ("pit:restitution", physics.get("restitution")),
                ):
                    if value is not None:
                        self._set_attr(prim, attr_name, Sdf.ValueTypeNames.Float, float(value))
            else:
                self._set_attr(col, "pit:collision_disabled", Sdf.ValueTypeNames.Bool, True)

            xf = UsdGeom.Xformable(prim)
            raw_tracks = object_traj.get(obj.object_id, [])
            tracks = [rec for rec in raw_tracks if _valid_vec3(rec.get("centroid_world")) is not None]
            if tracks:
                for rec in tracks:
                    t = float(rec.get("timestamp_sec", 0.0))
                    center = _valid_vec3(rec.get("centroid_world"))
                    if center is None:
                        continue
                    quat = _valid_quat_xyzw(rec.get("orientation_quat")) or _valid_quat_xyzw(
                        obj.geometry.pose_3d.orientation_quat
                    )
                    rot_world = _rotation_from_quat_xyzw(quat) if quat is not None else np.eye(3, dtype=np.float64)
                    rot_usd, trans_usd = convert_world_pose_to_usd(
                        rot_world,
                        center,
                        coord_convention,
                    )
                    tr = _ensure_op(xf, UsdGeom.XformOp.TypeTranslate)
                    tr.Set(Gf.Vec3d(float(trans_usd[0]), float(trans_usd[1]), float(trans_usd[2])), t)
                    scale = _scale_from_record(rec) or _valid_vec3(obj.geometry.scale_3d)
                    if scale is not None:
                        sc = _ensure_op(xf, UsdGeom.XformOp.TypeScale)
                        sc.Set(Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2])), t)
                    if quat is not None:
                        ori = _ensure_op(xf, UsdGeom.XformOp.TypeOrient)
                        qx, qy, qz, qw = _quat_from_rotation(rot_usd.tolist())
                        ori.Set(Gf.Quatf(float(qw), Gf.Vec3f(float(qx), float(qy), float(qz))), t)
            else:
                center = _valid_vec3(obj.geometry.pose_3d.position)
                if center is not None:
                    quat = _valid_quat_xyzw(obj.geometry.pose_3d.orientation_quat)
                    rot_world = _rotation_from_quat_xyzw(quat) if quat is not None else np.eye(3, dtype=np.float64)
                    rot_usd, trans_usd = convert_world_pose_to_usd(rot_world, center, coord_convention)
                    tr = _ensure_op(xf, UsdGeom.XformOp.TypeTranslate)
                    tr.Set(Gf.Vec3d(float(trans_usd[0]), float(trans_usd[1]), float(trans_usd[2])))
                    scale = _valid_vec3(obj.geometry.scale_3d)
                    if scale is not None:
                        sc = _ensure_op(xf, UsdGeom.XformOp.TypeScale)
                        sc.Set(Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2])))
                    if quat is not None:
                        ori = _ensure_op(xf, UsdGeom.XformOp.TypeOrient)
                        qx, qy, qz, qw = _quat_from_rotation(rot_usd.tolist())
                        ori.Set(Gf.Quatf(float(qw), Gf.Vec3f(float(qx), float(qy), float(qz))))

        # Relation metadata prims
        for rel in relations:
            rel_key = self._rel_key(rel.subject_id, rel.predicate, rel.object_id)
            rel_subject = _sanitize_usd_token(rel.subject_id, prefix="subj")
            rel_object = _sanitize_usd_token(rel.object_id, prefix="obj")
            rel_pred = _sanitize_usd_token(rel.predicate, prefix="rel")
            rel_path = f"/World/PIT/Relations/{rel_subject}__{rel_object}__{rel_pred}"
            rprim = UsdGeom.Xform.Define(stage, rel_path).GetPrim()
            self._set_attr(rprim, "pit:subject_id", Sdf.ValueTypeNames.String, rel.subject_id)
            self._set_attr(rprim, "pit:object_id", Sdf.ValueTypeNames.String, rel.object_id)
            self._set_attr(rprim, "pit:predicate", Sdf.ValueTypeNames.String, rel.predicate)
            self._set_attr(rprim, "pit:confidence", Sdf.ValueTypeNames.Float, float(rel.confidence))

            track = relation_tracks.get(rel_key, [])
            distances = [
                float(x["distance"]) if isinstance(x.get("distance"), (float, int)) else None
                for x in track
            ]
            world_labels = [str(x.get("relation_world", rel.predicate)) for x in track]
            camera_labels = [str(x.get("relation_camera", rel.predicate)) for x in track]
            confs = [float(x.get("confidence", rel.confidence)) for x in track]
            self._set_attr(rprim, "pit:distance_track", Sdf.ValueTypeNames.String, json.dumps(distances))
            self._set_attr(rprim, "pit:relation_world_track", Sdf.ValueTypeNames.String, json.dumps(world_labels))
            self._set_attr(rprim, "pit:relation_camera_track", Sdf.ValueTypeNames.String, json.dumps(camera_labels))
            self._set_attr(rprim, "pit:confidence_track", Sdf.ValueTypeNames.String, json.dumps(confs))

        # Background point cloud (StreamSplat or WildGS static map)
        fg_bg_labels_path = bg_recon.get("fg_bg_labels_path", "")
        bg_source = bg_recon.get("source", "streamsplat")
        obj_extent = _compute_object_extent(object_traj)

        if points_path and Path(points_path).exists() and HAS_TRIMESH:
            self._write_background_points(
                stage,
                points_path,
                fg_bg_labels_path,
                obj_extent,
                source=bg_source,
                coord_convention=coord_convention,
            )

        stage.GetRootLayer().Save()
        return build_coordinate_report(coord_convention, camera_track=pit_snapshot.get("camera_trajectory", []))

    def _write_background_points(
        self, stage, points_path: str, fg_bg_labels_path: str,
        obj_extent: dict[str, tuple[float, float]] | None = None,
        source: str = "wildgs",
        coord_convention=None,
    ) -> None:
        """Embed WildGS static-map point cloud into USD at /World/Background.

        Points are assumed to be in the shared world frame and are converted once
        into the final USD frame.
        ``obj_extent`` and ``source`` are kept for API compatibility.
        """
        import logging

        log = logging.getLogger("guanwu.video.pit2isaac")
        try:
            cloud = _trimesh.load(points_path)
            if not hasattr(cloud, "vertices"):
                log.warning("Background PLY has no vertices: %s", points_path)
                return

            pts = np.asarray(cloud.vertices, dtype=np.float64)
            colors = None
            if hasattr(cloud, "colors") and cloud.colors is not None:
                colors = np.asarray(cloud.colors)
            elif hasattr(cloud, "visual") and hasattr(cloud.visual, "vertex_colors"):
                colors = np.asarray(cloud.visual.vertex_colors)

            # Keep only background points when fg/bg labels are available
            if fg_bg_labels_path and Path(fg_bg_labels_path).exists():
                labels = np.load(fg_bg_labels_path)
                if len(labels) == len(pts):
                    bg_mask = labels == 0
                    pts = pts[bg_mask]
                    if colors is not None and len(colors) == len(labels):
                        colors = colors[bg_mask]
                    log.info("Background: keeping %d / %d points (fg removed)", len(pts), len(labels))

            # Subsample to keep USD manageable (max 300k points)
            max_pts = 300_000
            if len(pts) > max_pts:
                step = max(1, len(pts) // max_pts)
                pts = pts[::step]
                if colors is not None:
                    colors = colors[::step]

            # Scale background to match object coordinate extent
            if obj_extent:
                bg_min = pts.min(axis=0)
                bg_max = pts.max(axis=0)
                bg_range = bg_max - bg_min
                bg_range = np.where(bg_range < 1e-6, 1.0, bg_range)
                bg_center = (bg_min + bg_max) / 2.0

                obj_range = np.array([
                    obj_extent["x"][1] - obj_extent["x"][0],
                    obj_extent["y"][1] - obj_extent["y"][0],
                    obj_extent["z"][1] - obj_extent["z"][0],
                ])
                obj_center = np.array([
                    (obj_extent["x"][0] + obj_extent["x"][1]) / 2.0,
                    (obj_extent["y"][0] + obj_extent["y"][1]) / 2.0,
                    (obj_extent["z"][0] + obj_extent["z"][1]) / 2.0,
                ])

                # Scale each axis to match, then translate to same center
                scale = obj_range / bg_range
                pts = (pts - bg_center) * scale + obj_center
                log.info("Background rescaled: bg_range=%s → obj_range=%s", bg_range, obj_range)

            if coord_convention is not None:
                pts = convert_world_points_to_usd(pts, coord_convention)

            n = len(pts)
            log.info("Embedding %d background points into /World/Background", n)

            bg_mesh = UsdGeom.Points.Define(stage, "/World/Background")
            bg_mesh.CreatePointsAttr([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pts])
            bg_mesh.CreateWidthsAttr([0.02] * n)
            if colors is not None and len(colors) == n:
                rgb = colors[:, :3].astype(float)
                if rgb.max() > 1.0:
                    rgb = rgb / 255.0
                UsdGeom.PrimvarsAPI(bg_mesh).CreatePrimvar(
                    "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex,
                ).Set([Gf.Vec3f(float(c[0]), float(c[1]), float(c[2])) for c in rgb])
        except Exception as exc:
            logging.getLogger("guanwu.video.pit2isaac").warning("Failed to embed background points: %s", exc)

    def _build_relation_tracks(self, relations: list[RelationEdge], pit_snapshot: dict) -> dict[str, list[dict]]:
        tracks: dict[str, list[dict]] = {}
        src = pit_snapshot.get("relation_trajectories")
        if isinstance(src, dict):
            for rel_key, seq in src.items():
                if isinstance(seq, list):
                    tracks[rel_key] = [x for x in seq if isinstance(x, dict)]

        for rel in relations:
            rel_key = self._rel_key(rel.subject_id, rel.predicate, rel.object_id)
            if rel_key not in tracks:
                tracks[rel_key] = [
                    {
                        "timestamp_sec": float(rel.temporal.start_ts),
                        "distance": None,
                        "relation_world": rel.predicate,
                        "relation_camera": rel.predicate,
                        "confidence": float(rel.confidence),
                    }
                ]
        return tracks

    def _write_fallback_scene_json(
        self,
        objects: list[ObjectNode],
        relations: list[RelationEdge],
        pit_snapshot: dict,
        asset_plan: dict,
        physics_final: dict,
        relation_tracks: dict[str, list[dict]],
    ) -> None:
        import logging

        logging.getLogger("guanwu.video.pit2isaac").warning(
            "pxr (usd-core) is not installed; writing JSON fallback. "
            "Mesh geometry will NOT be embedded. Install with: pip install 'spwm-agent[usd]'"
        )
        data = {
            "note": "pxr unavailable; JSON fallback scene — meshes not embedded",
            "mode": self.mode,
            "objects": [o.model_dump() for o in objects],
            "relations": [r.model_dump() for r in relations],
            "asset_plan": asset_plan,
            "physics_final": physics_final,
            "relation_trajectories": relation_tracks,
            "pit_snapshot": pit_snapshot,
        }
        self.usd_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _validate(
        self,
        objects: list[ObjectNode],
        relations: list[RelationEdge],
        pit_snapshot: dict,
        asset_plan: dict,
        physics_final: dict,
    ) -> dict:
        object_tracks = pit_snapshot.get("object_trajectories", {})
        camera_track = pit_snapshot.get("camera_trajectory", [])

        visual_assets_resolved = all(
            p["visual_source"] != "usd_library" or bool(p.get("visual_asset_exists", False)) for p in asset_plan.values()
        )

        collision_ok = all(self._collision_plan_nonzero(p.get("collision_type", "none"), p.get("collision_params", {})) for p in asset_plan.values())

        physics_ok = all(
            (p["restitution"] is None or 0.0 <= p["restitution"] <= 1.0)
            and (p["mass_kg"] is None or p["mass_kg"] > 0.0)
            and (
                p["dynamic_friction"] is None
                or p["static_friction"] is None
                or p["dynamic_friction"] <= p["static_friction"]
            )
            for p in physics_final.values()
        )

        checks = {
            "all_objects_have_pose": all(_valid_vec3(o.geometry.pose_3d.position) is not None for o in objects),
            "visual_assets_resolved": visual_assets_resolved,
            "collision_size_nonzero": collision_ok,
            "physics_in_range": physics_ok,
            "camera_time_continuous": self._time_track_monotonic(camera_track),
            "object_time_continuous": all(self._time_track_monotonic(track) for track in object_tracks.values()),
            "camera_object_time_aligned": self._camera_object_time_aligned(camera_track, object_tracks),
            "relations_serializable": self._relations_serializable(relations),
            "mode_constraints_ok": self.mode != "physics_ready" or bool(pit_snapshot.get("metric_enabled", False)),
        }
        checks["ok"] = all(checks.values())
        return checks

    def _collision_plan_nonzero(self, collision_type: str, collision_params: dict) -> bool:
        if collision_type == "none":
            return True
        if collision_type == "convex_hull":
            return len(collision_params.get("points", [])) >= 4
        if collision_type == "convex_decomp":
            parts = collision_params.get("parts", [])
            return bool(parts) and all(all(float(s) > 0.0 for s in p.get("size", [])) for p in parts)
        size = collision_params.get("size", [0.0, 0.0, 0.0])
        return any(float(v) > 0.0 for v in size)

    def _time_track_monotonic(self, track: list[dict]) -> bool:
        if len(track) <= 1:
            return True
        prev = float(track[0].get("timestamp_sec", 0.0))
        for rec in track[1:]:
            cur = float(rec.get("timestamp_sec", 0.0))
            if cur < prev:
                return False
            prev = cur
        return True

    def _camera_object_time_aligned(self, camera_track: list[dict], object_tracks: dict[str, list[dict]]) -> bool:
        if not camera_track or not object_tracks:
            return True
        cam_times = {round(float(x.get("timestamp_sec", 0.0)), 3) for x in camera_track}
        for track in object_tracks.values():
            if not track:
                continue
            obj_times = {round(float(x.get("timestamp_sec", 0.0)), 3) for x in track}
            if cam_times.intersection(obj_times):
                continue
            return False
        return True

    def _relations_serializable(self, relations: list[RelationEdge]) -> bool:
        try:
            json.dumps([r.model_dump() for r in relations])
            return True
        except Exception:
            return False

    def _collision_type_for_label(self, label: str) -> str:
        if label in {"cup", "bottle"}:
            return "cylinder"
        if label in {"ball", "sphere"}:
            return "sphere"
        if label in {"robot arm", "robot gripper"}:
            return "capsule"
        return "box"

    def _set_attr(self, prim, name, type_name, value) -> None:
        attr = prim.GetAttribute(name)
        if not attr or not attr.IsValid():
            attr = prim.CreateAttribute(name, type_name, custom=True)
        attr.Set(value)

    def _path_exists(self, path_str: str) -> bool:
        p = Path(path_str)
        if p.is_absolute():
            return p.exists()
        return (Path.cwd() / p).exists()

    def _rel_key(self, subject_id: str, predicate: str, object_id: str) -> str:
        return f"{subject_id}|{predicate}|{object_id}"

    def _require_mandatory_prereqs(self, objects: list[ObjectNode], pit_snapshot: dict) -> None:
        sam3d_meshes = pit_snapshot.get("sam3d_meshes")
        vlm_priors = pit_snapshot.get("vlm_priors")
        if not isinstance(sam3d_meshes, dict):
            pit_snapshot["sam3d_meshes"] = {}
            sam3d_meshes = pit_snapshot["sam3d_meshes"]
        if not isinstance(vlm_priors, dict):
            pit_snapshot["vlm_priors"] = {}
            vlm_priors = pit_snapshot["vlm_priors"]

        missing_sam3d: list[str] = []
        invalid_sam3d: list[str] = []
        invalid_vlm: list[str] = []
        for obj in objects:
            sid = sam3d_meshes.get(obj.object_id)
            vid = vlm_priors.get(obj.object_id)
            if not isinstance(sid, dict):
                missing_sam3d.append(obj.object_id)
            if isinstance(sid, dict):
                mesh_path = str(sid.get("mesh_path", "")).strip()
                quality = sid.get("quality")
                segment_kind = str(sid.get("segment_kind", "")).strip().lower()
                if (
                    not mesh_path
                    or not self._path_exists(mesh_path)
                    or not isinstance(quality, (float, int))
                    or segment_kind not in {"object", "body"}
                ):
                    invalid_sam3d.append(obj.object_id)
            if isinstance(vid, dict):
                ranges = (
                    "mass_range_kg",
                    "static_friction_range",
                    "dynamic_friction_range",
                    "restitution_range",
                )
                if any(key in vid and _safe_range(vid.get(key)) is None for key in ranges):
                    invalid_vlm.append(obj.object_id)

        if missing_sam3d or invalid_sam3d or invalid_vlm:
            import logging
            logging.getLogger("guanwu.video.pit2isaac").warning(
                "SAM3D/VLM prerequisites missing or invalid (unavailable fields stay empty): "
                f"missing_sam3d={missing_sam3d}, "
                f"invalid_sam3d={invalid_sam3d}, invalid_vlm={invalid_vlm}"
            )


def _load_mesh_as_usd(stage, prim_path: str, asset_path: str, label: str = "") -> bool:
    """Load a GLB/OBJ/PLY mesh file and embed its geometry into the USD stage.

    Returns True on success, False if the mesh has no vertices/faces.
    """
    import logging

    log = logging.getLogger("guanwu.video.pit2isaac")
    if not HAS_TRIMESH:
        return False
    try:
        mesh = _trimesh.load(asset_path, force="mesh")
    except Exception as exc:
        log.warning("Failed to load mesh %s for %s: %s", asset_path, label, exc)
        return False

    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        return False
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return False

    usd_mesh = UsdGeom.Mesh.Define(stage, prim_path)
    usd_mesh.CreatePointsAttr([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in mesh.vertices])
    usd_mesh.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    usd_mesh.CreateFaceVertexIndicesAttr(mesh.faces.flatten().tolist())
    usd_mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    usd_mesh.CreateDoubleSidedAttr(True)

    if hasattr(mesh, "vertex_normals") and len(mesh.vertex_normals) == len(mesh.vertices):
        usd_mesh.CreateNormalsAttr(
            [Gf.Vec3f(float(n[0]), float(n[1]), float(n[2])) for n in mesh.vertex_normals]
        )
        usd_mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    # Extract vertex colors from the mesh visual if available
    if hasattr(mesh, "visual") and hasattr(mesh.visual, "vertex_colors"):
        try:
            vc = mesh.visual.vertex_colors
            if vc is not None and len(vc) == len(mesh.vertices):
                display_colors = [
                    Gf.Vec3f(float(c[0]) / 255.0, float(c[1]) / 255.0, float(c[2]) / 255.0)
                    for c in vc
                ]
                dc_pv = UsdGeom.PrimvarsAPI(usd_mesh).CreatePrimvar(
                    "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
                )
                dc_pv.Set(display_colors)
        except Exception:
            pass

    log.info("Embedded mesh %s (%d verts, %d faces) at %s", asset_path, len(mesh.vertices), len(mesh.faces), prim_path)
    return True


def _create_primitive_visual(stage, prim_path: str, obj: ObjectNode) -> None:
    """Create a box primitive as a visual fallback when no mesh is available."""
    scale = _valid_vec3(obj.geometry.scale_3d)
    if scale is None:
        return
    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube.GetPrim())
    sc = xf.AddScaleOp()
    sc.Set(Gf.Vec3f(*[max(0.001, float(v)) for v in scale]))


def _normalize_vlm_prior(vlm: dict) -> dict:
    """Keep VLM priors descriptive; do not synthesize physical ranges."""
    return dict(vlm)


def _ensure_op(xf: Any, op_type: Any):
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == op_type:
            return op
    if op_type == UsdGeom.XformOp.TypeTranslate:
        return xf.AddTranslateOp()
    if op_type == UsdGeom.XformOp.TypeScale:
        return xf.AddScaleOp()
    return xf.AddOrientOp()


def _first_unique_frame(
    object_id: str, object_traj: dict[str, list[dict]],
) -> int:
    """Return the index of the first trajectory record where *object_id*
    has a centroid that is not shared with any other object in the same frame.

    The state-estimator sometimes assigns an identical centroid / bbox to
    multiple objects in the first frame(s) while tracking is initialising.
    Skipping those frames avoids giving unrelated objects the same size.
    """
    track = object_traj.get(object_id, [])
    if not track:
        return 0

    # Build a set of (frame_id → centroid_key) for every OTHER object
    other_centroids: dict[int, set[tuple[float, ...]]] = {}
    for oid, otrack in object_traj.items():
        if oid == object_id:
            continue
        for rec in otrack:
            fid = rec.get("frame_id", -1)
            cw = rec.get("centroid_world")
            if cw and len(cw) >= 3:
                key = (round(cw[0], 6), round(cw[1], 6), round(cw[2], 6))
                other_centroids.setdefault(fid, set()).add(key)

    for idx, rec in enumerate(track):
        fid = rec.get("frame_id", -1)
        cw = rec.get("centroid_world")
        if cw and len(cw) >= 3:
            key = (round(cw[0], 6), round(cw[1], 6), round(cw[2], 6))
            if key not in other_centroids.get(fid, set()):
                return idx

    # All frames shared — fall back to first frame
    return 0


def _compute_object_extent(object_traj: dict[str, list[dict]]) -> dict[str, tuple[float, float]] | None:
    """Compute the coordinate extent from first-unique-frame object positions.

    Uses only the first trajectory record per object where the centroid is
    unique (not shared with other objects).  Returns
    ``{"x": (min, max), "y": (min, max), "z": (min, max)}`` or *None*.
    """
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for oid, track in object_traj.items():
        if not track:
            continue
        idx = _first_unique_frame(oid, object_traj)
        cw = track[idx].get("centroid_world")
        if cw and len(cw) >= 3:
            xs.append(float(cw[0]))
            ys.append(float(cw[1]))
            zs.append(float(cw[2]))
    if not xs:
        return None
    # Add a small margin (10%) to avoid clipping
    def _extent_with_margin(vals: list[float]) -> tuple[float, float]:
        lo, hi = min(vals), max(vals)
        margin = max(0.1, (hi - lo) * 0.1)
        return (lo - margin, hi + margin)
    return {
        "x": _extent_with_margin(xs),
        "y": _extent_with_margin(ys),
        "z": _extent_with_margin(zs),
    }


def _sanitize_usd_token(value: str, prefix: str = "p") -> str:
    token = re.sub(r"[^A-Za-z0-9_]", "_", str(value or ""))
    token = re.sub(r"_+", "_", token).strip("_")
    if not token:
        token = prefix
    if not (token[0].isalpha() or token[0] == "_"):
        token = f"{prefix}_{token}"
    return token


def _make_unique_usd_token(value: str, used: set[str], prefix: str = "p") -> str:
    base = _sanitize_usd_token(value, prefix=prefix)
    candidate = base
    idx = 1
    while candidate in used:
        idx += 1
        candidate = f"{base}_{idx}"
    used.add(candidate)
    return candidate



def _mid(rng: list[float]) -> float:
    if not rng:
        return 0.0
    if len(rng) == 1:
        return float(rng[0])
    return (float(rng[0]) + float(rng[1])) / 2.0


def _valid_vec3(value: Any) -> list[float] | None:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(3)
    except Exception:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return [float(v) for v in arr]


def _valid_quat_xyzw(value: Any) -> list[float] | None:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(4)
    except Exception:
        return None
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 1e-8:
        return None
    return [float(v) for v in (arr / norm)]


def _positive_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out) or out <= 0.0:
        return None
    return out


def _bounded_float(value: Any, lo: float, hi: float) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out) or not lo <= out <= hi:
        return None
    return out


def _safe_range(value: Any) -> list[float] | None:
    if isinstance(value, list) and len(value) >= 2:
        try:
            lo = float(value[0])
            hi = float(value[1])
        except (TypeError, ValueError):
            return None
        if not np.isfinite(lo) or not np.isfinite(hi):
            return None
        if lo > hi:
            lo, hi = hi, lo
        return [lo, hi]
    return None


def _aabb_corners(sx: float, sy: float, sz: float) -> list[list[float]]:
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
    return [
        [-hx, -hy, -hz],
        [hx, -hy, -hz],
        [-hx, hy, -hz],
        [hx, hy, -hz],
        [-hx, -hy, hz],
        [hx, -hy, hz],
        [-hx, hy, hz],
        [hx, hy, hz],
    ]


def _scale_from_record(record: dict) -> list[float] | None:
    aabb = record.get("bbox_3d_aabb")
    if isinstance(aabb, dict):
        lo = _valid_vec3(aabb.get("min"))
        hi = _valid_vec3(aabb.get("max"))
        if lo is not None and hi is not None:
            scale = [max(0.0, hi[i] - lo[i]) for i in range(3)]
            if any(v > 0.0 for v in scale):
                return scale
    return _valid_vec3(record.get("scale"))


def _quat_from_rotation(rot: Any) -> list[float]:
    if not isinstance(rot, list) or len(rot) != 3:
        return [0.0, 0.0, 0.0, 1.0]
    try:
        m00, m01, m02 = [float(v) for v in rot[0]]
        m10, m11, m12 = [float(v) for v in rot[1]]
        m20, m21, m22 = [float(v) for v in rot[2]]
    except Exception:
        return [0.0, 0.0, 0.0, 1.0]

    trace = m00 + m11 + m22
    if trace > 0.0:
        s = (trace + 1.0) ** 0.5 * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = (1.0 + m00 - m11 - m22) ** 0.5 * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = (1.0 + m11 - m00 - m22) ** 0.5 * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = (1.0 + m22 - m00 - m11) ** 0.5 * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s

    return [qx, qy, qz, qw]


def _rotation_from_quat_xyzw(quat: Any) -> np.ndarray:
    if not isinstance(quat, (list, tuple)) or len(quat) != 4:
        return np.eye(3, dtype=np.float64)
    try:
        x, y, z, w = [float(v) for v in quat]
    except Exception:
        return np.eye(3, dtype=np.float64)
    norm = max((x * x + y * y + z * z + w * w) ** 0.5, 1e-8)
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
