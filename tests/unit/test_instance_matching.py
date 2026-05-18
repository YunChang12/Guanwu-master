from __future__ import annotations

from guanwu.video.core.instance_matching import deduplicate_instances
from guanwu.video.core.types import DetectedInstance


def test_deduplicate_instances_collapses_identical_mask_candidates() -> None:
    mask_rle = {"size": [16, 16], "counts": "abc"}
    instances = [
        DetectedInstance(
            mask_ref="mask://frame_00001/s2t_1",
            bbox=[10.0, 20.0, 60.0, 80.0],
            object_id="s2t_1",
            concept_label="car",
            score=0.91,
            mask_rle=mask_rle,
        ),
        DetectedInstance(
            mask_ref="mask://frame_00001/s2t_2",
            bbox=[10.0, 20.0, 60.0, 80.0],
            object_id="s2t_2",
            concept_label="suv",
            score=0.72,
            mask_rle='{"size":[16,16],"counts":"abc"}',
        ),
        DetectedInstance(
            mask_ref="mask://frame_00001/s2t_3",
            bbox=[120.0, 30.0, 170.0, 90.0],
            object_id="s2t_3",
            concept_label="car",
            score=0.88,
            mask_rle={"size": [16, 16], "counts": "xyz"},
        ),
    ]

    deduplicated = deduplicate_instances(instances)

    assert [inst.object_id for inst in deduplicated] == ["s2t_1", "s2t_3"]


def test_deduplicate_instances_preserves_original_survivor_order() -> None:
    instances = [
        DetectedInstance(
            mask_ref="mask://frame_00001/s2t_5",
            bbox=[20.0, 20.0, 70.0, 80.0],
            object_id="s2t_5",
            concept_label="car",
            score=0.55,
        ),
        DetectedInstance(
            mask_ref="mask://frame_00001/s2t_2",
            bbox=[20.0, 20.0, 70.0, 80.0],
            object_id="s2t_2",
            concept_label="car",
            score=0.95,
        ),
        DetectedInstance(
            mask_ref="mask://frame_00001/s2t_9",
            bbox=[110.0, 20.0, 160.0, 80.0],
            object_id="s2t_9",
            concept_label="car",
            score=0.75,
        ),
    ]

    deduplicated = deduplicate_instances(instances)

    assert [inst.object_id for inst in deduplicated] == ["s2t_2", "s2t_9"]
