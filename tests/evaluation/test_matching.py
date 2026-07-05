from src.evaluation.matching import match_nodes, voxel_to_physical_um
from src.evaluation.schema import CellNode


def node(node_id: str, *, t: int = 0, z: float = 0.0, y: float = 0.0, x: float = 0.0) -> CellNode:
    return CellNode(node_id=node_id, sequence_id="seq", time=t, z=z, y=y, x=x)


def test_voxel_to_physical_uses_explicit_zyx_order() -> None:
    assert voxel_to_physical_um(2, 4, 8) == (3.25, 1.625, 3.25)


def test_matching_exactly_at_threshold_matches() -> None:
    result = match_nodes([node("gt")], [node("pred", z=7.0 / 1.625)])
    assert result.gt_to_pred == {"gt": "pred"}
    assert result.distances_um[("gt", "pred")] == 7.0


def test_matching_just_beyond_threshold_does_not_match() -> None:
    result = match_nodes([node("gt")], [node("pred", z=(7.0 / 1.625) + 0.001)])
    assert result.gt_to_pred == {}
    assert result.unmatched_gt == {"gt"}
    assert result.unmatched_pred == {"pred"}


def test_one_to_one_matching_when_predictions_compete_for_one_target() -> None:
    result = match_nodes(
        [node("gt")],
        [node("near", x=1), node("far", x=2)],
        voxel_spacing=(1.0, 1.0, 1.0),
        max_distance_um=7.0,
    )
    assert result.gt_to_pred == {"gt": "near"}
    assert result.unmatched_pred == {"far"}


def test_anisotropic_voxel_spacing_controls_distance() -> None:
    result = match_nodes(
        [node("gt")],
        [node("pred", z=1, y=0, x=0)],
        voxel_spacing=(8.0, 1.0, 1.0),
        max_distance_um=7.0,
    )
    assert result.gt_to_pred == {}
