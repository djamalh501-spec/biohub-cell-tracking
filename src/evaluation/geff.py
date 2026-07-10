"""GEFF graph decoding utilities for official training annotations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .schema import CellNode, LineageDivision, SequenceGraph, TemporalEdge


NODE_PROP_ALIASES: dict[str, tuple[str, ...]] = {
    "t": ("t", "time", "frame", "timepoint"),
    "z": ("z", "z_voxel", "position_z", "centroid_z", "center_z"),
    "y": ("y", "y_voxel", "position_y", "centroid_y", "center_y"),
    "x": ("x", "x_voxel", "position_x", "centroid_x", "center_x"),
}


@dataclass(frozen=True, slots=True)
class GeffArraySummary:
    path: str
    shape: tuple[int, ...] | None
    dtype: str | None
    attributes: dict[str, Any]


@dataclass(frozen=True, slots=True)
class GeffGraphSummary:
    dataset: str
    path: Path
    arrays: tuple[GeffArraySummary, ...]
    node_count: int
    edge_count: int
    division_count: int
    validation_errors: tuple[str, ...]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _array_attrs(array_dir: Path) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for name in ("attrs.json", ".zattrs", "zarr.json"):
        path = array_dir / name
        if not path.exists():
            continue
        data = _load_json(path)
        if name == "zarr.json":
            attrs.update(data.get("attributes", {}))
        else:
            attrs.update(data)
    return attrs


def _open_zarr_array(store: Path, array_path: str) -> np.ndarray | None:
    try:
        import zarr  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return None
    try:
        group = zarr.open_group(str(store), mode="r")
        return np.asarray(group[array_path][:])
    except (KeyError, OSError, ValueError, TypeError):
        return None


def _read_array(store: Path, array_path: str) -> np.ndarray:
    npy_path = store / f"{array_path}.npy"
    if npy_path.exists():
        return np.load(npy_path, allow_pickle=False)
    nested_npy_path = store / array_path / "data.npy"
    if nested_npy_path.exists():
        return np.load(nested_npy_path, allow_pickle=False)
    zarr_array = _open_zarr_array(store, array_path)
    if zarr_array is not None:
        return zarr_array
    raise FileNotFoundError(
        f"could not read GEFF array {array_path!r} from {store}. "
        "Install zarr for official GEFF stores or provide .npy fixture arrays."
    )


def _array_metadata_from_disk(store: Path, array_path: str) -> GeffArraySummary:
    array_dir = store / array_path
    attrs = _array_attrs(array_dir)
    metadata: dict[str, Any] = {}
    for name in ("zarr.json", ".zarray"):
        path = array_dir / name
        if path.exists():
            metadata = _load_json(path)
            break
    shape = metadata.get("shape")
    dtype = metadata.get("data_type", metadata.get("dtype"))
    npy_path = store / f"{array_path}.npy"
    data_npy_path = array_dir / "data.npy"
    if (shape is None or dtype is None) and (npy_path.exists() or data_npy_path.exists()):
        array = np.load(npy_path if npy_path.exists() else data_npy_path, mmap_mode="r", allow_pickle=False)
        shape = array.shape
        dtype = str(array.dtype)
    return GeffArraySummary(
        path=array_path,
        shape=tuple(int(value) for value in shape) if shape is not None else None,
        dtype=str(dtype) if dtype is not None else None,
        attributes=attrs,
    )


def list_geff_arrays(path: str | Path) -> tuple[GeffArraySummary, ...]:
    """List known arrays/groups in a GEFF store without loading full image volumes."""

    store = Path(path)
    candidates = {"nodes/ids", "nodes/props", "edges/ids", "edges/props"}
    for metadata_path in store.rglob("zarr.json"):
        if metadata_path == store / "zarr.json":
            continue
        candidates.add(str(metadata_path.parent.relative_to(store)).replace("\\", "/"))
    for metadata_path in store.rglob(".zarray"):
        candidates.add(str(metadata_path.parent.relative_to(store)).replace("\\", "/"))
    for npy_path in store.rglob("*.npy"):
        relative = str(npy_path.relative_to(store).with_suffix("")).replace("\\", "/")
        if relative.endswith("/data"):
            relative = relative[: -len("/data")]
        candidates.add(relative)
    return tuple(_array_metadata_from_disk(store, candidate) for candidate in sorted(candidates))


def _normalise_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _property_names(attrs: dict[str, Any]) -> tuple[str, ...]:
    for key in ("property_names", "prop_names", "columns", "fields", "names"):
        value = attrs.get(key)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return tuple(value)
    properties = attrs.get("properties")
    if isinstance(properties, list):
        names = []
        for item in properties:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict) and isinstance(item.get("name"), str):
                names.append(item["name"])
        if names:
            return tuple(names)
    return ()


def _find_property_index(names: Iterable[str], field: str) -> int | None:
    normalised = [_normalise_name(name) for name in names]
    for alias in NODE_PROP_ALIASES[field]:
        if alias in normalised:
            return normalised.index(alias)
    return None


def _decode_node_properties(props: np.ndarray, attrs: dict[str, Any]) -> dict[str, np.ndarray]:
    if props.dtype.names:
        decoded: dict[str, np.ndarray] = {}
        normalised_names = {_normalise_name(name): name for name in props.dtype.names}
        for field, aliases in NODE_PROP_ALIASES.items():
            source_name = next((normalised_names[alias] for alias in aliases if alias in normalised_names), None)
            if source_name is None:
                raise ValueError(f"nodes/props structured dtype is missing {field!r}; names={props.dtype.names}")
            decoded[field] = np.asarray(props[source_name])
        return decoded

    if props.ndim != 2:
        raise ValueError(f"nodes/props must be a structured array or 2D property matrix, got shape {props.shape}")
    names = _property_names(attrs)
    if not names:
        raise ValueError(
            "nodes/props is positional but no property_names/columns metadata was found; "
            "cannot safely infer t,z,y,x indices"
        )
    indices: dict[str, int] = {}
    for field in ("t", "z", "y", "x"):
        index = _find_property_index(names, field)
        if index is None:
            raise ValueError(f"nodes/props metadata names {names} do not identify {field!r}")
        indices[field] = index
    return {field: np.asarray(props[:, index]) for field, index in indices.items()}


def _read_grouped_node_property(store: Path, field: str) -> np.ndarray:
    paths = [f"nodes/props/{alias}/values" for alias in NODE_PROP_ALIASES[field]]
    failures: list[str] = []
    for array_path in paths:
        try:
            return np.asarray(_read_array(store, array_path)).reshape(-1)
        except FileNotFoundError as exc:
            failures.append(str(exc))
    raise FileNotFoundError(
        f"could not read GEFF grouped node property {field!r}; tried {paths}. "
        f"Last error: {failures[-1] if failures else 'no paths tried'}"
    )


def _read_node_properties(store: Path) -> dict[str, np.ndarray]:
    """Read GEFF node properties from legacy table or official grouped layout."""

    try:
        props = np.asarray(_read_array(store, "nodes/props"))
    except FileNotFoundError as table_error:
        grouped = {field: _read_grouped_node_property(store, field) for field in ("t", "z", "y", "x")}
        lengths = {field: len(values) for field, values in grouped.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"grouped nodes/props lengths differ: {lengths}") from table_error
        return grouped
    prop_attrs = _array_attrs(store / "nodes" / "props")
    return _decode_node_properties(props, prop_attrs)


def _decode_edge_ids(edge_ids: np.ndarray) -> np.ndarray:
    if edge_ids.size == 0:
        return np.empty((0, 2), dtype=edge_ids.dtype)
    if edge_ids.ndim != 2:
        raise ValueError(f"edges/ids must be a 2D source-target array, got shape {edge_ids.shape}")
    if edge_ids.shape[1] == 2:
        return edge_ids
    if edge_ids.shape[0] == 2:
        return edge_ids.T
    raise ValueError(f"edges/ids must have one dimension of length 2, got shape {edge_ids.shape}")


def read_geff_graph(path: str | Path, dataset: str | None = None) -> SequenceGraph:
    """Decode a GEFF training annotation store into a dataset-level graph."""

    store = Path(path)
    if not store.exists():
        raise FileNotFoundError(f"GEFF store does not exist: {store}")
    dataset_id = dataset or store.name.removesuffix(".geff")
    node_ids = np.asarray(_read_array(store, "nodes/ids")).reshape(-1)
    decoded_props = _read_node_properties(store)
    if len(node_ids) != len(decoded_props["t"]):
        raise ValueError(f"nodes/ids length {len(node_ids)} does not match nodes/props length {len(decoded_props['t'])}")

    nodes: list[CellNode] = []
    for index, node_id_value in enumerate(node_ids):
        try:
            time = int(decoded_props["t"][index])
            z = float(decoded_props["z"][index])
            y = float(decoded_props["y"][index])
            x = float(decoded_props["x"][index])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"node index {index} has invalid t,z,y,x properties: {exc}") from exc
        if not all(math.isfinite(value) for value in (time, z, y, x)):
            raise ValueError(f"node index {index} has non-finite t,z,y,x properties")
        nodes.append(
            CellNode(
                node_id=str(int(node_id_value)),
                sequence_id=dataset_id,
                time=time,
                z=z,
                y=y,
                x=x,
            )
        )

    edge_ids = _decode_edge_ids(np.asarray(_read_array(store, "edges/ids")))
    edges = [
        TemporalEdge(parent_id=str(int(source_id)), child_id=str(int(target_id)))
        for source_id, target_id in edge_ids
    ]
    graph = SequenceGraph(sequence_id=dataset_id, nodes=tuple(nodes), edges=tuple(edges))
    errors = graph.validate()
    if errors:
        raise ValueError(f"decoded GEFF graph {dataset_id!r} failed validation: {'; '.join(errors)}")
    return graph


def extract_divisions(graph: SequenceGraph) -> list[LineageDivision]:
    """Infer divisions as parents with exactly two outgoing child edges.

    No official explicit GEFF division metadata has been observed yet. If a future audit
    finds such metadata, this function should prefer it over out-degree inference.
    """

    children_by_parent: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        children_by_parent[edge.parent_id].append(edge.child_id)
    return [
        LineageDivision(parent_id=parent_id, child_ids=tuple(sorted(child_ids)))
        for parent_id, child_ids in sorted(children_by_parent.items())
        if len(set(child_ids)) == 2
    ]


def summarize_geff_graph(path: str | Path, dataset: str | None = None) -> GeffGraphSummary:
    """Decode and summarize one GEFF graph."""

    graph = read_geff_graph(path, dataset=dataset)
    return GeffGraphSummary(
        dataset=graph.sequence_id,
        path=Path(path),
        arrays=list_geff_arrays(path),
        node_count=len(graph.nodes),
        edge_count=len(graph.edges),
        division_count=len(extract_divisions(graph)),
        validation_errors=tuple(graph.validate()),
    )
