# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Convert an existing pi05 OpenVINO export so it takes its start noise as an input.

pi05 exports made before the noise became a model input sample it inside the graph.
OpenVINO's ONNX frontend builds that sampling from one ``RandomUniform`` op (so the
noise is not N(0, 1)), and the GPU plugin returns the same ``RandomUniform`` values on
every call of a compiled model, so the policy starts every denoising run from the
same noise.

This module rewrites such an export into what a current export produces: the ops
that compute the noise are replaced by a ``noise`` input, and the manifest's ``pi05``
preprocessor gets ``noise_shape`` so the runtime draws fresh N(0, 1) noise on every
call. No checkpoint or re-export is needed, and the weights are saved unchanged.

    python -m physicalai.policies.pi05.noise_input EXPORT_DIR [--out DIR] [--no-backup]
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from physicalai.data.constants import NOISE

if TYPE_CHECKING:
    import openvino as ov

MANIFEST_NAME = "manifest.json"
CHECKSUMS_NAME = "SHA256SUMS"  # optional `sha256sum` listing some copies of an export carry
BACKUP_SUFFIX = ".orig"  # default backup folder: <export_dir>.orig next to the export
_TMP_DIRNAME = ".noise-input-tmp"
_RANDOM_OPS = ("RandomUniform", "RandomNormal")
# Ops the in-graph sampling (Box-Muller) is built from; the replaced subgraph stays inside these.
_ELEMENTWISE = {"Add", "Subtract", "Multiply", "Divide", "Log", "Cos", "Sin", "Sqrt", "Power", "Negative", "Convert"}


class ConversionStatus(StrEnum):
    """Outcome of :func:`convert_openvino_export`."""

    CONVERTED = "converted"
    ALREADY_CONVERTED = "already_converted"


@dataclass(frozen=True)
class ConversionResult:
    """What :func:`convert_openvino_export` did.

    Attributes:
        status: Whether the export was converted or already took the noise as input.
        export_dir: Folder holding the resulting export.
        noise_shape: Shape of one noise sample, without the batch dimension.
        replaced_ops: Names of the graph ops the ``noise`` input replaced (empty if
            nothing was converted).
        backup_dir: Folder holding the replaced files, if any were kept.
    """

    status: ConversionStatus
    export_dir: Path
    noise_shape: tuple[int, ...]
    replaced_ops: tuple[str, ...] = ()
    backup_dir: Path | None = None


def runtime_draws_noise() -> bool:
    """Return whether the installed runtime's ``pi05`` preprocessor can draw the noise.

    Converted exports need openvinotoolkit/physicalai#442, whose preprocessor accepts
    ``noise_shape``.
    """
    from physicalai.inference.preprocessors.pi05 import Pi05Preprocessor  # noqa: PLC0415

    return "noise_shape" in inspect.signature(Pi05Preprocessor.__init__).parameters


def _random_ops(model: ov.Model) -> list[ov.Node]:
    return [op for op in model.get_ordered_ops() if op.get_type_name() in _RANDOM_OPS]


def _consumers(node: ov.Node) -> list[ov.Node]:
    return [target.get_node() for output in node.outputs() for target in output.get_target_inputs()]


def _noise_subgraph(model: ov.Model) -> tuple[list[ov.Node], ov.Node]:
    """Find the ops that turn the graph's single random op into the start noise.

    Starting from the random op, a consumer joins the subgraph when it is an
    elementwise op with the noise's shape whose inputs are either already in the
    subgraph or constant (they depend on no model input and no random op).

    Returns:
        The subgraph's ops and the one op whose output leaves it (the start noise).

    Raises:
        ValueError: If the graph has no single random op or no single noise output.
    """
    rand = _random_ops(model)
    if len(rand) != 1:
        msg = f"expected one random op in the graph, found {[op.get_type_name() for op in rand]}"
        raise ValueError(msg)

    varying: dict[str, bool] = {}
    for node in model.get_ordered_ops():
        sources = [inp.get_source_output().get_node().get_friendly_name() for inp in node.inputs()]
        varying[node.get_friendly_name()] = node.get_type_name() in {"Parameter", *_RANDOM_OPS} or any(
            varying[source] for source in sources
        )

    shape = rand[0].get_output_partial_shape(0)
    chain = {rand[0].get_friendly_name(): rand[0]}
    frontier = [rand[0]]
    while frontier:
        next_frontier = []
        for node in frontier:
            for consumer in _consumers(node):
                name = consumer.get_friendly_name()
                if name in chain or consumer.get_type_name() not in _ELEMENTWISE:
                    continue
                if consumer.get_output_partial_shape(0) != shape:
                    continue
                sources = [inp.get_source_output().get_node().get_friendly_name() for inp in consumer.inputs()]
                if all(source in chain or not varying[source] for source in sources):
                    chain[name] = consumer
                    next_frontier.append(consumer)
        frontier = next_frontier

    outputs = [node for node in chain.values() if any(c.get_friendly_name() not in chain for c in _consumers(node))]
    if len(outputs) != 1:
        msg = f"expected one noise output, found {[node.get_friendly_name() for node in outputs]}"
        raise ValueError(msg)
    return list(chain.values()), outputs[0]


def _replace_noise_with_input(model: ov.Model) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Replace the in-graph noise with a ``noise`` model input, in place.

    Returns:
        The noise shape without the batch dimension, and the names of the replaced ops.

    Raises:
        ValueError: If the noise does not have a static shape.
    """
    from openvino import opset13 as ops  # noqa: PLC0415

    chain, out = _noise_subgraph(model)
    shape = out.get_output_partial_shape(0)
    if not shape.is_static:
        msg = f"the start noise has a dynamic shape {shape}"
        raise ValueError(msg)
    param = ops.parameter(shape, out.get_output_element_type(0), name=NOISE)
    param.output(0).get_tensor().set_names({NOISE})
    for target in list(out.output(0).get_target_inputs()):
        target.replace_source_output(param.output(0))
    model.add_parameters([param])
    model.validate_nodes_and_infer_types()
    return tuple(int(d) for d in shape.to_shape()[1:]), tuple(node.get_friendly_name() for node in chain)


def _pi05_spec(manifest: dict[str, Any]) -> dict[str, Any]:
    specs = [spec for spec in manifest.get("model", {}).get("preprocessors", []) if spec.get("type") == "pi05"]
    if len(specs) != 1:
        msg = f"expected one `pi05` preprocessor in the manifest, found {len(specs)}"
        raise ValueError(msg)
    return specs[0]


def _noise_input_shape(model: ov.Model) -> tuple[int, ...] | None:
    for inp in model.inputs:
        if NOISE in inp.get_names():
            return tuple(int(d) for d in inp.get_partial_shape().to_shape()[1:])
    return None


def _require_empty(folder: Path) -> None:
    if folder.exists() and any(folder.iterdir()):
        msg = f"{folder} is not empty"
        raise FileExistsError(msg)


def _read_export(export_dir: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Return the export's manifest, its ``pi05`` preprocessor spec and its graph path.

    Raises:
        FileNotFoundError: If the folder has no manifest or no OpenVINO graph.
    """
    manifest_path = export_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        msg = f"no {MANIFEST_NAME} in {export_dir}"
        raise FileNotFoundError(msg)
    manifest = json.loads(manifest_path.read_text())
    graph_name = manifest.get("model", {}).get("artifacts", {}).get("openvino")
    if not graph_name or not (export_dir / graph_name).is_file():
        msg = f"{export_dir} is not an OpenVINO export (manifest artifact: {graph_name!r})"
        raise FileNotFoundError(msg)
    return manifest, _pi05_spec(manifest), export_dir / graph_name


def _refresh_checksums(folder: Path, names: list[str]) -> None:
    """Update the lines of an existing ``SHA256SUMS`` that list the rewritten files."""
    checksums = folder / CHECKSUMS_NAME
    if not checksums.is_file():
        return
    lines = []
    for line in checksums.read_text().splitlines():
        name = line.split(maxsplit=1)[-1].lstrip("*") if line.strip() else ""
        if name in names:
            digest = hashlib.sha256()
            with (folder / name).open("rb") as f:
                for block in iter(lambda: f.read(1 << 24), b""):
                    digest.update(block)
            line = f"{digest.hexdigest()}  {name}"  # noqa: PLW2901
        lines.append(line)
    checksums.write_text("\n".join(lines) + "\n")


def _save(model: ov.Model, manifest: dict[str, Any], folder: Path, graph_name: str) -> None:
    import openvino as ov  # noqa: PLC0415

    # Keep every constant as it was (INT8, fp16 or fp32).
    ov.save_model(model, folder / graph_name, compress_to_fp16=False)
    (folder / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))


def _rewritten_names(graph_name: str) -> list[str]:
    return [graph_name, Path(graph_name).with_suffix(".bin").name, MANIFEST_NAME]


def _copy_other_files(export_dir: Path, output_dir: Path, graph_name: str) -> None:
    """Copy everything in the export except the files the conversion rewrites."""
    skipped = {*_rewritten_names(graph_name), _TMP_DIRNAME}
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in export_dir.iterdir():
        if item.name in skipped:
            continue
        if item.is_dir():
            shutil.copytree(item, output_dir / item.name)
        else:
            shutil.copy2(item, output_dir / item.name)


def _swap_in(export_dir: Path, staged_dir: Path, backup_dir: Path | None, graph_name: str) -> None:
    """Move the staged files over the export's, moving the old ones to ``backup_dir`` if given."""
    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)
    for name in _rewritten_names(graph_name):
        if backup_dir is not None:
            shutil.move(export_dir / name, backup_dir / name)
        (staged_dir / name).replace(export_dir / name)


def convert_openvino_export(
    export_dir: Path | str,
    output_dir: Path | str | None = None,
    *,
    backup: bool = True,
    backup_dir: Path | str | None = None,
) -> ConversionResult:
    """Make a pi05 OpenVINO export take its start noise as a ``noise`` input.

    In place by default: the original graph, weights and manifest are moved to a
    backup folder (``<export_dir>.orig`` unless ``backup_dir`` is given) before the
    new ones take their place; moving them back restores the export. With
    ``output_dir`` the converted export is written there and the source is left
    untouched. An export that already takes the noise as input is left as it is.

    Args:
        export_dir: OpenVINO export folder (``manifest.json`` plus the graph it names).
        output_dir: Optional empty or new folder to write the converted export to.
        backup: Keep the replaced files when converting in place.
        backup_dir: Where to keep them. Must be new or empty.

    Returns:
        What was done and where the resulting export is.

    Raises:
        ValueError: If the export is not a pi05 export with in-graph noise.
    """
    import openvino as ov  # noqa: PLC0415

    export_dir = Path(export_dir)
    manifest, spec, graph_path = _read_export(export_dir)
    model = ov.Core().read_model(graph_path)
    existing = _noise_input_shape(model)
    if existing is not None:
        if spec.get("noise_shape") is None:
            msg = f"{graph_path} takes `{NOISE}` but the manifest's pi05 preprocessor has no `noise_shape`"
            raise ValueError(msg)
        return ConversionResult(ConversionStatus.ALREADY_CONVERTED, export_dir, existing)
    if not _random_ops(model):
        msg = f"{graph_path} samples no start noise (exported with use_random_input_noise=False?)"
        raise ValueError(msg)

    out = Path(output_dir) if output_dir is not None else None
    keep: Path | None = None
    if out is not None:
        _require_empty(out)
    elif backup:
        keep = Path(backup_dir) if backup_dir is not None else export_dir.with_name(export_dir.name + BACKUP_SUFFIX)
        _require_empty(keep)

    noise_shape, replaced_ops = _replace_noise_with_input(model)
    spec["noise_shape"] = list(noise_shape)
    graph_name = graph_path.name
    if out is not None:
        _copy_other_files(export_dir, out, graph_name)
        _save(model, manifest, out, graph_name)
        _refresh_checksums(out, _rewritten_names(graph_name))
        return ConversionResult(ConversionStatus.CONVERTED, out, noise_shape, replaced_ops)

    # Write the new files completely before replacing anything.
    staged_dir = export_dir / _TMP_DIRNAME
    shutil.rmtree(staged_dir, ignore_errors=True)
    staged_dir.mkdir()
    try:
        _save(model, manifest, staged_dir, graph_name)
        # Release the original weights (memory-mapped by read_model) before moving them.
        del model
        gc.collect()
        _swap_in(export_dir, staged_dir, keep, graph_name)
    finally:
        shutil.rmtree(staged_dir, ignore_errors=True)
    _refresh_checksums(export_dir, _rewritten_names(graph_name))
    return ConversionResult(ConversionStatus.CONVERTED, export_dir, noise_shape, replaced_ops, keep)


def main(argv: list[str] | None = None) -> None:
    """Command line entry point: ``python -m physicalai.policies.pi05.noise_input``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("export_dir", type=Path, help="pi05 OpenVINO export folder (with manifest.json)")
    parser.add_argument("--out", type=Path, default=None, help="write the converted export here instead")
    parser.add_argument("--no-backup", action="store_true", help="in place: do not keep the replaced files")
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="in place: keep them here (default EXPORT_DIR.orig)",
    )
    args = parser.parse_args(argv)
    result = convert_openvino_export(args.export_dir, args.out, backup=not args.no_backup, backup_dir=args.backup_dir)
    print(f"{result.status.value}: {result.export_dir} (noise_shape {list(result.noise_shape)})")  # noqa: T201
    if result.backup_dir is not None:
        print(f"original files kept in {result.backup_dir}")  # noqa: T201
    if not runtime_draws_noise():
        print("warning: the installed physicalai runtime cannot draw the noise yet; upgrade physicalai")  # noqa: T201


if __name__ == "__main__":
    main()
