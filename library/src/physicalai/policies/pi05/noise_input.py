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
    python -m physicalai.policies.pi05.noise_input EXPORT_DIR --restore
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import inspect
import json
import math
import os
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
_STAGED_DIRNAME = ".noise-input-new"  # new files, written completely before anything is replaced
_ASIDE_DIRNAME = ".noise-input-old"  # the replaced files while converting, when no backup is kept
_SCRATCH_DIRNAME = ".noise-input-swap"  # links renamed over the export's files one at a time
_RANDOM_OP = "RandomUniform"
# Ops OpenVINO builds Box-Muller from; the search for the noise stays inside these.
_ELEMENTWISE = {"Add", "Subtract", "Multiply", "Divide", "Log", "Cos", "Sin", "Sqrt", "Power", "Negative", "Convert"}


class RollbackError(RuntimeError):
    """A failed in-place conversion could not put the original files back; nothing was deleted."""


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
    return [op for op in model.get_ordered_ops() if op.get_type_name().startswith("Random")]


def _consumers(node: ov.Node) -> list[ov.Node]:
    return [target.get_node() for output in node.outputs() for target in output.get_target_inputs()]


def _varying(model: ov.Model) -> dict[str, bool]:
    """Map each op to whether it depends on a model input or the random op.

    Returns:
        Op name to True if it varies, False if it acts as a constant.
    """
    varying: dict[str, bool] = {}
    for node in model.get_ordered_ops():
        sources = [inp.get_source_output().get_node().get_friendly_name() for inp in node.inputs()]
        varying[node.get_friendly_name()] = node.get_type_name() in {"Parameter", _RANDOM_OP} or any(
            varying[source] for source in sources
        )
    return varying


def _source(node: ov.Node, index: int) -> ov.Node:
    return node.input_value(index).get_node()


def _scaled_operand(node: ov.Node, factor: float) -> ov.Node | None:
    """Return ``x`` if ``node`` is ``x * factor`` with a constant factor, else None."""
    if node.get_type_name() != "Multiply":
        return None
    for operand, other in ((_source(node, 0), _source(node, 1)), (_source(node, 1), _source(node, 0))):
        constant = _source(other, 0) if other.get_type_name() == "Convert" else other
        if constant.get_type_name() != "Constant":
            continue
        value = constant.get_data()
        if value.size and all(math.isclose(v, factor, rel_tol=1e-5) for v in value.ravel().tolist()):
            return operand
    return None


def _is_box_muller(node: ov.Node, rand_name: str) -> bool:
    """Return whether ``node`` is exactly ``sqrt(-2 ln u) * cos(2 pi u)`` (or ``sin``) of the random op ``u``."""
    if node.get_type_name() != "Multiply":
        return False
    for root, trig in ((_source(node, 0), _source(node, 1)), (_source(node, 1), _source(node, 0))):
        if root.get_type_name() != "Sqrt" or trig.get_type_name() not in {"Cos", "Sin"}:
            continue
        log = _scaled_operand(_source(root, 0), -2.0)
        angle = _scaled_operand(_source(trig, 0), 2 * math.pi)
        if (
            log is not None
            and log.get_type_name() == "Log"
            and _source(log, 0).get_friendly_name() == rand_name
            and angle is not None
            and angle.get_friendly_name() == rand_name
        ):
            return True
    return False


def _computed_by(out: ov.Node, nodes: dict[str, ov.Node]) -> list[ov.Node]:
    """Return ``out`` and the ops among ``nodes`` it is computed from."""
    computed: dict[str, ov.Node] = {}
    stack = [out]
    while stack:
        node = stack.pop()
        if node.get_friendly_name() in computed:
            continue
        computed[node.get_friendly_name()] = node
        for inp in node.inputs():
            source = inp.get_source_output().get_node()
            if source.get_friendly_name() in nodes:
                stack.append(source)
    return list(computed.values())


def _box_muller_output(model: ov.Model) -> tuple[ov.Node, list[ov.Node]]:
    """Find the op that outputs the start noise OpenVINO builds from ``RandomUniform``.

    OpenVINO's ONNX frontend turns ``RandomNormalLike`` into Box-Muller:
    ``sqrt(-2 ln u) * cos(2 pi u)`` with a single ``RandomUniform`` u. Starting from
    that op, this follows elementwise ops of the same shape whose other inputs are
    constant, and looks for the one ``Multiply`` that computes exactly that, constants
    included. Ops after it (casts, the model's own arithmetic) stay.

    Returns:
        The Box-Muller output op and the ops that compute it.

    Raises:
        ValueError: If the graph does not have exactly one ``RandomUniform`` feeding
            exactly one Box-Muller product.
    """
    rand = _random_ops(model)
    if [op.get_type_name() for op in rand] != [_RANDOM_OP]:
        msg = f"expected one {_RANDOM_OP} op in the graph, found {[op.get_type_name() for op in rand]}"
        raise ValueError(msg)
    varying = _varying(model)
    shape = rand[0].get_output_partial_shape(0)
    nodes = {rand[0].get_friendly_name(): rand[0]}
    candidates = []
    frontier = [rand[0]]
    while frontier:
        next_frontier = []
        for consumer in (c for node in frontier for c in _consumers(node)):
            name = consumer.get_friendly_name()
            sources = [inp.get_source_output().get_node().get_friendly_name() for inp in consumer.inputs()]
            if (
                name in nodes
                or consumer.get_type_name() not in _ELEMENTWISE
                or consumer.get_output_partial_shape(0) != shape
                or not all(source in nodes or not varying[source] for source in sources)
            ):
                continue
            nodes[name] = consumer
            next_frontier.append(consumer)
            if _is_box_muller(consumer, rand[0].get_friendly_name()):
                candidates.append(consumer)
        frontier = next_frontier

    if len(candidates) != 1:
        msg = f"the noise is not the Box-Muller pattern OpenVINO builds (found {len(candidates)} candidates)"
        raise ValueError(msg)
    return candidates[0], _computed_by(candidates[0], nodes)


def _replace_noise_with_input(model: ov.Model) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Replace the in-graph noise with a ``noise`` model input, in place.

    Returns:
        The noise shape without the batch dimension, and the names of the replaced ops.

    Raises:
        ValueError: If the noise has a dynamic shape, or the random op also feeds
            something other than the noise.
    """
    from openvino import opset13 as ops  # noqa: PLC0415

    out, computed = _box_muller_output(model)
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
    if _random_ops(model):
        msg = f"the {_RANDOM_OP} op also feeds parts of the graph other than the start noise"
        raise ValueError(msg)
    return tuple(int(d) for d in shape.to_shape()[1:]), tuple(node.get_friendly_name() for node in computed)


def _is_pi05_spec(spec: dict[str, Any]) -> bool:
    if spec.get("class_path"):
        return str(spec["class_path"]).rsplit(".", 1)[-1].startswith("Pi05")
    return spec.get("type") == "pi05"


def _pi05_args(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the arguments the runtime passes to the manifest's ``pi05`` preprocessor.

    Those are ``init_args`` for a ``class_path`` spec (which the runtime prefers over
    ``type``) and the spec's own fields otherwise. The dict is the manifest's own, so
    changes to it change the manifest.

    Raises:
        ValueError: If the manifest does not have exactly one ``pi05`` preprocessor.
    """
    specs = [spec for spec in manifest.get("model", {}).get("preprocessors", []) if _is_pi05_spec(spec)]
    if len(specs) != 1:
        msg = f"expected one `pi05` preprocessor in the manifest, found {len(specs)}"
        raise ValueError(msg)
    spec = specs[0]
    return spec.setdefault("init_args", {}) if spec.get("class_path") else spec


def _graph_name(manifest: dict[str, Any], where: Path) -> str:
    """Return the manifest's OpenVINO graph file name.

    Raises:
        ValueError: If it is missing or not a plain file name inside the folder.
    """
    name = manifest.get("model", {}).get("artifacts", {}).get("openvino") if isinstance(manifest, dict) else None
    if not isinstance(name, str) or name in {"", ".", ".."} or Path(name).name != name or "\\" in name:
        msg = f"{where / MANIFEST_NAME}: the OpenVINO artifact must be a file name in the folder, got {name!r}"
        raise ValueError(msg)
    return name


def read_noise_shape(export_dir: Path | str) -> list[int] | None:
    """Return the ``noise_shape`` a pi05 export's manifest gives its preprocessor, if any.

    Also raises ``ValueError`` if the manifest has no single ``pi05`` preprocessor.

    Raises:
        FileNotFoundError: If the folder has no manifest.
    """
    manifest_path = Path(export_dir) / MANIFEST_NAME
    if not manifest_path.is_file():
        msg = f"no {MANIFEST_NAME} in {export_dir}"
        raise FileNotFoundError(msg)
    return _pi05_args(json.loads(manifest_path.read_text())).get("noise_shape")


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
    """Return the export's manifest, its ``pi05`` preprocessor arguments and its graph path.

    Raises:
        FileNotFoundError: If the folder has no manifest or no OpenVINO graph.
    """
    manifest_path = export_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        msg = f"no {MANIFEST_NAME} in {export_dir}"
        raise FileNotFoundError(msg)
    manifest = json.loads(manifest_path.read_text())
    graph_path = export_dir / _graph_name(manifest, export_dir)
    if not graph_path.is_file():
        msg = f"{export_dir} is not an OpenVINO export (no {graph_path.name})"
        raise FileNotFoundError(msg)
    return manifest, _pi05_args(manifest), graph_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_checksums(source: Path, files_dir: Path, names: list[str], dest: Path) -> None:
    """Write ``source`` (a ``SHA256SUMS``) to ``dest`` with the lines for ``names`` hashed from ``files_dir``."""
    lines = []
    for line in source.read_text(encoding="utf-8").splitlines():
        listed = line.split(maxsplit=1)[-1].lstrip("*") if line.strip() else ""
        # sha256sum also accepts "./name"; keep the spelling the listing uses.
        name = listed.removeprefix("./")
        lines.append(f"{_sha256(files_dir / name)}  {listed}" if name in names else line)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save(model: ov.Model, manifest: dict[str, Any], folder: Path, graph_name: str) -> None:
    import openvino as ov  # noqa: PLC0415

    # Keep every constant as it was (INT8, fp16 or fp32).
    ov.save_model(model, folder / graph_name, compress_to_fp16=False)
    (folder / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))


def _rewritten_names(graph_name: str) -> list[str]:
    return [graph_name, Path(graph_name).with_suffix(".bin").name, MANIFEST_NAME]


def _link_or_copy(src: Path, dst: Path) -> None:
    """Give ``dst`` the content of ``src``: a hard link (instant, no extra disk) where possible."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _replace_from(source: Path, target: Path, names: list[str], scratch: Path) -> None:
    """Replace ``target/name`` with the content of ``source/name`` for each name, leaving ``source`` complete.

    Each file is linked into ``scratch`` first and then renamed over the target, so
    every target file is either the old or the new one at any moment, even if the
    process is killed.
    """
    shutil.rmtree(scratch, ignore_errors=True)  # links left by an interrupted earlier run
    scratch.mkdir(parents=True)
    try:
        for name in names:
            _link_or_copy(source / name, scratch / name)
        for name in names:
            (scratch / name).replace(target / name)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _copy_other_files(export_dir: Path, output_dir: Path, graph_name: str) -> None:
    """Copy everything in the export except the files the conversion rewrites."""
    skipped = {*_rewritten_names(graph_name), _STAGED_DIRNAME, _ASIDE_DIRNAME, _SCRATCH_DIRNAME}
    items = [item for item in export_dir.iterdir() if item.name not in skipped]
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        if item.is_dir():
            shutil.copytree(item, output_dir / item.name)
        else:
            shutil.copy2(item, output_dir / item.name)


def _overlaps(a: Path, b: Path) -> bool:
    a, b = a.resolve(), b.resolve()
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def convert_openvino_export(
    export_dir: Path | str,
    output_dir: Path | str | None = None,
    *,
    backup: bool = True,
    backup_dir: Path | str | None = None,
) -> ConversionResult:
    """Make a pi05 OpenVINO export take its start noise as a ``noise`` input.

    In place by default: the original graph, weights and manifest (and
    ``SHA256SUMS``, if any) are kept in a backup folder (``<export_dir>.orig`` unless
    ``backup_dir`` is given) that is complete before any file of the export changes,
    and :func:`restore_openvino_export` puts them back. Each file is then replaced
    with an atomic rename, so the export never lacks a file. With ``output_dir`` the
    converted export is written there and the source is left untouched. An export
    that already takes the noise as input is left as it is.

    Args:
        export_dir: OpenVINO export folder (``manifest.json`` plus the graph it names).
        output_dir: Optional empty or new folder to write the converted export to.
        backup: Keep the replaced files when converting in place.
        backup_dir: Where to keep them. Must be new or empty.

    Returns:
        What was done and where the resulting export is.

    Raises:
        ValueError: If the export is not a pi05 export with in-graph noise, or the
            output or backup folder overlaps the export.
        RollbackError: If replacing the files failed and putting the originals back
            failed too. The original files are kept; the message says where.
    """
    import openvino as ov  # noqa: PLC0415

    export_dir = Path(export_dir).resolve()
    manifest, pi05_args, graph_path = _read_export(export_dir)
    model = ov.Core().read_model(graph_path)
    existing = _noise_input_shape(model)
    if existing is not None:
        if pi05_args.get("noise_shape") is None:
            msg = f"{graph_path} takes `{NOISE}` but the manifest's pi05 preprocessor has no `noise_shape`"
            raise ValueError(msg)
        return ConversionResult(ConversionStatus.ALREADY_CONVERTED, export_dir, existing)
    if not _random_ops(model):
        msg = f"{graph_path} samples no start noise (exported with use_random_input_noise=False?)"
        raise ValueError(msg)

    out, keep, aside = _target_folders(export_dir, output_dir, backup=backup, backup_dir=backup_dir)
    noise_shape, replaced_ops = _replace_noise_with_input(model)
    pi05_args["noise_shape"] = list(noise_shape)
    graph_name = graph_path.name
    names = _rewritten_names(graph_name)
    if out is not None:
        _copy_other_files(export_dir, out, graph_name)
        _save(model, manifest, out, graph_name)
        if (out / CHECKSUMS_NAME).is_file():
            _write_checksums(out / CHECKSUMS_NAME, out, names, out / CHECKSUMS_NAME)
        return ConversionResult(ConversionStatus.CONVERTED, out, noise_shape, replaced_ops)

    staged_dir = export_dir / _STAGED_DIRNAME
    shutil.rmtree(staged_dir, ignore_errors=True)
    staged_dir.mkdir()
    try:
        # Write every new file, the checksums included, before touching the export.
        _save(model, manifest, staged_dir, graph_name)
        if (export_dir / CHECKSUMS_NAME).is_file():
            _write_checksums(export_dir / CHECKSUMS_NAME, staged_dir, names, staged_dir / CHECKSUMS_NAME)
            names = [*names, CHECKSUMS_NAME]
        # Release the original weights (memory-mapped by read_model) before replacing them.
        del model
        gc.collect()
        _swap_in(export_dir, staged_dir, aside, names)
    except RollbackError:
        raise  # keep every file where it is: the message says where the originals are
    except BaseException:
        shutil.rmtree(staged_dir, ignore_errors=True)
        shutil.rmtree(aside, ignore_errors=True)  # the export has its original files again
        raise
    shutil.rmtree(staged_dir, ignore_errors=True)
    if keep is None:
        shutil.rmtree(aside, ignore_errors=True)
    return ConversionResult(ConversionStatus.CONVERTED, export_dir, noise_shape, replaced_ops, keep)


def _target_folders(
    export_dir: Path,
    output_dir: Path | str | None,
    *,
    backup: bool,
    backup_dir: Path | str | None,
) -> tuple[Path | None, Path | None, Path]:
    """Check the folders a conversion writes to.

    Returns:
        The output folder (or None), the backup folder (or None), and where the
        originals go while swapping.

    Raises:
        ValueError: If the output or backup folder overlaps the export.
    """
    out = Path(output_dir).resolve() if output_dir is not None else None
    keep: Path | None = None
    if out is not None:
        if _overlaps(out, export_dir):
            msg = f"the output folder {out} must be outside the export {export_dir}, and the other way round"
            raise ValueError(msg)
        _require_empty(out)
    elif backup:
        keep = Path(backup_dir).resolve() if backup_dir is not None else default_backup_dir(export_dir)
        if _overlaps(keep, export_dir):
            msg = f"the backup folder {keep} must be outside the export {export_dir}, and the other way round"
            raise ValueError(msg)
        _require_empty(keep)
    aside = keep if keep is not None else export_dir / _ASIDE_DIRNAME
    _require_empty(aside)
    return out, keep, aside


def _swap_in(export_dir: Path, staged_dir: Path, aside: Path, names: list[str]) -> None:
    """Replace the export's files with the staged ones, keeping the originals in ``aside``.

    ``aside`` gets a complete copy of the originals (hard links) before any of them
    changes. Each file is then replaced with one atomic rename, so the export never
    lacks a file; on failure the originals are put back the same way.

    Raises:
        RollbackError: If putting the originals back fails as well.
    """
    aside.mkdir(parents=True, exist_ok=True)
    for name in names:
        _link_or_copy(export_dir / name, aside / name)
    try:
        for name in names:
            (staged_dir / name).replace(export_dir / name)
    except BaseException:
        try:
            _replace_from(aside, export_dir, names, export_dir / _SCRATCH_DIRNAME)
        except BaseException as rollback_exc:
            msg = (
                f"replacing the files in {export_dir} failed and so did putting the originals back. "
                f"The original files are in {aside}; copy them back into the export."
            )
            raise RollbackError(msg) from rollback_exc
        raise


def default_backup_dir(export_dir: Path | str) -> Path:
    """Return where an in-place conversion keeps the replaced files by default: ``<export_dir>.orig``."""
    export_dir = Path(export_dir).resolve()
    return export_dir.with_name(export_dir.name + BACKUP_SUFFIX)


def _backup_graph_name(backup: Path) -> str | None:
    try:
        manifest = json.loads((backup / MANIFEST_NAME).read_text())
        return _graph_name(manifest, backup)
    except (OSError, ValueError):
        return None


def has_backup(export_dir: Path | str, backup_dir: Path | str | None = None) -> bool:
    """Return whether a backup that :func:`restore_openvino_export` can put back exists."""
    backup = Path(backup_dir) if backup_dir is not None else default_backup_dir(export_dir)
    graph_name = _backup_graph_name(backup)
    return graph_name is not None and all((backup / name).is_file() for name in _rewritten_names(graph_name))


def restore_openvino_export(export_dir: Path | str, backup_dir: Path | str | None = None) -> Path:
    """Undo an in-place :func:`convert_openvino_export` from its backup.

    Each export file is replaced with an atomic rename, also when the export lacks
    it, and the backup stays complete until every file is back, so a failed or
    interrupted restore can simply be run again. The backup folder is then removed.

    Args:
        export_dir: The converted OpenVINO export folder.
        backup_dir: Where the conversion kept the replaced files (default ``<export_dir>.orig``).

    Returns:
        The backup folder the files came from.

    Raises:
        FileNotFoundError: If the export folder does not exist, or the backup is missing or incomplete.
        ValueError: If the backup folder overlaps the export.
    """
    export_dir = Path(export_dir).resolve()
    backup = Path(backup_dir).resolve() if backup_dir is not None else default_backup_dir(export_dir)
    if not export_dir.is_dir():
        msg = f"no export folder {export_dir}"
        raise FileNotFoundError(msg)
    if _overlaps(backup, export_dir):
        msg = f"the backup folder {backup} must be outside the export {export_dir}, and the other way round"
        raise ValueError(msg)
    graph_name = _backup_graph_name(backup)
    if graph_name is None or not has_backup(export_dir, backup):
        msg = f"no complete backup in {backup} (graph, weights and manifest)"
        raise FileNotFoundError(msg)
    names = _rewritten_names(graph_name)
    if (backup / CHECKSUMS_NAME).is_file():
        names.append(CHECKSUMS_NAME)
    _replace_from(backup, export_dir, names, export_dir / _SCRATCH_DIRNAME)
    for name in names:
        (backup / name).unlink()
    with contextlib.suppress(OSError):
        backup.rmdir()
    return backup


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
    parser.add_argument("--restore", action="store_true", help="undo an in-place conversion from its backup")
    args = parser.parse_args(argv)
    if args.restore:
        backup = restore_openvino_export(args.export_dir, args.backup_dir)
        print(f"restored: {args.export_dir} (from {backup})")  # noqa: T201
        return
    result = convert_openvino_export(args.export_dir, args.out, backup=not args.no_backup, backup_dir=args.backup_dir)
    print(f"{result.status.value}: {result.export_dir} (noise_shape {list(result.noise_shape)})")  # noqa: T201
    if result.backup_dir is not None:
        print(f"original files kept in {result.backup_dir}")  # noqa: T201
    if not runtime_draws_noise():
        print("warning: the installed physicalai runtime cannot draw the noise yet; upgrade physicalai")  # noqa: T201


if __name__ == "__main__":
    main()
