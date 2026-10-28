#!/usr/bin/env python3
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""Export DexMimicGen Python objects as individual physics-enabled USD assets.

Run in a Python environment with this project's robosuite fork, MuJoCo, and
Isaac Lab / Isaac Sim installed (start Isaac Sim only when converting to USD):

    python scripts/convert_objects_to_usd.py --list
    python scripts/convert_objects_to_usd.py --output-dir ./object_usd
    python scripts/convert_objects_to_usd.py --objects CupObject DrawerObject
    python scripts/convert_objects_to_usd.py --mjcf-only --output-dir ./object_usd
    python scripts/convert_objects_to_usd.py --from-mjcf ./object_usd/mjcf

Each object produces NAME.usd. Layers and mesh geometry are flattened into
that file; image textures are copied to textures/NAME/. Use --no-textures for
single-file assets with constant-color materials instead of image textures.
The original object definitions and source assets are not modified.
Use --disable-numba to bypass robosuite's optional JIT acceleration when Numba
cannot import with the installed NumPy version. This affects only this process.

Most classes use their constructor defaults. StackedBoxObject uses the box
cleanup task's dimensions. BoxPatternObject uses an illustrative L-shaped
pattern, NOT a specific randomized task instance. BlenderObject is a factory:
each bundled objaverse model.xml becomes a separate BlenderObject_NAME asset,
using scale=1.0 (not the task-specific scales).

--config accepts JSON mapping export names to constructor keyword overrides:
    {"CupObject": {"add_handle": true},
     "StackedBoxObject": {"box_1_size": [0.075, 0.075, 0.02]}}
"""

import argparse
import copy
import hashlib
import importlib
import json
import shutil
import sys
import tempfile
import types
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "dexmimicgen"


def object_specs():
    """Return export names and their module, class, and constructor arguments."""
    specs = {}
    modules = {
        "composite.bin": ["Bin"],
        "composite.lid": ["Lid"],
        "composite.needle": ["NeedleObject"],
        "composite.pot_with_handles": ["PotWithHandlesObject"],
        "composite.ring_tripod": ["RingTripodObject"],
        "composite.box_pattern_object": ["BoxPatternObject"],
        "composite_body.coffee_machine": ["CoffeeMachineObject"],
        "composite_body.cup": ["CupObject"],
        "composite_body.stacked_box": ["StackedBoxObject"],
        "xml_objects": [
            "CoffeeMachineBaseObject", "CoffeeMachineBodyObject",
            "CoffeeMachineLidObject", "CoffeeMachinePodObject",
            "DrawerObject", "LongDrawerObject",
        ],
    }
    for module, classes in modules.items():
        for cls in classes:
            specs[cls] = (module, cls, {})
    specs["StackedBoxObject"][2].update(
        box_1_size=[0.075, 0.075, 0.02], box_2_size=[0.125, 0.125, 0.03]
    )
    specs["BoxPatternObject"][2].update(
        unit_size=[0.017, 0.017, 0.017], pattern=[[[1, 1], [1, 0]]]
    )
    asset_root = PACKAGE_ROOT / "models" / "assets" / "objects" / "objaverse"
    for xml in sorted(asset_root.glob("*/model.xml")):
        name = "BlenderObject_" + xml.parent.name
        specs[name] = ("xml_objects", "BlenderObject", {"mjcf_path": str(xml)})
    return specs


def disable_robosuite_numba():
    """Use ordinary Python functions instead of the robosuite JIT decorator."""
    module_name = "robosuite.utils.numba"
    if module_name in sys.modules:
        raise RuntimeError("--disable-numba must be applied before importing robosuite")
    # Avoid importing Numba at all: NUMBA_DISABLE_JIT still runs its NumPy check.
    module = types.ModuleType(module_name)
    module.__package__ = "robosuite.utils"
    module.jit_decorator = lambda func: func
    sys.modules[module_name] = module
    print("[INFO] robosuite Numba acceleration disabled; using Python functions")


def load_object_packages():
    """Load only object packages, avoiding environment/robot registration."""
    # dexmimicgen.__init__ imports every task, including unrelated robot plugins.
    for name, directory in (
        ("dexmimicgen", PACKAGE_ROOT),
        ("dexmimicgen.utils", PACKAGE_ROOT / "utils"),
    ):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__file__ = str(directory / "__init__.py")
            package.__path__ = [str(directory)]
            package.__package__ = name
            sys.modules[name] = package
    models = importlib.import_module("dexmimicgen.models")
    sys.modules["dexmimicgen"].models = models


def make_mjcf(obj, name, no_textures=False):
    """Assemble a complete MJCF without a robot, table, floor, or lights."""
    root = ET.Element("mujoco", model=name)
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81")
    assets = copy.deepcopy(obj.asset)
    if no_textures:
        for material in assets.findall("material"):
            material.attrib.pop("texture", None)
        for texture in list(assets.findall("texture")):
            assets.remove(texture)
    root.append(assets)
    body = copy.deepcopy(obj.get_obj())
    # Sites describe sampling/bounds, not visible object geometry.
    for parent in body.iter():
        for site in list(parent.findall("site")):
            parent.remove(site)
    ET.SubElement(root, "worldbody").append(body)
    return root


def copy_mjcf_assets(root, directory):
    """Copy file dependencies so the intermediate MJCF can move to another host."""
    directory.mkdir(parents=True, exist_ok=True)
    for element in root.findall("./asset/*"):
        for key, value in list(element.attrib.items()):
            if key != "file" and not key.startswith("file"):
                continue
            source = Path(value).expanduser()
            if not source.is_absolute():
                raise ValueError(f"Unresolved asset path: {source}")
            if not source.is_file():
                raise FileNotFoundError(source)
            digest = hashlib.sha256(str(source).encode()).hexdigest()[:12]
            target = directory / f"{digest}_{source.name}"
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
            element.set(key, target.relative_to(directory.parent).as_posix())


def export_mjcf(name, spec, overrides, output_dir, no_textures):
    module, cls, defaults = spec
    kwargs = dict(defaults)
    kwargs.update(overrides)
    kwargs.setdefault("name", name)
    obj_class = getattr(
        importlib.import_module("dexmimicgen.models.objects." + module), cls
    )
    obj = obj_class(**kwargs)
    root = make_mjcf(obj, name, no_textures)
    directory = output_dir / "mjcf"
    directory.mkdir(parents=True, exist_ok=True)
    copy_mjcf_assets(root, directory / "assets")
    target = directory / f"{name}.xml"
    ET.ElementTree(root).write(target, encoding="utf-8", xml_declaration=True)
    # Compile before invoking the USD importer; catches missing assets and bad MJCF.
    import mujoco

    mujoco.MjModel.from_xml_path(str(target))
    return target


def flatten_usd(source, destination, texture_dir):
    """Flatten USD composition and relocate file-valued material dependencies."""
    from pxr import Sdf, Usd, UsdUtils

    stage = Usd.Stage.Open(str(source))
    if not stage:
        raise RuntimeError(f"Cannot open importer output: {source}")
    # Preserve texture paths relative to the layer that originally authored them.
    layer = UsdUtils.FlattenLayerStack(stage)
    flattened = Usd.Stage.Open(layer)
    flattened = Usd.Stage.Open(flattened.Flatten())
    for prim in flattened.Traverse():
        for attr in prim.GetAttributes():
            value = attr.Get()
            if attr.GetTypeName() == Sdf.ValueTypeNames.Asset:
                values = [value] if value else []
            elif attr.GetTypeName() == Sdf.ValueTypeNames.AssetArray:
                values = list(value) if value else []
            else:
                continue
            updated = []
            for asset in values:
                path = Path(asset.resolvedPath or asset.path)
                if path.is_file():
                    texture_dir.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256(str(path).encode()).hexdigest()[:12]
                    target = texture_dir / f"{digest}_{path.name}"
                    if path.resolve() != target.resolve():
                        shutil.copy2(path, target)
                    asset = Sdf.AssetPath(target.relative_to(destination.parent).as_posix())
                elif asset.path:
                    raise FileNotFoundError(f"Unresolved USD dependency: {asset.path}")
                updated.append(asset)
            if values:
                attr.Set(updated[0] if attr.GetTypeName() == Sdf.ValueTypeNames.Asset else updated)
    if not flattened.GetDefaultPrim():
        roots = list(flattened.GetPseudoRoot().GetChildren())
        if len(roots) != 1:
            raise RuntimeError("Importer output has no unambiguous default prim")
        flattened.SetDefaultPrim(roots[0])
    temporary = destination.with_suffix(".tmp.usd")
    try:
        if not flattened.GetRootLayer().Export(str(temporary)):
            raise RuntimeError(f"Could not export {destination}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def convert_usd(xml, output_dir, fix_base):
    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

    with tempfile.TemporaryDirectory(prefix="dexmimicgen_usd_") as work:
        cfg = MjcfConverterCfg(
            asset_path=str(xml),
            usd_dir=work,
            usd_file_name=f"{xml.stem}.usd",
            force_usd_conversion=True,
            make_instanceable=False,
            fix_base=fix_base,
            import_physics_scene=False,
        )
        # Newer importers produce a layered asset by default; flatten it below.
        if hasattr(cfg, "run_asset_transformer"):
            cfg.run_asset_transformer = False
        converter = MjcfConverter(cfg)
        destination = output_dir / f"{xml.stem}.usd"
        flatten_usd(
            Path(converter.usd_path), destination,
            output_dir / "textures" / xml.stem,
        )
    return destination


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "object_usd")
    parser.add_argument("--objects", nargs="+", help="Export names from --list; default: all")
    parser.add_argument("--list", action="store_true", help="List objects without loading dependencies")
    parser.add_argument("--config", type=Path, help="JSON constructor keyword overrides")
    parser.add_argument("--mjcf-only", action="store_true", help="Generate and validate MJCF without Isaac Sim")
    parser.add_argument("--disable-numba", action="store_true", help="Bypass robosuite JIT without importing Numba")
    parser.add_argument("--from-mjcf", type=Path, help="Convert previously generated MJCF, without robosuite")
    parser.add_argument("--no-textures", action="store_true", help="Use constant colors for single-file USD assets")
    parser.add_argument("--fix-base", action="store_true", help="Fix the imported root to the world")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing output files")
    args = parser.parse_args()
    if args.from_mjcf and (args.mjcf_only or args.config or args.no_textures):
        parser.error("--from-mjcf cannot be combined with --mjcf-only, --config, or --no-textures")
    return parser, args


def main():
    parser, args = parse_args()
    specs = object_specs()
    if args.from_mjcf:
        specs = {p.stem: p.resolve() for p in sorted(args.from_mjcf.expanduser().glob("*.xml"))}
        if not specs:
            parser.error("No XML files found in --from-mjcf directory")
    if args.list:
        for name in specs:
            print(name)
        return 0
    names = args.objects or list(specs)
    unknown = set(names) - specs.keys()
    if unknown:
        parser.error("Unknown objects: " + ", ".join(sorted(unknown)))
    overrides = json.loads(args.config.read_text()) if args.config else {}
    if not isinstance(overrides, dict) or any(not isinstance(v, dict) for v in overrides.values()):
        parser.error("--config must map export names to keyword argument dictionaries")
    if overrides.keys() - specs.keys():
        parser.error("Unknown names in --config: " + ", ".join(sorted(overrides.keys() - specs.keys())))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.disable_numba:
        disable_robosuite_numba()
    if not args.from_mjcf:
        load_object_packages()
    app = None
    failures = []
    completed = skipped = 0
    try:
        if not args.mjcf_only:
            try:
                from isaaclab.app import AppLauncher

                app = AppLauncher(headless=True).app
            except ImportError as exc:
                raise RuntimeError(
                    "USD conversion requires an Isaac Lab / Isaac Sim Python environment. "
                    "Use --mjcf-only here, then --from-mjcf on that environment."
                ) from exc
        for name in dict.fromkeys(names):
            destination = output_dir / (f"mjcf/{name}.xml" if args.mjcf_only else f"{name}.usd")
            if destination.exists() and not args.overwrite:
                print(f"[SKIP] {destination} (use --overwrite to regenerate)")
                skipped += 1
                continue
            try:
                xml = specs[name] if args.from_mjcf else export_mjcf(
                    name, specs[name], overrides.get(name, {}), output_dir, args.no_textures
                )
                result = xml if args.mjcf_only else convert_usd(xml, output_dir, args.fix_base)
                print(f"[OK] {name} -> {result}")
                completed += 1
            except Exception as exc:
                failures.append(name)
                print(f"[FAIL] {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        if app is not None:
            app.close()
    print(f"Completed: {completed}; skipped: {skipped}; failed: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
