"""Convert the bowl_7, cup_2 and mug_1 OBJ assets to individual USD files.

Uses only the Python standard library. The .usd files contain USDA (text USD),
including geometry, face-varying UVs / normals and a UsdPreviewSurface material.
Textures remain external references; keep the original texture files available.
No axis conversion, rescaling, collision geometry or rigid-body physics is added.

Usage:
    python scripts/convert_obj_to_usd.py
    python scripts/convert_obj_to_usd.py --output-dir /path/to/output
    python scripts/convert_obj_to_usd.py /path/to/model.obj --output-dir /tmp/usd
"""

import argparse
import math
import os
from pathlib import Path


ASSET_ROOT = (
    Path(__file__).resolve().parents[1]
    / "dexmimicgen/models/assets/objects/objaverse"
)
DEFAULT_INPUTS = [
    ASSET_ROOT / name / "visual/model_normalized_0.obj"
    for name in ("bowl_7", "cup_2", "mug_1")
]


def obj_index(value, size):
    index = int(value)
    if index == 0 or not -size <= index <= size:
        raise ValueError(f"Invalid OBJ index {index} for {size} entries")
    return index - 1 if index > 0 else size + index


def read_obj(path):
    vertices, uvs, normals, faces = [], [], [], []
    libraries, materials = [], set()
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            fields = line.partition("#")[0].split()
            if not fields:
                continue
            kind, values = fields[0], fields[1:]
            if kind in ("v", "vt", "vn"):
                width = 2 if kind == "vt" else 3
                row = tuple(float(value) for value in values[:width])
                if len(row) != width or not all(math.isfinite(v) for v in row):
                    raise ValueError(f"Invalid {kind} record in {path}")
                {"v": vertices, "vt": uvs, "vn": normals}[kind].append(row)
            elif kind == "mtllib":
                libraries.append(path.parent / " ".join(values))
            elif kind == "usemtl":
                materials.add(" ".join(values))
            elif kind == "f":
                face = []
                for value in values:
                    parts = value.split("/")
                    vertex = obj_index(parts[0], len(vertices))
                    uv = obj_index(parts[1], len(uvs)) if len(parts) > 1 and parts[1] else None
                    normal = obj_index(parts[2], len(normals)) if len(parts) > 2 and parts[2] else None
                    face.append((vertex, uv, normal))
                if len(face) < 3:
                    raise ValueError(f"Face has fewer than three vertices in {path}")
                faces.append(face)
    if not vertices or not faces:
        raise ValueError(f"No mesh geometry in {path}")
    if len(materials) > 1:
        raise ValueError("This converter supports one material per OBJ")
    material = {}
    for library in libraries:
        active = False
        with library.open(encoding="utf-8-sig") as stream:
            for line in stream:
                fields = line.partition("#")[0].split()
                if not fields:
                    continue
                kind, values = fields[0], fields[1:]
                if kind == "newmtl":
                    active = " ".join(values) in materials
                elif active and kind == "Kd":
                    material["color"] = tuple(float(v) for v in values[:3])
                elif active and kind == "Ns":
                    material["roughness"] = min(1.0, math.sqrt(2.0 / (max(0.0, float(values[0])) + 2.0)))
                elif active and kind == "map_Kd":
                    texture = (library.parent / " ".join(values)).resolve()
                    if not texture.is_file():
                        raise FileNotFoundError(f"Diffuse texture not found: {texture}")
                    material["texture"] = texture
    return vertices, uvs, normals, faces, material


def usd_array(rows):
    return "[" + ", ".join(
        "(" + ", ".join(format(value, ".9g") for value in row) + ")"
        for row in rows
    ) + "]"


def convert_obj_to_usd(source, destination, up_axis="Z", meters_per_unit=1.0):
    vertices, uvs, normals, faces, material = read_obj(source)
    corners = [corner for face in faces for corner in face]
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "Asset"',
        f'    upAxis = "{up_axis}"',
        f"    metersPerUnit = {meters_per_unit}",
        ")",
        'def Xform "Asset" {',
        '    def Mesh "Mesh" (prepend apiSchemas = ["MaterialBindingAPI"]) {',
        f"        point3f[] points = {usd_array(vertices)}",
        f"        int[] faceVertexCounts = {[len(face) for face in faces]}",
        f"        int[] faceVertexIndices = {[corner[0] for corner in corners]}",
        '        uniform token subdivisionScheme = "none"',
        "        bool doubleSided = true",
        "        rel material:binding = </Asset/Material>",
    ]
    if all(corner[1] is not None for corner in corners):
        lines.extend([
            f"        texCoord2f[] primvars:st = {usd_array([uvs[c[1]] for c in corners])} (",
            '            interpolation = "faceVarying"',
            "        )",
        ])
    elif "texture" in material:
        raise ValueError(f"Textured mesh has missing UV coordinates: {source}")
    if all(corner[2] is not None for corner in corners):
        lines.extend([
            f"        normal3f[] normals = {usd_array([normals[c[2]] for c in corners])} (",
            '            interpolation = "faceVarying"',
            "        )",
        ])
    color = material.get("color", (0.8, 0.8, 0.8))
    lines.extend([
        "    }",
        '    def Material "Material" {',
        "        token outputs:surface.connect = </Asset/Material/Surface.outputs:surface>",
        '        def Shader "Surface" {',
        '            uniform token info:id = "UsdPreviewSurface"',
        f"            color3f inputs:diffuseColor = {usd_array([color])[1:-1]}",
        f'            float inputs:roughness = {material.get("roughness", 0.5):.9g}',
        "            float inputs:metallic = 0",
        "            token outputs:surface",
    ])
    if "texture" in material:
        texture_path = Path(os.path.relpath(material["texture"], destination.parent)).as_posix()
        if any(character in texture_path for character in ("@", "\n", "\r")):
            raise ValueError(f"Unsupported character in texture path: {texture_path}")
        lines.extend([
            "            color3f inputs:diffuseColor.connect = </Asset/Material/Texture.outputs:rgb>",
            "        }",
            '        def Shader "Texture" {',
            '            uniform token info:id = "UsdUVTexture"',
            f"            asset inputs:file = @{texture_path}@",
            '            token inputs:sourceColorSpace = "sRGB"',
            f"            float4 inputs:scale = ({color[0]}, {color[1]}, {color[2]}, 1)",
            "            float2 inputs:st.connect = </Asset/Material/UV.outputs:result>",
            "            float3 outputs:rgb",
            "        }",
            '        def Shader "UV" {',
            '            uniform token info:id = "UsdPrimvarReader_float2"',
            '            string inputs:varname = "st"',
            "            float2 outputs:result",
        ])
    lines.extend(["        }", "    }", "}", ""])
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing exports from accidental overwrite.
    with destination.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    print(f"{source} -> {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, help="Defaults to the three supplied OBJ assets")
    parser.add_argument("--output-dir", type=Path, help="Default: each OBJ's visual directory")
    parser.add_argument("--up-axis", choices=("Y", "Z"), default="Z", help="USD metadata only; does not rotate geometry")
    parser.add_argument("--meters-per-unit", type=float, default=1.0, help="USD metadata only; does not rescale geometry")
    args = parser.parse_args()
    if not math.isfinite(args.meters_per_unit) or args.meters_per_unit <= 0:
        parser.error("--meters-per-unit must be finite and positive")
    jobs = []
    for source in args.inputs or DEFAULT_INPUTS:
        source = source.resolve()
        name = source.parent.parent.name if source.parent.name == "visual" else source.stem
        directory = args.output_dir.resolve() if args.output_dir else source.parent
        jobs.append((source, directory / f"{name}.usd"))
    destinations = [destination for _, destination in jobs]
    if len(set(destinations)) != len(destinations):
        parser.error("Multiple inputs would produce the same output filename")
    for source, destination in jobs:
        if not source.is_file():
            parser.error(f"Input file not found: {source}")
        if destination.exists():
            parser.error(f"Output already exists (will not overwrite): {destination}")
    for source, destination in jobs:
        convert_obj_to_usd(source, destination, args.up_axis, args.meters_per_unit)


if __name__ == "__main__":
    main()
