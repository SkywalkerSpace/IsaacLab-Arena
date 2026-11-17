# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Preserve the drawer mug's visual mesh and generate explicit hollow collision using only the standard library."""

from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path

# Coordinates belong to drawer_mug.usd, BEFORE the environment's uniform 0.1 scale.
BODY_CENTER_X = 0.1565378
BODY_HANDLE_SPLIT_X = -0.18569796
CAVITY_FLOOR_Z = -0.33234432
# Split at the inner floor fillet and changes in the inner wall profile.
WALL_LEVELS = (CAVITY_FLOOR_Z, -0.29077509, -0.20145115, 0.30252904, 0.39299562)
WALL_SEGMENTS = 20
HANDLE_CENTER_X = -0.28
HANDLE_SEGMENTS = 20
EPSILON = 1.0e-10


def _array(source: str, declaration: str) -> list:
    """Read a single-line numeric array from the source USDA."""
    for line in source.splitlines():
        if line.strip().startswith(declaration + " ="):
            return ast.literal_eval(line.split("=", 1)[1].strip())
    raise AssertionError(f"Missing source array: {declaration}")


def _clip(faces: list, normal: tuple[float, float, float], offset: float) -> list:
    """Clip surface polygons to the half-space dot(normal, point) >= offset."""
    result = []
    for face in faces:
        polygon = []
        previous = face[-1]
        previous_distance = sum(a * b for a, b in zip(normal, previous)) - offset
        for current in face:
            distance = sum(a * b for a, b in zip(normal, current)) - offset
            if (distance >= -EPSILON) != (previous_distance >= -EPSILON):
                fraction = previous_distance / (previous_distance - distance)
                polygon.append(tuple(a + fraction * (b - a) for a, b in zip(previous, current)))
            if distance >= -EPSILON:
                polygon.append(current)
            previous, previous_distance = current, distance
        if len(polygon) >= 3:
            result.append(polygon)
    return result


def _sector(faces: list, center_x: float, index: int, count: int, plane: str) -> list:
    """Clip surfaces to one angular sector in the body's XY or handle's XZ plane."""
    for boundary, sign in ((index, 1.0), (index + 1, -1.0)):
        angle = 2.0 * math.pi * boundary / count
        nx = -sign * math.sin(angle)
        other = sign * math.cos(angle)
        normal = (nx, other, 0.0) if plane == "xy" else (nx, 0.0, other)
        faces = _clip(faces, normal, nx * center_x)
    return faces


def _collider(name: str, faces: list) -> str:
    """Serialize a surface patch whose convex hull defines one solid collider."""
    points = []
    lookup = {}
    polygons = []
    for face in faces:
        indices = []
        for point in face:
            key = tuple(round(value, 10) for value in point)
            if key not in lookup:
                lookup[key] = len(points)
                points.append(key)
            index = lookup[key]
            if not indices or indices[-1] != index:
                indices.append(index)
        if len(indices) > 1 and indices[0] == indices[-1]:
            indices.pop()
        if len(set(indices)) < 3:
            continue
        # Clipping exactly through source vertices can leave zero-area boundary polygons.
        normal = [0.0, 0.0, 0.0]
        for first, second in zip(indices, indices[1:] + indices[:1]):
            a, b = points[first], points[second]
            normal[0] += (a[1] - b[1]) * (a[2] + b[2])
            normal[1] += (a[2] - b[2]) * (a[0] + b[0])
            normal[2] += (a[0] - b[0]) * (a[1] + b[1])
        if sum(value * value for value in normal) > EPSILON**2:
            polygons.append(indices)
    if not polygons:
        return ""
    assert len(points) >= 4, f"Degenerate collider: {name}"
    point_text = ", ".join(str(point) for point in points)
    counts = ", ".join(str(len(face)) for face in polygons)
    indices = ", ".join(str(index) for face in polygons for index in face)
    schemas = '["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxCollisionAPI"]'
    return f'''        def Mesh "{name}" (prepend apiSchemas = {schemas})
        {{
            uniform token subdivisionScheme = "none"
            bool physics:collisionEnabled = true
            uniform token physics:approximation = "convexHull"
            float physxCollision:contactOffset = 0.001
            float physxCollision:restOffset = 0
            point3f[] points = [{point_text}]
            int[] faceVertexCounts = [{counts}]
            int[] faceVertexIndices = [{indices}]
        }}
'''


def build_usda(source: str) -> str:
    """Return the unchanged visual asset with separate bottom, wall, and handle convex colliders."""
    points = _array(source, "point3f[] points")
    counts = _array(source, "int[] faceVertexCounts")
    indices = _array(source, "int[] faceVertexIndices")
    assert sum(counts) == len(indices), "Invalid source mesh topology"
    faces = []
    offset = 0
    for count in counts:
        faces.append([points[index] for index in indices[offset : offset + count]])
        offset += count

    body = _clip(faces, (1.0, 0.0, 0.0), BODY_HANDLE_SPLIT_X)
    handle = _clip(faces, (-1.0, 0.0, 0.0), -BODY_HANDLE_SPLIT_X)
    bottom = _clip(body, (0.0, 0.0, -1.0), -CAVITY_FLOOR_Z)
    colliders = [_collider("Bottom", bottom)]
    for band, (lower, upper) in enumerate(zip(WALL_LEVELS, WALL_LEVELS[1:])):
        strip = _clip(body, (0.0, 0.0, 1.0), lower)
        strip = _clip(strip, (0.0, 0.0, -1.0), -upper)
        for index in range(WALL_SEGMENTS):
            patch = _sector(strip, BODY_CENTER_X, index, WALL_SEGMENTS, "xy")
            colliders.append(_collider(f"Wall_{index:02d}_{band}", patch))
    for index in range(HANDLE_SEGMENTS):
        patch = _sector(handle, HANDLE_CENTER_X, index, HANDLE_SEGMENTS, "xz")
        collider = _collider(f"Handle_{index:02d}", patch)
        if collider:
            colliders.append(collider)

    old_root = 'def Xform "Asset" {'
    new_root = '''# Generated by assets/create_drawer_hollow_mug.py from drawer_mug.usd.
# Original visual mesh, UVs, normals and material are preserved verbatim.
# Colliders follow the source surface in 20 angular sectors and four height bands.
# Each surface patch is cooked independently as a convex hull; never hull the entire mug.
# Dimensions are authored for the drawer task's MUG_SCALE=(0.1, 0.1, 0.1).
def Xform "Asset" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]) {
    bool physics:rigidBodyEnabled = true
    bool physics:kinematicEnabled = false
    float physics:mass = 0.15
    bool physxRigidBody:enableSpeculativeCCD = true
    int physxRigidBody:solverPositionIterationCount = 32
    int physxRigidBody:solverVelocityIterationCount = 8'''
    old_mesh = (
        'def Mesh "Mesh" (prepend apiSchemas = '
        '["MaterialBindingAPI", "PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]) {'
    )
    new_mesh = 'def Mesh "Mesh" (prepend apiSchemas = ["MaterialBindingAPI"]) {'
    assert source.count(old_root) == 1 and source.count(old_mesh) == 1, "Unexpected source asset layout"
    source = source.replace(old_root, new_root).replace(old_mesh, new_mesh)
    source = source.replace('        bool physics:collisionEnabled = true\n', "", 1)
    source = source.replace('        uniform token physics:approximation = "convexDecomposition"\n', "", 1)
    assert source.rstrip().endswith("}"), "Source root must be closed"
    return (
        source.rstrip()[:-1]
        + '    def Xform "Colliders" {\n        token visibility = "invisible"\n'
        + "".join(colliders)
        + "    }\n}\n"
    )


def main() -> None:
    """Generate the new mug without changing its source asset."""
    asset_dir = Path(__file__).resolve().parent / "usd"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=asset_dir / "drawer_mug.usd")
    parser.add_argument("--output", type=Path, default=asset_dir / "drawer_hollow_mug.usd")
    args = parser.parse_args()
    assert args.source.resolve() != args.output.resolve(), "Do not overwrite the original mug"
    result = build_usda(args.source.read_text())
    args.output.write_text(result)
    print(f"Generated {args.output}")


if __name__ == "__main__":
    main()
