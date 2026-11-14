# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate matching chamfered cup and holder USDs using only the Python standard library."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

# Dimensions in meters AFTER the pouring environment applies CUP_SCALE.
CUP_SCALE_XY = 0.7
WALL_THICKNESS = 0.008
WIDTH = 0.038 + 2.0 * WALL_THICKNESS  # Preserve the 38 mm cavity width.
DEPTH = 0.036 + 2.0 * WALL_THICKNESS  # Preserve the 36 mm cavity depth.
HEIGHT = 0.115
BOTTOM_THICKNESS = 0.005
# Offset the original 3 mm-wall exterior outward, preserving the entire inner corner profile.
CORNER_CHAMFER = 0.007 + (2.0 - math.sqrt(2.0)) * (WALL_THICKNESS - 0.003)
EDGE_BEVEL = 0.001
CUP_COLOR = (0.42, 0.28, 0.035)  # Linear RGB: muted ochre yellow, not emissive.

# Holder XY dimensions are authored at task size; Z is doubled by CUP_HOLDER_SCALE.
HOLDER_CLEARANCE = 0.005  # Per face, larger than the two 2 mm contact offsets combined.
HOLDER_WALL_THICKNESS = 0.006
HOLDER_HEIGHT = 0.020
HOLDER_STRAIGHT_HEIGHT = 0.012
HOLDER_FLARE = 0.004


def _profile(inset: float) -> list[tuple[float, float]]:
    """Return a counterclockwise, inward-offset chamfered rectangle in task meters."""
    x = WIDTH / 2.0 - inset
    y = DEPTH / 2.0 - inset
    chamfer = CORNER_CHAMFER - (2.0 - math.sqrt(2.0)) * inset
    assert 0.0 < chamfer < min(x, y), "Inset must preserve the eight-sided profile"
    return [
        (x, y - chamfer),
        (x - chamfer, y),
        (-x + chamfer, y),
        (-x, y - chamfer),
        (-x, -y + chamfer),
        (-x + chamfer, -y),
        (x - chamfer, -y),
        (x, -y + chamfer),
    ]


def _point(xy: tuple[float, float], z: float) -> tuple[float, float, float]:
    """Convert a task-space point into the asset's unscaled coordinates."""
    return (xy[0] / CUP_SCALE_XY, xy[1] / CUP_SCALE_XY, z)


def _mesh(name: str, points: list, faces: list[list[int]], collision: bool) -> str:
    """Serialize one polygon mesh, optionally with convex collision schemas."""
    schemas = ""
    if collision:
        schemas = (
            ' (prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxCollisionAPI"])'
        )
    point_text = ", ".join("(" + ", ".join(f"{value:.10f}" for value in point) + ")" for point in points)
    counts = ", ".join(str(len(face)) for face in faces)
    indices = ", ".join(str(index) for face in faces for index in face)
    lines = [
        f'        def Mesh "{name}"{schemas}',
        "        {",
        '            uniform token subdivisionScheme = "none"',
        '            uniform token orientation = "rightHanded"',
        f"            point3f[] points = [{point_text}]",
        f"            int[] faceVertexCounts = [{counts}]",
        f"            int[] faceVertexIndices = [{indices}]",
    ]
    if collision:
        lines.extend(
            [
                "            bool physics:collisionEnabled = true",
                '            uniform token physics:approximation = "convexHull"',
                "            float physxCollision:contactOffset = 0.002",
                "            float physxCollision:restOffset = 0",
            ]
        )
    else:
        lines.append(f"            color3f[] primvars:displayColor = [{CUP_COLOR}]")
    return "\n".join([*lines, "        }", ""])


def build_usda() -> str:
    """Return a watertight visual shell and separate convex bottom/wall colliders."""
    bottom = -HEIGHT / 2.0
    top = HEIGHT / 2.0
    floor = bottom + BOTTOM_THICKNESS
    # Follow the shell from the underside, up the outside, across the lip, down the inside.
    # Offsetting both loops at the lip bevels its inner and outer edges without closing the opening.
    ring_specs = [
        (EDGE_BEVEL, bottom),
        (0.0, bottom + EDGE_BEVEL),
        (0.0, top - EDGE_BEVEL),
        (EDGE_BEVEL, top),
        (WALL_THICKNESS - EDGE_BEVEL, top),
        (WALL_THICKNESS, top - EDGE_BEVEL),
        (WALL_THICKNESS, floor),
    ]
    visual_points = []
    for inset, z in ring_specs:
        visual_points.extend(_point(xy, z) for xy in _profile(inset))
    visual_faces = [list(reversed(range(8)))]
    for ring in range(len(ring_specs) - 1):
        for index in range(8):
            following = (index + 1) % 8
            visual_faces.append(
                [8 * ring + index, 8 * ring + following, 8 * (ring + 1) + following, 8 * (ring + 1) + index]
            )
    visual_faces.append(list(range(48, 56)))

    # Keep each full-thickness wall as a separate convex prism, with exterior end points first.
    # Do not convex-hull the whole cup: that would close the cavity.
    prism_faces = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    wall_levels = [
        (floor, 0.0, WALL_THICKNESS),
        (top - EDGE_BEVEL, 0.0, WALL_THICKNESS),
        (top, EDGE_BEVEL, WALL_THICKNESS - EDGE_BEVEL),
    ]
    colliders = []
    for band in range(len(wall_levels) - 1):
        for index in range(8):
            following = (index + 1) % 8
            points = []
            for z, outer_inset, inner_inset in wall_levels[band : band + 2]:
                outer = _profile(outer_inset)
                inner = _profile(inner_inset)
                points.extend(
                    [
                        _point(outer[index], z),
                        _point(outer[following], z),
                        _point(inner[following], z),
                        _point(inner[index], z),
                    ]
                )
            colliders.append(_mesh(f"Wall_{index:02d}_{band}", points, prism_faces, collision=True))

    # Two convex plates reproduce the bottom bevel and the solid floor exactly.
    for band, (z_low, inset_low, z_high, inset_high) in enumerate(
        [(bottom, EDGE_BEVEL, bottom + EDGE_BEVEL, 0.0), (bottom + EDGE_BEVEL, 0.0, floor, 0.0)]
    ):
        points = [_point(xy, z_low) for xy in _profile(inset_low)]
        points.extend(_point(xy, z_high) for xy in _profile(inset_high))
        faces = [list(reversed(range(8))), list(range(8, 16))]
        for index in range(8):
            following = (index + 1) % 8
            faces.append([index, following, following + 8, index + 8])
        colliders.append(_mesh(f"Bottom_{band}", points, faces, collision=True))

    return "\n".join(
        [
            "#usda 1.0",
            "(",
            '    defaultPrim = "Asset"',
            "    metersPerUnit = 1",
            '    upAxis = "Z"',
            ")",
            "# Generated by assets/create_pouring_rectangular_cup.py; no external dependencies.",
            "# At CUP_SCALE=(0.7, 0.7, 1): 54 x 52 x 115 mm; 8 mm walls; 5 mm floor.",
            "# Original inner corner profile; 1 mm lip/base bevels; 38 x 36 mm cavity.",
            "# Matches pouring_rectangular_cup_holder.usd; keep each collider convex separately.",
            'def Xform "Asset" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"])',
            "{",
            "    bool physics:rigidBodyEnabled = true",
            "    bool physics:kinematicEnabled = false",
            "    float physics:mass = 0.10",
            "    float3 physics:diagonalInertia = (0, 0, 0)",
            "    bool physxRigidBody:enableSpeculativeCCD = true",
            "    int physxRigidBody:solverPositionIterationCount = 32",
            "    int physxRigidBody:solverVelocityIterationCount = 8",
            '    def Xform "Visuals" (prepend apiSchemas = ["MaterialBindingAPI"])',
            "    {",
            "        rel material:binding = </Asset/Looks/CupMaterial>",
            _mesh("Shell", visual_points, visual_faces, collision=False),
            "    }",
            '    def Scope "Looks"',
            "    {",
            '        def Material "CupMaterial"',
            "        {",
            "            token outputs:surface.connect = </Asset/Looks/CupMaterial/Surface.outputs:surface>",
            '            def Shader "Surface"',
            "            {",
            '                uniform token info:id = "UsdPreviewSurface"',
            f"                color3f inputs:diffuseColor = {CUP_COLOR}",
            "                float inputs:roughness = 0.85",
            "                float inputs:metallic = 0",
            "                float inputs:ior = 1.3",
            "                token outputs:surface",
            "            }",
            "        }",
            "    }",
            '    def Xform "Colliders"',
            "    {",
            '        token visibility = "invisible"',
            *colliders,
            "    }",
            "}",
            "",
        ]
    )


def build_holder_usda() -> str:
    """Return an open-bottom rectangular holder with separate convex walls and a flared lip."""
    # Negative insets offset the cup profile outward, preserving clearance along the chamfers too.
    levels = [(0.0, 0.0), (HOLDER_STRAIGHT_HEIGHT, 0.0), (HOLDER_HEIGHT, HOLDER_FLARE)]
    faces = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    walls = []
    for band in range(len(levels) - 1):
        for index in range(8):
            following = (index + 1) % 8
            points = []
            for z, flare in levels[band : band + 2]:
                inner = _profile(-HOLDER_CLEARANCE - flare)
                outer = _profile(-HOLDER_CLEARANCE - flare - HOLDER_WALL_THICKNESS)
                for xy in (outer[index], outer[following], inner[following], inner[index]):
                    points.append((*xy, z))
            walls.append(_mesh(f"Sector_{index:02d}_{band}", points, faces, collision=True))
    return "\n".join(
        [
            "#usda 1.0",
            "(",
            '    defaultPrim = "Asset"',
            "    metersPerUnit = 1",
            '    upAxis = "Z"',
            ")",
            "# Generated by assets/create_pouring_rectangular_cup.py; dimensions are in meters.",
            "# Matches pouring_rectangular_cup.usd at CUP_SCALE=(0.7, 0.7, 1).",
            "# At CUP_HOLDER_SCALE=(1, 1, 2): 40 mm tall, straight for the lower 24 mm.",
            "# Lower opening: 64 x 62 mm; upper opening: 72 x 70 mm; wall thickness: 6 mm.",
            "# 5 mm clearance normal to every cup face, including the chamfered corners.",
            "# Open bottom rests directly on the table; never convex-hull the whole ring.",
            'def Xform "Asset" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"])',
            "{",
            "    bool physics:rigidBodyEnabled = true",
            "    bool physics:kinematicEnabled = true",
            "    float physics:mass = 0.2",
            "    color3f[] primvars:displayColor = [(0.20, 0.30, 0.38)]",
            *walls,
            "}",
            "",
        ]
    )


def main() -> None:
    """Write the rectangular cup and matching holder USDs to the requested output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "usd" / "pouring_rectangular_cup.usd")
    parser.add_argument(
        "--holder-output", type=Path, default=Path(__file__).parent / "usd" / "pouring_rectangular_cup_holder.usd"
    )
    args = parser.parse_args()
    args.output.write_text(build_usda(), encoding="utf-8")
    args.holder_output.write_text(build_holder_usda(), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.holder_output}")


if __name__ == "__main__":
    main()
