"""
render_blender.py — Publication-quality SMPL mesh rendering in Blender.

Run from command line (NOT inside Blender GUI):
  blender --background --python render_blender.py -- \
      --input  mesh_export/ \
      --output renders/ \
      --color  blue \
      --mode   turntable \
      --fps    30

Produces:
  • Individual .png frames (transparent background option)
  • Composited .mp4 video

Requirements:
  • Blender ≥ 3.0 (tested on 3.6 / 4.0)
  • The .obj sequence from export_meshes.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import bpy
import bmesh
import sys
import os
import glob
import math
import argparse
from mathutils import Vector, Euler

# ──────────────────────────────────────────────
# 0. Parse CLI args (everything after "--")
# ──────────────────────────────────────────────
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Directory of frame_XXXXX.obj files")
parser.add_argument("--output", default="renders", help="Output directory")
parser.add_argument("--color", default="blue", choices=[
    "grey", "blue", "salmon", "lavender", "green", "white"
], help="Body color preset")
parser.add_argument("--mode", default="static", choices=[
    "static", "turntable", "front"
], help="Camera mode")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--resolution_x", type=int, default=1920)
parser.add_argument("--resolution_y", type=int, default=1080)
parser.add_argument("--samples", type=int, default=128,
                    help="Cycles render samples (128=good, 256=great, 64=fast preview)")
parser.add_argument("--transparent_bg", action="store_true",
                    help="Transparent background (for compositing)")
parser.add_argument("--floor", action="store_true", default=True,
                    help="Add checkered ground plane")
parser.add_argument("--no_floor", dest="floor", action="store_false")
parser.add_argument("--shadow_catcher", action="store_true", default=True,
                    help="Floor catches shadows but is otherwise invisible (needs transparent_bg)")
parser.add_argument("--video", action="store_true", default=True,
                    help="Also produce an mp4 video from the rendered frames")
parser.add_argument("--side_by_side", type=str, default=None,
                    help="Comma-separated paths to additional mesh dirs for side-by-side comparison")
parser.add_argument("--side_by_side_colors", type=str, default=None,
                    help="Comma-separated color names for side-by-side meshes")
parser.add_argument("--side_by_side_labels", type=str, default=None,
                    help="Comma-separated labels (not rendered, just for reference)")
parser.add_argument("--spacing", type=float, default=2.0,
                    help="X-spacing between side-by-side characters")
args = parser.parse_args(argv)


# ──────────────────────────────────────────────
# 1. Color palettes (publication-friendly)
# ──────────────────────────────────────────────
COLOR_MAP = {
    # (R, G, B) — these are chosen to match the muted tones
    # common in CVPR / ECCV motion-generation papers.
    "grey":     (0.55, 0.55, 0.55),
    "blue":     (0.40, 0.55, 0.78),
    "salmon":   (0.85, 0.55, 0.45),
    "lavender": (0.68, 0.55, 0.75),
    "green":    (0.45, 0.72, 0.55),
    "white":    (0.90, 0.90, 0.90),
}


# ──────────────────────────────────────────────
# 2. Scene setup
# ──────────────────────────────────────────────
def clean_scene():
    """Remove all default objects."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    # Remove default collections' orphans
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)


def setup_renderer(samples, res_x, res_y, transparent):
    """Configure Cycles renderer."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"

    # Use GPU if available
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"  # or "OPTIX" / "HIP" for AMD
    prefs.get_devices()
    for dev in prefs.devices:
        dev.use = True
    scene.cycles.device = "GPU"

    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = transparent

    # Color management — filmic for nice falloff
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium Contrast"


def create_body_material(name, color_rgb, roughness=0.45, subsurface=0.05):
    """
    Principled BSDF material that mimics the smooth matte look
    used in most motion-generation papers.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Principled BSDF
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*color_rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Subsurface Weight"].default_value = subsurface

    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    return mat


def add_checkered_floor(size=20.0, shadow_catcher=False):
    """Add a large plane with a subtle checkerboard texture."""
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "Floor"

    if shadow_catcher:
        floor.is_shadow_catcher = True
        return floor

    # Checkerboard material
    mat = bpy.data.materials.new("FloorMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Texture coordinate → scale → checker
    coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (8, 8, 8)
    checker = nodes.new("ShaderNodeTexChecker")
    checker.inputs["Color1"].default_value = (0.85, 0.85, 0.85, 1.0)
    checker.inputs["Color2"].default_value = (0.95, 0.95, 0.95, 1.0)
    checker.inputs["Scale"].default_value = 1.0

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.7
    bsdf.inputs["Specular IOR Level"].default_value = 0.1

    output = nodes.new("ShaderNodeOutputMaterial")

    links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], checker.inputs["Vector"])
    links.new(checker.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    floor.data.materials.append(mat)
    return floor


def setup_lighting():
    """Three-point lighting setup for soft, even illumination."""
    # Key light (warm, strong)
    bpy.ops.object.light_add(type="AREA", location=(4, -3, 6))
    key = bpy.context.active_object
    key.name = "KeyLight"
    key.data.energy = 300
    key.data.size = 3
    key.data.color = (1.0, 0.95, 0.9)
    key.rotation_euler = Euler((math.radians(50), 0, math.radians(30)))

    # Fill light (cool, softer)
    bpy.ops.object.light_add(type="AREA", location=(-4, -2, 4))
    fill = bpy.context.active_object
    fill.name = "FillLight"
    fill.data.energy = 120
    fill.data.size = 4
    fill.data.color = (0.9, 0.93, 1.0)
    fill.rotation_euler = Euler((math.radians(45), 0, math.radians(-40)))

    # Rim/back light
    bpy.ops.object.light_add(type="AREA", location=(0, 4, 5))
    rim = bpy.context.active_object
    rim.name = "RimLight"
    rim.data.energy = 150
    rim.data.size = 2
    rim.data.color = (1.0, 1.0, 1.0)
    rim.rotation_euler = Euler((math.radians(-30), 0, math.radians(180)))

    # Subtle environment lighting
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.75, 0.78, 0.82, 1.0)
        bg.inputs["Strength"].default_value = 0.3


def setup_camera(mode, num_frames, n_subjects=1, spacing=2.0):
    """
    Camera rig.
    - static:    fixed 3/4 elevated view
    - front:     straight-on front view
    - turntable: orbits around the subject(s)
    """
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.name = "Camera"
    bpy.context.scene.camera = cam

    # Frame width adjustment for multi-subject
    total_width = (n_subjects - 1) * spacing
    base_dist = 5.0
    cam_dist = base_dist + total_width * 0.5

    center_x = 0.0  # subjects are centered around 0

    if mode == "front":
        cam.location = (center_x, -cam_dist, 1.3)
        cam.rotation_euler = Euler((math.radians(85), 0, 0))
    elif mode == "turntable":
        # Animate orbit
        cam.location = (center_x, -cam_dist, 2.0)
        cam.rotation_euler = Euler((math.radians(75), 0, 0))

        # Add an empty at center to track
        bpy.ops.object.empty_add(location=(center_x, 0, 1.0))
        target = bpy.context.active_object
        target.name = "CamTarget"

        constraint = cam.constraints.new("TRACK_TO")
        constraint.target = target
        constraint.track_axis = "TRACK_NEGATIVE_Z"
        constraint.up_axis = "UP_Y"

        # Keyframe the orbit
        cam.location = (cam_dist * math.sin(0), -cam_dist * math.cos(0), 2.0)
        cam.keyframe_insert(data_path="location", frame=1)
        cam.location = (
            cam_dist * math.sin(2 * math.pi),
            -cam_dist * math.cos(2 * math.pi),
            2.0,
        )
        cam.keyframe_insert(data_path="location", frame=num_frames)

        # Make interpolation linear
        if cam.animation_data and cam.animation_data.action:
            for fc in cam.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"
    else:  # static — the default publication 3/4 view
        cam.location = (center_x + 1.5, -cam_dist, 2.2)
        cam.rotation_euler = Euler((math.radians(72), math.radians(2), math.radians(15)))

        # Track-to for stability
        bpy.ops.object.empty_add(location=(center_x, 0, 0.9))
        target = bpy.context.active_object
        target.name = "CamTarget"
        constraint = cam.constraints.new("TRACK_TO")
        constraint.target = target
        constraint.track_axis = "TRACK_NEGATIVE_Z"
        constraint.up_axis = "UP_Y"

    # Lens
    cam.data.lens = 85  # slight telephoto — less distortion on body


# ──────────────────────────────────────────────
# 3. Mesh sequence import + animation
# ──────────────────────────────────────────────
def load_mesh_sequence(mesh_dir, material, x_offset=0.0):
    """
    Import a sequence of .obj files and animate them using mesh_cache
    or manual per-frame shape key swapping.

    Strategy: Import frame 0 as the base object, then use a frame-change
    handler to swap vertex positions on each frame. This is much faster
    than importing hundreds of separate objects.
    """
    obj_files = sorted(glob.glob(os.path.join(mesh_dir, "frame_*.obj")))
    if not obj_files:
        raise FileNotFoundError(f"No frame_*.obj files in {mesh_dir}")

    n_frames = len(obj_files)
    print(f"Loading {n_frames} frames from {mesh_dir} …")

    # ── Step 1: Import first frame to get topology ──
    bpy.ops.wm.obj_import(filepath=obj_files[0])
    base_obj = bpy.context.selected_objects[0]
    base_obj.name = f"Body_{os.path.basename(mesh_dir)}"

    # Apply material
    if base_obj.data.materials:
        base_obj.data.materials[0] = material
    else:
        base_obj.data.materials.append(material)

    # Smooth shading
    bpy.context.view_layer.objects.active = base_obj
    bpy.ops.object.shade_smooth()

    # Apply offset
    if x_offset != 0:
        base_obj.location.x = x_offset

    # ── Step 2: Pre-load all vertex positions ──
    n_verts = len(base_obj.data.vertices)
    import numpy as np
    all_verts = np.zeros((n_frames, n_verts, 3), dtype=np.float32)

    for i, obj_path in enumerate(obj_files):
        # Fast: parse just the vertex lines
        verts = []
        with open(obj_path, "r") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.split()
                    verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
        all_verts[i] = np.array(verts, dtype=np.float32)

        if (i + 1) % 50 == 0:
            print(f"  Loaded {i + 1}/{n_frames}")

    # ── Step 3: Create shape keys for animation ──
    # Basis
    base_obj.shape_key_add(name="Basis", from_mix=False)

    # One shape key per frame
    for i in range(n_frames):
        sk = base_obj.shape_key_add(name=f"frame_{i:05d}", from_mix=False)
        for vi in range(n_verts):
            sk.data[vi].co = Vector(all_verts[i, vi].tolist())

        # Animate: this key is 1.0 at its frame, 0.0 everywhere else
        sk.value = 0.0
        sk.keyframe_insert(data_path="value", frame=max(1, i))  # off before
        sk.value = 1.0
        sk.keyframe_insert(data_path="value", frame=i + 1)      # on at frame
        sk.value = 0.0
        sk.keyframe_insert(data_path="value", frame=i + 2)      # off after

    print(f"  Created {n_frames} shape keys.")
    return base_obj, n_frames


# ──────────────────────────────────────────────
# 4. Main
# ──────────────────────────────────────────────
def main():
    os.makedirs(args.output, exist_ok=True)

    clean_scene()
    setup_renderer(args.samples, args.resolution_x, args.resolution_y, args.transparent_bg)
    setup_lighting()

    # ── Determine all mesh directories ──
    mesh_dirs = [args.input]
    color_names = [args.color]
    labels = ["main"]

    if args.side_by_side:
        extra_dirs = [d.strip() for d in args.side_by_side.split(",")]
        mesh_dirs.extend(extra_dirs)

        if args.side_by_side_colors:
            extra_colors = [c.strip() for c in args.side_by_side_colors.split(",")]
        else:
            defaults = ["grey", "salmon", "lavender", "green"]
            extra_colors = defaults[: len(extra_dirs)]
        color_names.extend(extra_colors)

        if args.side_by_side_labels:
            labels = [l.strip() for l in args.side_by_side_labels.split(",")]

    n_subjects = len(mesh_dirs)

    # Compute X offsets (centered around 0)
    if n_subjects == 1:
        offsets = [0.0]
    else:
        total = (n_subjects - 1) * args.spacing
        offsets = [i * args.spacing - total / 2 for i in range(n_subjects)]

    # ── Load meshes ──
    max_frames = 0
    for mesh_dir, cname, xoff in zip(mesh_dirs, color_names, offsets):
        rgb = COLOR_MAP.get(cname, COLOR_MAP["blue"])
        mat = create_body_material(f"BodyMat_{cname}", rgb)
        _, nf = load_mesh_sequence(mesh_dir, mat, x_offset=xoff)
        max_frames = max(max_frames, nf)

    # ── Floor ──
    if args.floor:
        use_shadow_catcher = args.transparent_bg and args.shadow_catcher
        add_checkered_floor(size=20.0, shadow_catcher=use_shadow_catcher)

    # ── Camera ──
    setup_camera(args.mode, max_frames, n_subjects, args.spacing)

    # ── Timeline ──
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = max_frames
    scene.render.fps = args.fps

    # ── Render ──
    scene.render.filepath = os.path.join(args.output, "frame_")
    print(f"\nRendering {max_frames} frames at {args.resolution_x}×{args.resolution_y} …")
    bpy.ops.render.render(animation=True)

    # ── Composite to video ──
    if args.video:
        import subprocess
        frame_pattern = os.path.join(args.output, "frame_%04d.png")
        video_path = os.path.join(args.output, "output.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(args.fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            video_path,
        ]
        print(f"Compositing video: {video_path}")
        subprocess.run(cmd, check=True)

    print("✓ Rendering complete.")


if __name__ == "__main__":
    main()