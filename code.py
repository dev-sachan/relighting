import bpy
import os
import math
import random
import json
from mathutils import Vector, Euler

# ======================================================
# PATHS (EDIT IF NEEDED)
# ======================================================
FACE_ROOT_INPUT = r"D:\UNDER100"
HDRI_FOLDER = r"C:\Users\Vaibhav singh\Desktop\HDRI"
OUTPUT_ROOT = r"D:\under100_renders"

# ======================================================
# SETTINGS
# ======================================================
ANGLES = [-25, 0, 25]
RES = 512
SAMPLES = 128
LIGHT_RANDOM_OFFSET = 0.3  # maximum offset for random light position
HDRIS_PER_OBJ = 40  # Number of HDRIs to use per OBJ file

# ======================================================
# SCENE SETUP
# ======================================================
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.render.resolution_x = RES
scene.render.resolution_y = RES
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = True
scene.cycles.device = 'GPU'

# ======================================================
# WORLD / HDRI
# ======================================================
world = scene.world or bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True

nodes = world.node_tree.nodes
links = world.node_tree.links
nodes.clear()

env = nodes.new("ShaderNodeTexEnvironment")
bg = nodes.new("ShaderNodeBackground")
out = nodes.new("ShaderNodeOutputWorld")

links.new(env.outputs["Color"], bg.inputs["Color"])
links.new(bg.outputs["Background"], out.inputs["Surface"])

bg.inputs["Strength"].default_value = 0.8

# Load all HDRIs with their filenames
HDRI_FILES = [
    f for f in os.listdir(HDRI_FOLDER)
    if f.lower().endswith((".hdr", ".exr"))
]

HDRIS = [
    (filename, bpy.data.images.load(os.path.join(HDRI_FOLDER, filename), check_existing=True))
    for filename in HDRI_FILES
]

# ======================================================
# HELPERS
# ======================================================
def look_at(cam, target):
    direction = target - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

def add_skin_material(obj):
    obj.data.materials.clear()
    mat = bpy.data.materials.new(name="SkinMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (0.8, 0.6, 0.5, 1.0)
    bsdf.inputs['Subsurface Weight'].default_value = 0.1
    bsdf.inputs['Subsurface Radius'].default_value = (1.0, 0.5, 0.25)
    bsdf.inputs['Subsurface Scale'].default_value = 0.01
    bsdf.inputs['Roughness'].default_value = 0.4
    bsdf.inputs['Specular IOR Level'].default_value = 0.5
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    obj.data.materials.append(mat)
    print(f"    Added skin material to {obj.name}")

def normalize_and_orient_face():
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        print("  ❌ No meshes found!")
        return 1.0, Vector((0, 0, 0))
    for obj in meshes:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        obj.select_set(False)
        add_skin_material(obj)
    bpy.context.view_layer.update()
    all_verts = [obj.matrix_world @ v.co for obj in meshes for v in obj.data.vertices]
    if not all_verts:
        return 1.0, Vector((0, 0, 0))
    min_co = Vector((min(v.x for v in all_verts), min(v.y for v in all_verts), min(v.z for v in all_verts)))
    max_co = Vector((max(v.x for v in all_verts), max(v.y for v in all_verts), max(v.z for v in all_verts)))
    center = (min_co + max_co) / 2
    size = max_co - min_co
    max_dim = max(size)
    if max_dim < 0.001:
        return 1.0, Vector((0, 0, 0))
    scale_factor = 2.0 / max_dim
    for obj in meshes:
        obj.location = -center
        obj.scale = Vector((scale_factor, scale_factor, scale_factor))
    bpy.context.view_layer.update()
    all_verts = [obj.matrix_world @ v.co for obj in meshes for v in obj.data.vertices]
    min_co = Vector((min(v.x for v in all_verts), min(v.y for v in all_verts), min(v.z for v in all_verts)))
    max_co = Vector((max(v.x for v in all_verts), max(v.y for v in all_verts), max(v.z for v in all_verts)))
    final_center = (min_co + max_co) / 2
    final_size = max_co - min_co
    return max(final_size) / 2, final_center

def create_camera_and_lights():
    bpy.ops.object.camera_add(location=(0, -5, 0))
    cam = bpy.context.active_object
    cam.data.lens = 50
    cam.data.clip_start = 0.01
    cam.data.clip_end = 100
    scene.camera = cam

    lights = []

    # Key light
    bpy.ops.object.light_add(type='AREA', location=(2, -3, 3))
    key_light = bpy.context.active_object
    key_light.data.energy = 200
    key_light.data.size = 3
    lights.append(key_light)

    # Fill light
    bpy.ops.object.light_add(type='AREA', location=(-2, -3, 2))
    fill_light = bpy.context.active_object
    fill_light.data.energy = 100
    fill_light.data.size = 3
    lights.append(fill_light)

    # Rim light
    bpy.ops.object.light_add(type='AREA', location=(0, 3, 2))
    rim_light = bpy.context.active_object
    rim_light.data.energy = 120
    rim_light.data.size = 3
    lights.append(rim_light)

    return cam, lights

# ======================================================
# MAIN LOOP
# ======================================================
render_count = 0

for face_id in os.listdir(FACE_ROOT_INPUT):
    face_dir = os.path.join(FACE_ROOT_INPUT, face_id)
    if not os.path.isdir(face_dir):
        continue

    for obj_file in os.listdir(face_dir):
        if not obj_file.lower().endswith(".obj"):
            continue

        print(f"\n{'='*60}")
        print(f"▶ Processing {face_id}/{obj_file}")
        print(f"{'='*60}")

        # Clear scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

        # Create camera and lights
        cam, lights = create_camera_and_lights()

        # Import OBJ
        obj_path = os.path.join(face_dir, obj_file)
        print(f"  Importing: {obj_path}")
        try:
            bpy.ops.wm.obj_import(
                filepath=obj_path,
                forward_axis='NEGATIVE_Z',
                up_axis='Y'
            )
        except Exception as e:
            print(f"  ❌ Import failed: {e}")
            continue

        meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
        if not meshes:
            print("  ❌ No mesh found!")
            continue

        # Normalize and add materials
        radius, face_center = normalize_and_orient_face()
        print(f"  Radius: {radius:.2f}")

        # Prepare output folder
        out_dir = os.path.join(OUTPUT_ROOT, f"face_{face_id}", obj_file[:-4])
        os.makedirs(out_dir, exist_ok=True)
        metadata_path = os.path.join(out_dir, "metadata.json")

        # If JSON exists already (from previous runs), load it
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata_list = json.load(f)
        else:
            metadata_list = []

        # Randomly select HDRIS_PER_OBJ HDRIs for this OBJ file
        selected_hdris = random.sample(HDRIS, min(HDRIS_PER_OBJ, len(HDRIS)))
        print(f"  Selected {len(selected_hdris)} HDRIs randomly")

        # Render with selected HDRIs and angles
        for hdri_idx, (hdri_name, hdri_image) in enumerate(selected_hdris):
            env.image = hdri_image
            for ang in ANGLES:
                fov = cam.data.angle
                dist = (radius * 1.3) / math.tan(fov / 2)
                cam.location = face_center + Vector((
                    math.sin(math.radians(ang)) * dist,
                    -math.cos(math.radians(ang)) * dist,
                    dist * 0.1
                ))
                look_at(cam, face_center)

                # Randomize lights slightly for this render and store positions
                light_positions = []
                for light in lights:
                    offset = Vector((
                        random.uniform(-LIGHT_RANDOM_OFFSET, LIGHT_RANDOM_OFFSET),
                        random.uniform(-LIGHT_RANDOM_OFFSET, LIGHT_RANDOM_OFFSET),
                        random.uniform(-LIGHT_RANDOM_OFFSET, LIGHT_RANDOM_OFFSET)
                    ))
                    light.location += offset
                    light_positions.append(list(light.location))
                    light.location -= offset  # reset for next render

                idx = len([f for f in os.listdir(out_dir) if f.endswith('.png')])
                out_path = os.path.join(out_dir, f"img_{idx:04d}.png")
                scene.render.filepath = out_path
                bpy.ops.render.render(write_still=True)
                render_count += 1

                # Save metadata immediately with HDRI name
                metadata_entry = {
                    "filename": os.path.basename(out_path),
                    "angle": ang,
                    "hdri_idx": hdri_idx,
                    "hdri_name": hdri_name,
                    "cam_location": list(cam.location),
                    "cam_rotation": list(cam.rotation_euler),
                    "lights": light_positions
                }
                metadata_list.append(metadata_entry)

                # Write metadata JSON immediately
                with open(metadata_path, "w") as f:
                    json.dump(metadata_list, f, indent=4)

print(f"\n{'='*60}")
print(f"✅ COMPLETE: {render_count} renders with metadata")
print(f"{'='*60}")

bpy.ops.wm.quit_blender()
