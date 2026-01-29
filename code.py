import bpy
import os
import math
import random
import json
from mathutils import Vector

# ======================================================
# PATHS
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
MAX_HDRI_PER_FACE = 40
TOTAL_IMAGES_PER_FACE = 120
LIGHT_RANDOM_OFFSET = 0.3

# ======================================================
# SCENE SETUP (CYCLES)
# ======================================================
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.render.resolution_x = RES
scene.render.resolution_y = RES
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = True
scene.cycles.device = 'GPU'

# ======================================================
# WORLD / HDRI SETUP
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

HDRIS = [
    bpy.data.images.load(os.path.join(HDRI_FOLDER, f), check_existing=True)
    for f in os.listdir(HDRI_FOLDER)
    if f.lower().endswith((".hdr", ".exr"))
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

    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")

    bsdf.inputs["Base Color"].default_value = (0.78, 0.62, 0.55, 1)
    bsdf.inputs["Subsurface Weight"].default_value = 0.18
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.6, 0.3)
    bsdf.inputs["Subsurface Scale"].default_value = 0.012
    bsdf.inputs["Roughness"].default_value = 0.48
    bsdf.inputs["Specular IOR Level"].default_value = 0.45
    bsdf.inputs["Sheen Weight"].default_value = 0.05

    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    obj.data.materials.append(mat)

def normalize_face():
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]

    for obj in meshes:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        add_skin_material(obj)

    verts = [obj.matrix_world @ v.co for obj in meshes for v in obj.data.vertices]
    min_v = Vector((min(v.x for v in verts), min(v.y for v in verts), min(v.z for v in verts)))
    max_v = Vector((max(v.x for v in verts), max(v.y for v in verts), max(v.z for v in verts)))

    center = (min_v + max_v) / 2
    size = max(max_v - min_v)
    scale = 2.0 / size

    for obj in meshes:
        obj.location = -center
        obj.scale = (scale, scale, scale)

    bpy.context.view_layer.update()
    return 1.0, Vector((0, 0, 0))

def create_camera_and_lights():
    bpy.ops.object.camera_add(location=(0, -5, 0))
    cam = bpy.context.active_object
    cam.data.lens = 50
    scene.camera = cam

    lights = []
    for loc, energy in [((2, -3, 3), 200), ((-2, -3, 2), 100), ((0, 3, 2), 120)]:
        bpy.ops.object.light_add(type="AREA", location=loc)
        l = bpy.context.active_object
        l.data.energy = energy
        l.data.size = 3
        lights.append(l)

    return cam, lights

# ======================================================
# MAIN LOOP (CRASH-RESUME SAFE)
# ======================================================
for face_id in os.listdir(FACE_ROOT_INPUT):
    face_dir = os.path.join(FACE_ROOT_INPUT, face_id)
    if not os.path.isdir(face_dir):
        continue

    for obj_file in os.listdir(face_dir):
        if not obj_file.lower().endswith(".obj"):
            continue

        print(f"\n▶ Processing {face_id}/{obj_file}")

        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)

        cam, lights = create_camera_and_lights()

        bpy.ops.wm.obj_import(
            filepath=os.path.join(face_dir, obj_file),
            forward_axis="NEGATIVE_Z",
            up_axis="Y"
        )

        radius, center = normalize_face()

        out_dir = os.path.join(OUTPUT_ROOT, f"face_{face_id}", obj_file[:-4])
        os.makedirs(out_dir, exist_ok=True)
        metadata_path = os.path.join(out_dir, "metadata.json")

        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = []

        if len(metadata) >= TOTAL_IMAGES_PER_FACE:
            print("  ✅ Already complete, skipping")
            continue

        random.seed(face_id)
        selected_hdris = random.sample(HDRIS, min(MAX_HDRI_PER_FACE, len(HDRIS)))

        img_index = len(metadata)

        for hdri_idx, hdri in enumerate(selected_hdris):
            if img_index >= TOTAL_IMAGES_PER_FACE:
                break

            env.image = hdri
            hdri_name = hdri.name  # ✅ CORRECT FIX

            for ang in ANGLES:
                if img_index >= TOTAL_IMAGES_PER_FACE:
                    break

                dist = (radius * 1.3) / math.tan(cam.data.angle / 2)
                cam.location = Vector((
                    math.sin(math.radians(ang)) * dist,
                    -math.cos(math.radians(ang)) * dist,
                    dist * 0.1
                ))
                look_at(cam, center)

                light_data = []
                for l in lights:
                    offset = Vector((
                        random.uniform(-LIGHT_RANDOM_OFFSET, LIGHT_RANDOM_OFFSET),
                        random.uniform(-LIGHT_RANDOM_OFFSET, LIGHT_RANDOM_OFFSET),
                        random.uniform(-LIGHT_RANDOM_OFFSET, LIGHT_RANDOM_OFFSET)
                    ))
                    l.location += offset
                    light_data.append(list(l.location))
                    l.location -= offset

                out_path = os.path.join(out_dir, f"img_{img_index:04d}.png")
                scene.render.filepath = out_path
                bpy.ops.render.render(write_still=True)

                metadata.append({
                    "filename": os.path.basename(out_path),
                    "angle": ang,
                    "hdri_idx": hdri_idx,
                    "hdri_name": hdri_name,
                    "cam_location": list(cam.location),
                    "cam_rotation": list(cam.rotation_euler),
                    "lights": light_data
                })

                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=4)

                img_index += 1

print("\n✅ ALL RENDERS COMPLETE")
bpy.ops.wm.quit_blender()
