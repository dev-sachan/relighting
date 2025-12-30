# ============================================================
# FACE 1 DATASET SCRIPT (OPTIMIZED VERSION)
# 4 angles × 10 HDRIs per expression
# Blender 4.x / 5.x
# ============================================================

import bpy, os, math, json, random
from mathutils import Vector

# ---------------- PATHS ----------------
FACE_FOLDER  = r"D:/dataset_photo/2"
HDRI_FOLDER  = r"D:/hdri_library"
OUTPUT_ROOT  = r"D:/renders_face2"

# ---------------- SETTINGS ----------------
ANGLES = [-45, -15, 15, 45]   # ONLY 4 angles
MAX_HDRIS = 10               # ONLY 10 HDRIs
RES = 512
SAMPLES = 32
CAMERA_FACTOR = 1.8
# ----------------------------------------

scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.render.resolution_x = RES
scene.render.resolution_y = RES
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = True

# SPEED OPTIMIZATIONS
scene.render.use_persistent_data = True  # Cache scene data
scene.cycles.device = 'GPU'  # Use GPU if available
scene.cycles.use_adaptive_sampling = True  # Faster sampling

# ---------------- WORLD ----------------
if scene.world is None:
    scene.world = bpy.data.worlds.new("World")

world = scene.world
world.use_nodes = True
nodes = world.node_tree.nodes
links = world.node_tree.links
nodes.clear()

tc = nodes.new("ShaderNodeTexCoord")
mp = nodes.new("ShaderNodeMapping")
et = nodes.new("ShaderNodeTexEnvironment")
bg = nodes.new("ShaderNodeBackground")
wo = nodes.new("ShaderNodeOutputWorld")

links.new(tc.outputs["Generated"], mp.inputs["Vector"])
links.new(mp.outputs["Vector"], et.inputs["Vector"])
links.new(et.outputs["Color"], bg.inputs["Color"])
links.new(bg.outputs["Background"], wo.inputs["Surface"])
bg.inputs["Strength"].default_value = 1.1

# load & limit HDRIs
HDRIS_ALL = [
    os.path.join(HDRI_FOLDER, f)
    for f in os.listdir(HDRI_FOLDER)
    if f.lower().endswith((".hdr", ".exr"))
]
HDRIS = random.sample(HDRIS_ALL, min(MAX_HDRIS, len(HDRIS_ALL)))

# Pre-load all HDRIs to avoid repeated disk access
HDRI_IMAGES = [bpy.data.images.load(hdri, check_existing=True) for hdri in HDRIS]

# ---------------- CAMERA ----------------
def ensure_camera():
    if scene.camera:
        return scene.camera
    bpy.ops.object.camera_add()
    scene.camera = bpy.context.active_object
    return scene.camera

cam = ensure_camera()

def look_at(obj, target=Vector((0,0,0))):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat('-Z','Y').to_euler()

# ---------------- SKIN MATERIAL ----------------
def apply_skin(objs):
    mat = bpy.data.materials.get("Skin_Test")
    if mat is None:
        mat = bpy.data.materials.new("Skin_Test")
        mat.use_nodes = True

        nodes = mat.node_tree.nodes
        bsdf = nodes.get("Principled BSDF")

        # Base skin color
        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = (0.68, 0.52, 0.45, 1)

        # Roughness
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.55

        # Specular (Blender version safe)
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.25
        elif "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = 0.25

        # Subsurface (Blender version safe)
        if "Subsurface Weight" in bsdf.inputs:
            bsdf.inputs["Subsurface Weight"].default_value = 0.18
            if "Subsurface Color" in bsdf.inputs:
                bsdf.inputs["Subsurface Color"].default_value = (0.85, 0.35, 0.30, 1)
        elif "Subsurface" in bsdf.inputs:
            bsdf.inputs["Subsurface"].default_value = 0.18
            if "Subsurface Color" in bsdf.inputs:
                bsdf.inputs["Subsurface Color"].default_value = (0.85, 0.35, 0.30, 1)

    for o in objs:
        o.data.materials.clear()
        o.data.materials.append(mat)


# ---------------- EXPRESSIONS ----------------
expressions = [f for f in os.listdir(FACE_FOLDER) if f.endswith(".obj")]

for expr in expressions:
    expr_name = os.path.splitext(expr)[0]
    expr_out = os.path.join(OUTPUT_ROOT, expr_name)
    os.makedirs(expr_out, exist_ok=True)

    print("Rendering:", expr_name)

    # clear old mesh
    bpy.ops.object.select_all(action='DESELECT')
    for o in bpy.data.objects:
        if o.type == 'MESH':
            o.select_set(True)
    bpy.ops.object.delete()

    # import face
    bpy.ops.wm.obj_import(filepath=os.path.join(FACE_FOLDER, expr))
    objs = [o for o in bpy.context.selected_objects if o.type == 'MESH']

    for o in objs:
        o.location = (0,0,0)
        o.rotation_euler = (0,0,0)
        o.scale = (1,1,1)

    # center mesh
    pts = [o.matrix_world @ v.co for o in objs for v in o.data.vertices]
    min_v = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    max_v = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    center = (min_v + max_v) / 2
    size = max(max_v - min_v)

    for o in objs:
        o.location -= center

    apply_skin(objs)

    cam_dist = size * CAMERA_FACTOR
    metadata = []
    img_idx = 0

    for idx, hdri_img in enumerate(HDRI_IMAGES):
        et.image = hdri_img  # Use pre-loaded image

        for ang in ANGLES:
            cam.location = Vector((
                cam_dist * math.sin(math.radians(ang)),
                0,
                cam_dist * math.cos(math.radians(ang))
            ))
            look_at(cam)

            fname = f"img_{img_idx:03d}.png"
            scene.render.filepath = os.path.join(expr_out, fname)
            bpy.ops.render.render(write_still=True)

            metadata.append({
                "file": fname,
                "angle_deg": ang,
                "hdri": os.path.basename(HDRIS[idx])
            })

            img_idx += 1

    with open(os.path.join(expr_out, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

print("✅ FACE 1 TEST DATASET COMPLETE") 