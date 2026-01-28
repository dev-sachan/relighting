import bpy, os, math, random, json
from mathutils import Vector

# ---------------- ROOT PATHS ----------------
FACE_ROOT_INPUT = r"D:/dataset_photo"          # contains MANY folders
HDRI_FOLDER = r"C:\Users\Vaibhav singh\Desktop\HDRI"
OUTPUT_ROOT = r"D:/renders_face2_AI_final"

# ---------------- BATCH CONTROL (NEW) ----------------
BATCH_SIZE = 20
BATCH_INDEX = 0   # 0 = first 20, 1 = next 20, 2 = next 20 ...

# ---------------- SETTINGS ----------------
ANGLES = [-30, 0, 30]
MAX_HDRIS = 40
RES = 512
SAMPLES = 128
CAMERA_FACTOR = 4.5

# ---------------- SCENE ----------------
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.render.resolution_x = RES
scene.render.resolution_y = RES
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = True
scene.cycles.device = 'GPU'
scene.cycles.use_adaptive_sampling = True
scene.render.film_transparent = False
scene.view_settings.exposure = -0.1
scene.view_settings.gamma = 1.1
scene.view_settings.look = 'AgX - Medium High Contrast'

# ---------------- WORLD ----------------
world = scene.world or bpy.data.worlds.new("World")
scene.world = world
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
bg.inputs["Strength"].default_value = 0.7

# ---------------- HDRIs ----------------
HDRIS = [os.path.join(HDRI_FOLDER, f) for f in os.listdir(HDRI_FOLDER)
         if f.lower().endswith((".hdr", ".exr"))]
HDRIS = random.sample(HDRIS, min(MAX_HDRIS, len(HDRIS)))
HDRI_IMAGES = [bpy.data.images.load(h, check_existing=True) for h in HDRIS]
HDRI_ROT = [random.uniform(-math.pi, math.pi) for _ in HDRI_IMAGES]

MAX_IMAGES_PER_EXPRESSION = len(HDRI_IMAGES) * len(ANGLES)

# ---------------- CAMERA ----------------
if not scene.camera:
    bpy.ops.object.camera_add()
    scene.camera = bpy.context.active_object
cam = scene.camera
cam.data.lens = 100
cam.data.dof.use_dof = True
cam.data.dof.aperture_fstop = 2.2
cam.data.dof.aperture_blades = 6

# ---------------- LIGHTS ----------------
def ensure_light(name, energy, size):
    if name not in bpy.data.objects:
        bpy.ops.object.light_add(type='AREA')
        l = bpy.context.active_object
        l.name = name
        l.data.energy = energy
        l.data.size = size
    return bpy.data.objects[name]

key = ensure_light("KeyLight", 40, 4)
fill = ensure_light("FillLight", 15, 3)
rim = ensure_light("RimLight", 25, 2)

# ---------------- SKIN ----------------
def apply_skin(objs):
    mat = bpy.data.materials.get("SkinAI") or bpy.data.materials.new("SkinAI")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    out = nodes.new("ShaderNodeOutputMaterial")
    coord = nodes.new("ShaderNodeTexCoord")

    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 200
    links.new(coord.outputs["Object"], noise.inputs["Vector"])

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.85, 0.6, 0.5, 1)
    ramp.color_ramp.elements[1].color = (0.95, 0.75, 0.65, 1)
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    if "Subsurface Weight" in bsdf.inputs:
        bsdf.inputs["Subsurface Weight"].default_value = 0.35
        bsdf.inputs["Subsurface Scale"].default_value = 0.05

    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    for o in objs:
        o.data.materials.clear()
        o.data.materials.append(mat)

# ================== MAIN LOOP ==================

all_faces = sorted([
    f for f in os.listdir(FACE_ROOT_INPUT)
    if os.path.isdir(os.path.join(FACE_ROOT_INPUT, f))
])

start = BATCH_INDEX * BATCH_SIZE
end = start + BATCH_SIZE
faces_to_process = all_faces[start:end]

print(f"\n🚀 Processing batch {BATCH_INDEX} → faces {start} to {end-1}")

for face_id in faces_to_process:
    FACE_FOLDER = os.path.join(FACE_ROOT_INPUT, face_id)
    if not os.path.isdir(FACE_FOLDER):
        continue

    print(f"\n🔥 Processing face folder: {face_id}")

    FACE_OUTPUT_ROOT = os.path.join(OUTPUT_ROOT, f"face_{face_id}")
    os.makedirs(FACE_OUTPUT_ROOT, exist_ok=True)

    for file in os.listdir(FACE_FOLDER):
        if not file.endswith(".obj"):
            continue

        expression_name = os.path.splitext(file)[0]
        expression_folder = os.path.join(FACE_OUTPUT_ROOT, expression_name)
        os.makedirs(expression_folder, exist_ok=True)

        metadata_path = os.path.join(expression_folder, "metadata.json")
        metadata = json.load(open(metadata_path)) if os.path.exists(metadata_path) else []

        existing = [f for f in os.listdir(expression_folder) if f.endswith(".png")]
        if len(existing) >= MAX_IMAGES_PER_EXPRESSION:
            print(f"✔ {expression_name} complete")
            continue

        ids = []
        for f in existing:
            try:
                ids.append(int(f.split("_")[1].split(".")[0]))
            except:
                pass
        idx = max(ids) + 1 if ids else 0

        bpy.ops.object.select_all(action='DESELECT')
        for o in bpy.data.objects:
            if o.type == 'MESH':
                o.select_set(True)
        bpy.ops.object.delete()

        bpy.ops.wm.obj_import(filepath=os.path.join(FACE_FOLDER, file))
        objs = [o for o in bpy.context.selected_objects if o.type == "MESH"]
        apply_skin(objs)

        pts = [o.matrix_world @ v.co for o in objs for v in o.data.vertices]
        size = max(pts) - min(pts)
        cam_dist = size.length * CAMERA_FACTOR
        face_center = sum(pts, Vector()) / len(pts)

        key.location = face_center + Vector((2, -3, 2))
        fill.location = face_center + Vector((-2, -2, 1))
        rim.location = face_center + Vector((0, 2, 1.5))

        for h, hdri in enumerate(HDRI_IMAGES):
            et.image = hdri
            mp.inputs["Rotation"].default_value[2] = HDRI_ROT[h]

            for ang in ANGLES:
                if idx >= MAX_IMAGES_PER_EXPRESSION:
                    break

                img_name = f"img_{idx:04d}.png"
                img_path = os.path.join(expression_folder, img_name)

                if os.path.exists(img_path):
                    idx += 1
                    continue

                ang_rad = math.radians(ang)
                cam.location = Vector((
                    cam_dist * math.sin(ang_rad),
                    -cam_dist * math.cos(ang_rad),
                    size.length * 0.15
                ))
                cam.rotation_euler = (math.radians(90), 0, math.radians(ang))
                cam.data.dof.focus_distance = (cam.location - face_center).length

                scene.render.filepath = img_path
                bpy.ops.render.render(write_still=True)

                metadata.append({
                    "image": img_name,
                    "model": file,
                    "expression": expression_name,
                    "hdri": os.path.basename(HDRIS[h]),
                    "angle": ang,
                    "camera_distance": float(cam_dist)
                })

                json.dump(metadata, open(metadata_path, "w"), indent=2)
                idx += 1

print("\n✅ BATCH COMPLETE — crash-safe & capped")
