# ============================================================
# FACE DATASET — OPTIMIZED AI PIPELINE
# Three-Point Lighting + Enhanced Skin + Better Framing
# Blender 3.x / 4.x compatible
# ============================================================

import bpy, os, math, random, json, time
from mathutils import Vector, Euler

# ---------------- PATHS ----------------
FACE_FOLDER = r"D:/dataset_photo/2"
HDRI_FOLDER = r"C:\Users\Vaibhav singh\Desktop\HDRI"
OUTPUT_ROOT = r"D:/renders_face2_AI_final"

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

# ---------------- REALISTIC SKIN SHADER ----------------
def apply_skin(objs):
    mat = bpy.data.materials.get("SkinAI")
    if not mat:
        mat = bpy.data.materials.new("SkinAI")
        mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    out = nodes.new("ShaderNodeOutputMaterial")

    # ==== REALISTIC SKIN COLOR with subtle variation ====
    coord = nodes.new("ShaderNodeTexCoord")
    
    # Color variation noise
    color_noise = nodes.new("ShaderNodeTexNoise")
    color_noise.inputs["Scale"].default_value = 25
    color_noise.inputs["Detail"].default_value = 3
    color_noise.inputs["Roughness"].default_value = 0.6
    
    # Color ramp for skin tone variation
    color_ramp = nodes.new("ShaderNodeValToRGB")
    color_ramp.color_ramp.elements[0].position = 0.45
    color_ramp.color_ramp.elements[0].color = (0.82, 0.58, 0.48, 1)  # Darker/redder
    color_ramp.color_ramp.elements[1].position = 0.55
    color_ramp.color_ramp.elements[1].color = (0.88, 0.68, 0.58, 1)  # Lighter/yellower
    
    links.new(coord.outputs["Object"], color_noise.inputs["Vector"])
    links.new(color_noise.outputs["Fac"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], bsdf.inputs["Base Color"])

    # ==== REALISTIC ROUGHNESS variation ====
    rough_noise = nodes.new("ShaderNodeTexNoise")
    rough_noise.inputs["Scale"].default_value = 100
    rough_noise.inputs["Detail"].default_value = 4
    
    rough_ramp = nodes.new("ShaderNodeValToRGB")
    rough_ramp.color_ramp.elements[0].position = 0.4
    rough_ramp.color_ramp.elements[0].color = (0.3, 0.3, 0.3, 1)  # Smoother areas
    rough_ramp.color_ramp.elements[1].position = 0.6
    rough_ramp.color_ramp.elements[1].color = (0.5, 0.5, 0.5, 1)  # Rougher areas
    
    links.new(coord.outputs["Object"], rough_noise.inputs["Vector"])
    links.new(rough_noise.outputs["Fac"], rough_ramp.inputs["Fac"])
    links.new(rough_ramp.outputs["Color"], bsdf.inputs["Roughness"])

    # ==== SPECULAR (skin shininess) ====
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.45
    else:
        bsdf.inputs["Specular"].default_value = 0.45

    # ==== ENHANCED SUBSURFACE SCATTERING ====
    if "Subsurface Weight" in bsdf.inputs:
        bsdf.inputs["Subsurface Weight"].default_value = 0.35  # Increased for more realism
        bsdf.inputs["Subsurface Scale"].default_value = 0.05  # Proper skin scale
        if "Subsurface Radius" in bsdf.inputs:
            bsdf.inputs["Subsurface Radius"].default_value = (1.2, 0.5, 0.3)  # R, G, B scattering
        if "Subsurface IOR" in bsdf.inputs:
            bsdf.inputs["Subsurface IOR"].default_value = 1.4  # Skin IOR
    else:
        bsdf.inputs["Subsurface"].default_value = 0.35
        if "Subsurface Radius" in bsdf.inputs:
            bsdf.inputs["Subsurface Radius"].default_value = (1.2, 0.5, 0.3)
        if "Subsurface Color" in bsdf.inputs:
            bsdf.inputs["Subsurface Color"].default_value = (0.9, 0.5, 0.4, 1)

    # ==== REALISTIC SKIN TEXTURE (pores + wrinkles) ====
    # Large features (wrinkles, subtle contours)
    noise_large = nodes.new("ShaderNodeTexNoise")
    noise_large.inputs["Scale"].default_value = 80
    noise_large.inputs["Detail"].default_value = 3
    noise_large.inputs["Roughness"].default_value = 0.5
    links.new(coord.outputs["Object"], noise_large.inputs["Vector"])
    
    # Medium features (skin texture)
    noise_medium = nodes.new("ShaderNodeTexNoise")
    noise_medium.inputs["Scale"].default_value = 250
    noise_medium.inputs["Detail"].default_value = 5
    noise_medium.inputs["Roughness"].default_value = 0.6
    links.new(coord.outputs["Object"], noise_medium.inputs["Vector"])
    
    # Fine features (pores)
    noise_fine = nodes.new("ShaderNodeTexNoise")
    noise_fine.inputs["Scale"].default_value = 1200
    noise_fine.inputs["Detail"].default_value = 2
    links.new(coord.outputs["Object"], noise_fine.inputs["Vector"])
    
    # Mix large + medium
    mix1 = nodes.new("ShaderNodeMix")
    mix1.data_type = 'FLOAT'
    mix1.inputs[0].default_value = 0.6
    links.new(noise_large.outputs["Fac"], mix1.inputs[2])
    links.new(noise_medium.outputs["Fac"], mix1.inputs[3])
    
    # Add fine details
    mix2 = nodes.new("ShaderNodeMix")
    mix2.data_type = 'FLOAT'
    mix2.inputs[0].default_value = 0.3
    links.new(mix1.outputs[1], mix2.inputs[2])
    links.new(noise_fine.outputs["Fac"], mix2.inputs[3])
    
    # Apply bump
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.2
    bump.inputs["Distance"].default_value = 0.1
    links.new(mix2.outputs[1], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    for o in objs:
        o.data.materials.clear()
        o.data.materials.append(mat)

    return mat

# ---- IMPROVEMENT 1: METADATA LIST ----
metadata = []

# ---------------- RENDER ----------------
for file in os.listdir(FACE_FOLDER):
    if not file.endswith(".obj"):
        continue

    bpy.ops.object.select_all(action='DESELECT')
    for o in bpy.data.objects:
        if o.type == 'MESH':
            o.select_set(True)
    bpy.ops.object.delete()

    bpy.ops.wm.obj_import(filepath=os.path.join(FACE_FOLDER, file))
    objs = [o for o in bpy.context.selected_objects if o.type == "MESH"]

    skin_mat = apply_skin(objs)

    pts = [o.matrix_world @ v.co for o in objs for v in o.data.vertices]
    size = max(pts) - min(pts)
    cam_dist = size.length * CAMERA_FACTOR
    
    # ---- IMPROVEMENT 4: Calculate face center for adaptive lighting ----
    face_center = sum(pts, Vector()) / len(pts)
    
    # Position lights relative to face
    key.location = face_center + Vector((2, -3, 2))
    key.rotation_euler = Euler((math.radians(60), 0, math.radians(30)), 'XYZ')
    
    fill.location = face_center + Vector((-2, -2, 1))
    fill.rotation_euler = Euler((math.radians(45), 0, math.radians(-30)), 'XYZ')
    
    rim.location = face_center + Vector((0, 2, 1.5))
    rim.rotation_euler = Euler((math.radians(120), 0, 0), 'XYZ')
    # ---- END IMPROVEMENT 4 ----

    idx = 0
    for h, hdri in enumerate(HDRI_IMAGES):
        et.image = hdri
        mp.inputs["Rotation"].default_value[2] = HDRI_ROT[h]

        for ang in ANGLES:
            ang_rad = math.radians(ang)
            cam.location = Vector((cam_dist * math.sin(ang_rad),
                                   -cam_dist * math.cos(ang_rad),
                                   size.length * 0.15))
            cam.rotation_euler = (math.radians(90), 0, math.radians(ang))

            # ---- IMPROVEMENT 5: Better DOF focus ----
            cam.data.dof.focus_distance = (cam.location - face_center).length
            # ---- END IMPROVEMENT 5 ----

            # ---- IMPROVEMENT 1: Log metadata ----
            meta_entry = {
                "image": f"img_{idx:04d}.png",
                "model": file,
                "hdri": os.path.basename(HDRIS[h]),
                "hdri_rotation": float(HDRI_ROT[h]),
                "angle": ang,
                "camera_distance": float(cam_dist),
                "camera_location": [float(cam.location.x), float(cam.location.y), float(cam.location.z)],
                "timestamp": time.time()
            }
            metadata.append(meta_entry)
            # ---- END IMPROVEMENT 1 ----

            scene.render.filepath = os.path.join(OUTPUT_ROOT, f"img_{idx:04d}.png")
            bpy.ops.render.render(write_still=True)
            idx += 1

# ---- IMPROVEMENT 1: Save metadata ----
with open(os.path.join(OUTPUT_ROOT, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Script complete! Rendered {len(metadata)} images with metadata")
