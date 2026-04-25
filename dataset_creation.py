import bpy
import os
import math
import json
import tempfile
import time                         # [TIME ESTIMATION] added
from mathutils import Vector

# ======================================================
# PATHS (EDIT THESE)
# ======================================================
FACE_ROOT_INPUT = r"C:\Users\Vaibhav singh\Documents\model.zip"
HDRI_FOLDER     = r"C:\Users\Vaibhav singh\Desktop\HDRI"
OUTPUT_ROOT     = r"D:\alexander\synthlight_renders_v6"

# ======================================================
# RENDER SETTINGS
# ======================================================
RES              = 512
SAMPLES          = 256
CAMERA_LENS_MM   = 50
CAMERA_DIST_MULT = 1.15
CAMERA_ELEVATION = 0.08
NUM_HDRI_ROTATIONS = 45

CAMERA_POSITIONS = {
    'front':     0,
}

# HDRI tier thresholds
TIER_SUNNY  = 50
TIER_DECENT = 12
TIER_SOFT   = 3

# Target face luminance range (slightly reduced max)
TARGET_LUM_MIN = 0.5
TARGET_LUM_MAX = 0.7

# ======================================================
# SCENE SETUP
# ======================================================
scene = bpy.context.scene
scene.render.engine        = 'CYCLES'
scene.render.resolution_x  = RES
scene.render.resolution_y  = RES
scene.cycles.samples       = SAMPLES
scene.cycles.use_denoising = True
scene.cycles.denoiser      = 'OPTIX'
scene.cycles.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
scene.cycles.device        = 'GPU'
scene.cycles.use_adaptive_sampling = True
scene.cycles.adaptive_threshold = 0.01
scene.cycles.adaptive_min_samples = 32
scene.cycles.max_bounces             =8
scene.cycles.diffuse_bounces         = 3
scene.cycles.glossy_bounces          = 3
scene.cycles.transmission_bounces    = 4
scene.cycles.transparent_max_bounces = 4
scene.cycles.volume_bounces          = 0
scene.cycles.caustics_reflective     = False
scene.cycles.caustics_refractive     = False
scene.cycles.blur_glossy             = 0.5
scene.cycles.sample_clamp_indirect   = 1.0

scene.render.film_transparent = False

try:
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look           = 'Default'
    print("Using Filmic (Default)")
except:
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look           = 'None'
    print("Filmic not available, using Standard")

scene.view_settings.exposure = 0.0
scene.view_settings.gamma    = 1.0
scene.render.use_persistent_data = True
scene.cycles.tile_size = 256

prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'OPTIX'
prefs.get_devices()
for d in prefs.devices:
    d.use = True

# ======================================================
# WORLD / HDRI NODES
# ======================================================
if not scene.world:
    scene.world = bpy.data.worlds.new("World")
world = scene.world
world.use_nodes = True
wt_nodes = world.node_tree.nodes
wt_links = world.node_tree.links
wt_nodes.clear()

coord   = wt_nodes.new("ShaderNodeTexCoord")
mapping = wt_nodes.new("ShaderNodeMapping")
env     = wt_nodes.new("ShaderNodeTexEnvironment")
bg      = wt_nodes.new("ShaderNodeBackground")
out     = wt_nodes.new("ShaderNodeOutputWorld")

hue_sat = wt_nodes.new("ShaderNodeHueSaturation")
hue_sat.name = "DesaturateBrightHDRI"
hue_sat.inputs["Saturation"].default_value = 1.0
hue_sat.location = (env.location.x + 200, env.location.y)

wt_links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
wt_links.new(mapping.outputs["Vector"],  env.inputs["Vector"])
wt_links.new(env.outputs["Color"],       bg.inputs["Color"])
wt_links.new(bg.outputs["Background"],   out.inputs["Surface"])

# ======================================================
# HELPER: Recalculate strength & saturation (fine-tuned)
# ======================================================
def recalc_strength_and_saturation(entry):
    avg_lum = entry.get('avg_lum', 0.5)
    peak_lum = entry.get('peak_lum', 1.0)
    tier = entry['tier']

    # Base strength (slightly reduced for sunny)
    if tier == 1:  # SUNNY
        if avg_lum > 0.8:
            strength = 0.45
        elif avg_lum > 0.5:
            strength = 0.55
        else:
            strength = 0.65
    elif tier == 2:  # DECENT
        if avg_lum > 0.4:
            strength = 0.8
        elif avg_lum > 0.25:
            strength = 1.0
        else:
            strength = 1.2
    elif tier == 3:  # SOFT
        if avg_lum > 0.2:
            strength = 1.8
        elif avg_lum > 0.1:
            strength = 2.2
        else:
            strength = 2.5
    else:  # FLAT
        if avg_lum > 0.15:
            strength = 3.0
        elif avg_lum > 0.08:
            strength = 3.5
        else:
            strength = 4.0

    # Reduce strength for extremely high peak, keep floor
    if tier == 1:
        floor = 0.4
    else:
        floor = 0.3
    if peak_lum > 50:
        strength *= 0.6
    elif peak_lum > 20:
        strength *= 0.8
    elif peak_lum > 10:
        strength *= 0.9
    strength = max(strength, floor)

    # Saturation reduction only for very bright HDRIs
    saturation = 1.0
    if peak_lum > 50:
        saturation = 0.8

    entry['strength'] = round(strength, 2)
    entry['saturation'] = round(saturation, 2)
    return entry

# ======================================================
# HDRI SCAN & RANKING
# ======================================================
def scan_and_rank_hdris(hdri_folder):
    rank_path = os.path.join(OUTPUT_ROOT, "hdri_ranking.json")
    if os.path.exists(rank_path):
        with open(rank_path) as f:
            ranked = json.load(f)
        print(f"Loaded existing HDRI ranking: {len(ranked)} HDRIs")
        for entry in ranked:
            entry = recalc_strength_and_saturation(entry)
        counts = {1:0, 2:0, 3:0, 4:0}
        for r in ranked: counts[r['tier']] += 1
        print(f"SUNNY:{counts[1]}  DECENT:{counts[2]}  SOFT:{counts[3]}  FLAT:{counts[4]}")
        with open(rank_path, 'w') as f:
            json.dump(ranked, f, indent=2)
        return ranked

    all_files = sorted(f for f in os.listdir(hdri_folder)
                       if f.lower().endswith(('.hdr', '.exr')))
    print(f"\nScanning {len(all_files)} HDRIs (first time)...")

    ranked = []
    for i, fname in enumerate(all_files):
        fpath = os.path.join(hdri_folder, fname)
        try:
            img    = bpy.data.images.load(fpath, check_existing=False)
            pixels = img.pixels[:]
            n      = len(pixels) // 4
            bpy.data.images.remove(img)
            if n == 0: continue

            step = max(1, n // 20000)
            lums = []
            for j in range(0, len(pixels) - 3, step * 4):
                r, g, b = pixels[j], pixels[j+1], pixels[j+2]
                lums.append(0.299*r + 0.587*g + 0.114*b)
            if not lums: continue

            avg_lum   = sum(lums) / len(lums)
            peak_lum  = max(lums)
            dark_vals = [v for v in lums if v > 0.0001]
            dark_lum  = min(dark_vals) if dark_vals else 0.0001
            dyn_range = peak_lum / max(dark_lum, 0.0001)
            score     = (peak_lum * 0.65) + (min(dyn_range, 500) / 500 * 100 * 0.35)

            if   score >= TIER_SUNNY:  tier, tier_name = 1, "SUNNY"
            elif score >= TIER_DECENT: tier, tier_name = 2, "DECENT"
            elif score >= TIER_SOFT:   tier, tier_name = 3, "SOFT"
            else:                      tier, tier_name = 4, "FLAT"

            entry = {
                "filename": fname, "score": round(score, 2),
                "tier": tier, "tier_name": tier_name,
                "avg_lum": round(avg_lum, 4), "peak_lum": round(peak_lum, 2),
                "dyn_range": round(dyn_range, 1),
            }
            entry = recalc_strength_and_saturation(entry)
            ranked.append(entry)
            print(f"  [{i+1:03d}] {tier_name:6s} score={score:6.1f} str={entry['strength']:.2f} sat={entry['saturation']:.2f} {fname}")

        except Exception as e:
            print(f"  ERROR {fname}: {e}")

    ranked.sort(key=lambda x: (x['tier'], -x['score']))
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with open(rank_path, 'w') as f:
        json.dump(ranked, f, indent=2)
    print(f"Ranking saved to {rank_path}")
    return ranked

# ======================================================
# MATERIAL SETUP - same as before
# ======================================================
def classify_material(mat):
    name = mat.name.lower()
    if any(k in name for k in ['hair', 'fur', 'strand', 'eyebrow', 'lash', 'brow']):
        return 'hair'
    if any(k in name for k in ['eye', 'cornea', 'iris', 'sclera', 'pupil']):
        return 'eye'
    if any(k in name for k in ['teeth', 'tooth', 'gum']):
        return 'teeth'
    return 'skin'

def setup_skin_material(bsdf):
    inp = bsdf.inputs
    if 'Subsurface Weight' in inp:
        inp['Subsurface Weight'].default_value = 0.15
    if 'Subsurface Radius' in inp:
        inp['Subsurface Radius'].default_value = (1.0, 0.5, 0.25)
    if 'Subsurface Scale' in inp:
        inp['Subsurface Scale'].default_value = 0.018
    if 'Subsurface IOR' in inp:
        inp['Subsurface IOR'].default_value = 1.4
    if 'Roughness' in inp:
        inp['Roughness'].default_value = 0.6
    if 'Specular IOR Level' in inp:
        inp['Specular IOR Level'].default_value = 0.2
    if 'IOR' in inp:
        inp['IOR'].default_value = 1.4
    if 'Sheen Weight' in inp:
        inp['Sheen Weight'].default_value = 0.03
    if 'Sheen Roughness' in inp:
        inp['Sheen Roughness'].default_value = 0.6

def setup_hair_material(mat, bsdf):
    inp = bsdf.inputs
    if 'Subsurface Weight'  in inp: inp['Subsurface Weight'].default_value = 0.0
    if 'Specular IOR Level' in inp: inp['Specular IOR Level'].default_value = 0.2
    if 'Anisotropic'        in inp: inp['Anisotropic'].default_value = 0.6
    if 'Anisotropic Rotation' in inp: inp['Anisotropic Rotation'].default_value = 0.05
    if 'Sheen Weight'       in inp: inp['Sheen Weight'].default_value = 0.0

    try:
        nt = mat.node_tree
        nodes = nt.nodes
        links = nt.links
        if any(n.name == 'HairRoughnessNoise' for n in nodes):
            return
        noise = nodes.new("ShaderNodeTexNoise")
        noise.name = 'HairRoughnessNoise'
        noise.inputs["Scale"].default_value = 8.0
        noise.inputs["Detail"].default_value = 4.0
        noise.inputs["Roughness"].default_value = 0.6
        noise.inputs["Distortion"].default_value = 0.1
        map_range = nodes.new("ShaderNodeMapRange")
        map_range.name = 'HairRoughnessMap'
        map_range.inputs["From Min"].default_value = 0.0
        map_range.inputs["From Max"].default_value = 1.0
        map_range.inputs["To Min"].default_value = 0.40
        map_range.inputs["To Max"].default_value = 0.65
        links.new(noise.outputs["Fac"], map_range.inputs["Value"])
        links.new(map_range.outputs["Result"], inp["Roughness"])
        print(f"      Hair noise added to {mat.name}")
    except Exception as e:
        if 'Roughness' in inp:
            inp['Roughness'].default_value = 0.52
        print(f"      Hair noise fallback: {e}")

def setup_eye_material(bsdf):
    inp = bsdf.inputs
    if 'Subsurface Weight'  in inp: inp['Subsurface Weight'].default_value = 0.0
    if 'Roughness'          in inp: inp['Roughness'].default_value = 0.05
    if 'Specular IOR Level' in inp: inp['Specular IOR Level'].default_value = 0.8
    if 'IOR'                in inp: inp['IOR'].default_value = 1.45

def setup_teeth_material(bsdf):
    inp = bsdf.inputs
    if 'Subsurface Weight'  in inp: inp['Subsurface Weight'].default_value = 0.08
    if 'Roughness'          in inp: inp['Roughness'].default_value = 0.45
    if 'Specular IOR Level' in inp: inp['Specular IOR Level'].default_value = 0.35

def setup_materials():
    for obj in scene.objects:
        if obj.type != 'MESH':
            continue
        obj.visible_shadow       = True
        obj.visible_diffuse      = True
        obj.visible_glossy       = True
        obj.visible_transmission = True
        for slot in obj.material_slots:
            mat = slot.material
            if not mat or not mat.node_tree:
                continue
            try:
                bsdf = next((n for n in mat.node_tree.nodes
                             if n.type == 'BSDF_PRINCIPLED'), None)
                if not bsdf:
                    continue
                mat_type = classify_material(mat)
                if   mat_type == 'skin':  setup_skin_material(bsdf)
                elif mat_type == 'hair':  setup_hair_material(mat, bsdf)
                elif mat_type == 'eye':   setup_eye_material(bsdf)
                elif mat_type == 'teeth': setup_teeth_material(bsdf)
                print(f"    [{mat_type.upper():6s}] {mat.name}")
            except Exception as e:
                print(f"    Material error {mat.name}: {e}")

# ======================================================
# CAMERA HELPERS (unchanged)
# ======================================================
def look_at(cam, target):
    direction = target - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

def normalize_and_center():
    meshes = [o for o in scene.objects if o.type == 'MESH']
    if not meshes:
        return 1.0, Vector((0, 0, 0))
    for obj in meshes:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        obj.select_set(False)
    bpy.context.view_layer.update()
    verts  = [obj.matrix_world @ v.co for obj in meshes for v in obj.data.vertices]
    min_co = Vector((min(v.x for v in verts), min(v.y for v in verts), min(v.z for v in verts)))
    max_co = Vector((max(v.x for v in verts), max(v.y for v in verts), max(v.z for v in verts)))
    center  = (min_co + max_co) / 2.0
    max_dim = max(max_co - min_co)
    if max_dim < 0.001:
        return 1.0, Vector((0, 0, 0))
    scale = 2.0 / max_dim
    for obj in meshes:
        obj.location = -center
        obj.scale    = Vector((scale, scale, scale))
    bpy.context.view_layer.update()
    verts   = [obj.matrix_world @ v.co for obj in meshes for v in obj.data.vertices]
    min_co2 = Vector((min(v.x for v in verts), min(v.y for v in verts), min(v.z for v in verts)))
    max_co2 = Vector((max(v.x for v in verts), max(v.y for v in verts), max(v.z for v in verts)))
    center2 = (min_co2 + max_co2) / 2.0
    radius  = max(max_co2 - min_co2) / 2.0
    return radius, center2

def create_camera():
    bpy.ops.object.camera_add(location=(0, -5, 0))
    cam = bpy.context.active_object
    cam.data.lens        = CAMERA_LENS_MM
    cam.data.clip_start  = 0.01
    cam.data.clip_end    = 100.0
    cam.data.dof.use_dof = False
    scene.camera = cam
    return cam

def place_camera(cam, center, radius, yaw_deg):
    fov  = cam.data.angle
    dist = (radius / math.tan(fov / 2.0)) * CAMERA_DIST_MULT
    yaw  = math.radians(yaw_deg)
    aim_target = center + Vector((0, 0, 0.12))
    cam.location = center + Vector((
        math.sin(yaw) * dist,
        -math.cos(yaw) * dist,
        dist * CAMERA_ELEVATION
    ))
    look_at(cam, aim_target + Vector((0, 0, 0.20)))

def rotate_hdri(deg):
    mapping.inputs["Rotation"].default_value = (0, 0, math.radians(deg))

def already_done(meta_set, hdri_name, cam_pos, rot_deg):
    return (hdri_name, cam_pos, round(rot_deg, 2)) in meta_set

def build_done_set(meta_list):
    return {
        (e['hdri_name'], e['camera_position'], round(e['hdri_rotation_deg'], 2))
        for e in meta_list
    }

# ======================================================
# EXPOSURE ADJUSTMENT (same as before)
# ======================================================
def measure_face_luminance_tempfile():
    orig_filepath = scene.render.filepath
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        temp_path = tmp.name
    scene.render.filepath = temp_path
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(temp_path)
    pixels = img.pixels[:]
    h, w = img.size[1], img.size[0]
    bpy.data.images.remove(img)
    os.unlink(temp_path)
    scene.render.filepath = orig_filepath

    if h == 0 or w == 0 or not pixels:
        return None

    lum_sum = 0
    count = 0
    for y in range(h//4, 3*h//4):
        for x in range(w//4, 3*w//4):
            idx = (y * w + x) * 4
            if idx + 2 >= len(pixels):
                continue
            r, g, b = pixels[idx], pixels[idx+1], pixels[idx+2]
            lum = 0.299*r + 0.587*g + 0.114*b
            lum_sum += lum
            count += 1
    if count == 0:
        return None
    return lum_sum / count

def adjust_exposure_for_current_view(base_exposure):
    orig_samples = scene.cycles.samples
    orig_percent = scene.render.resolution_percentage
    scene.cycles.samples = 32
    scene.render.resolution_percentage = 25
    try:
        avg_lum = measure_face_luminance_tempfile()
        if avg_lum is None:
            print("      Test render failed to measure luminance, using base exposure")
            return base_exposure
        new_exposure = base_exposure
        if avg_lum < TARGET_LUM_MIN:
            needed = math.log2(TARGET_LUM_MIN / avg_lum)
            new_exposure = base_exposure + needed * 0.5
        elif avg_lum > TARGET_LUM_MAX:
            needed = math.log2(TARGET_LUM_MAX / avg_lum)
            new_exposure = base_exposure + needed * 0.5
        new_exposure = max(-2.0, min(2.0, new_exposure))
        print(f"      Test: face lum={avg_lum:.3f} -> exposure {base_exposure:+.2f} → {new_exposure:+.2f}")
        return new_exposure
    except Exception as e:
        print(f"      Test render error: {e}")
        return base_exposure
    finally:
        scene.cycles.samples = orig_samples
        scene.render.resolution_percentage = orig_percent

# ======================================================
# MAIN
# ======================================================
RANKED_HDRIS = scan_and_rank_hdris(HDRI_FOLDER)
OBJ_FILES    = sorted(f for f in os.listdir(FACE_ROOT_INPUT)
                      if f.lower().endswith('.obj'))

print(f"\nSubjects: {len(OBJ_FILES)} | HDRIs: {len(RANKED_HDRIS)}")
print("FINE-TUNED SETTINGS (slightly reduced oversaturation):")
print("  - Sunny strengths: 0.45–0.65")
print("  - Target face luminance: 0.5–0.75")
print("  - Base exposure for sunny: +0.15")
print("  - Exposure adjustment factor: 0.5\n")

# [TIME ESTIMATION] Calculate total planned renders
total_planned = len(OBJ_FILES) * len(RANKED_HDRIS) * len(CAMERA_POSITIONS) * NUM_HDRI_ROTATIONS
print(f"Total planned renders: {total_planned}")

# [TIME ESTIMATION] Variables for moving average
render_times = []
avg_render_time = 0.0
last_progress_print = 0

total_renders = 0
rot_angles = [i * 360.0 / NUM_HDRI_ROTATIONS for i in range(NUM_HDRI_ROTATIONS)]

for obj_file in OBJ_FILES:
    obj_path = os.path.join(FACE_ROOT_INPUT, obj_file)
    print(f"\n{'='*60}")
    print(f"SUBJECT: {obj_file}")

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    cam = create_camera()

    try:
        bpy.ops.wm.obj_import(filepath=obj_path,
                               forward_axis='NEGATIVE_Z', up_axis='Y')
    except Exception as e:
        print(f"  Import FAILED: {e}")
        continue

    meshes = [o for o in scene.objects if o.type == 'MESH']
    if not meshes:
        continue

    if any(len(o.material_slots) > 0 for o in meshes):
        setup_materials()

    radius, face_center = normalize_and_center()

    out_dir   = os.path.join(OUTPUT_ROOT, obj_file[:-4])
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "metadata.json")

    meta_list = []
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta_list = json.load(f)
        print(f"  Resuming from {len(meta_list)} existing renders")
    else:
        print(f"  Starting fresh")

    done_set = build_done_set(meta_list)
    obj_new = obj_skip = 0

    for rank_idx, hdri_info in enumerate(RANKED_HDRIS):
        hdri_name = hdri_info['filename']
        hdri_path = os.path.join(HDRI_FOLDER, hdri_name)

        all_done = all(
            already_done(done_set, hdri_name, pos, rot_deg)
            for pos in CAMERA_POSITIONS
            for rot_deg in rot_angles
        )
        if all_done:
            obj_skip += len(CAMERA_POSITIONS) * NUM_HDRI_ROTATIONS
            continue

        try:
            hdri_img  = bpy.data.images.load(hdri_path, check_existing=False)
            env.image = hdri_img
        except Exception as e:
            print(f"  Load FAILED {hdri_name}: {e}")
            continue

        bg.inputs["Strength"].default_value = hdri_info['strength']

        desat_node = world.node_tree.nodes.get("DesaturateBrightHDRI")
        if desat_node and hdri_info.get('saturation', 1.0) < 1.0:
            desat_node.inputs["Saturation"].default_value = hdri_info['saturation']
            wt_links.new(env.outputs["Color"], desat_node.inputs["Color"])
            wt_links.new(desat_node.outputs["Color"], bg.inputs["Color"])
        else:
            wt_links.new(env.outputs["Color"], bg.inputs["Color"])

        # Base exposure: sunny +0.15, decent +0.1, soft 0.0, flat +0.2
        if hdri_info['tier'] == 1:   # SUNNY
            base_exposure = 0.15
        elif hdri_info['tier'] == 2: # DECENT
            base_exposure = 0.1
        elif hdri_info['tier'] == 3: # SOFT
            base_exposure = 0.0
        else:                         # FLAT
            base_exposure = 0.2

        print(f"\n  [{rank_idx+1:03d}/{len(RANKED_HDRIS)}] "
              f"[{hdri_info['tier_name']:6s}] str={hdri_info['strength']:.2f} "
              f"sat={hdri_info.get('saturation', 1.0):.2f} base exp={base_exposure:+.2f} {hdri_name}")

        for pos_name, yaw in CAMERA_POSITIONS.items():
            place_camera(cam, face_center, radius, yaw)

            for rot_idx, rot_deg in enumerate(rot_angles):
                if already_done(done_set, hdri_name, pos_name, rot_deg):
                    obj_skip += 1
                    continue

                rotate_hdri(rot_deg)
                rot_exposure = adjust_exposure_for_current_view(base_exposure)
                scene.view_settings.exposure = rot_exposure

                img_name = (f"t{hdri_info['tier']}_r{rank_idx:03d}"
                            f"_{pos_name}_rot{rot_idx:02d}.png")
                scene.render.filepath = os.path.join(out_dir, img_name)

                # [TIME ESTIMATION] Record start time
                start_time = time.time()

                try:
                    bpy.ops.render.render(write_still=True)
                    obj_new += 1
                    total_renders += 1

                    # [TIME ESTIMATION] Calculate elapsed and update moving average
                    elapsed = time.time() - start_time
                    render_times.append(elapsed)
                    if len(render_times) > 20:
                        render_times.pop(0)
                    avg_render_time = sum(render_times) / len(render_times)

                    # [TIME ESTIMATION] Print ETA every 5 renders
                    if total_renders % 5 == 0 and total_renders != last_progress_print:
                        last_progress_print = total_renders
                        remaining = total_planned - total_renders
                        eta_seconds = avg_render_time * remaining
                        hours = int(eta_seconds // 3600)
                        minutes = int((eta_seconds % 3600) // 60)
                        seconds = int(eta_seconds % 60)
                        print(f"      ⏱️  Avg render time: {avg_render_time:.2f}s | "
                              f"Rendered: {total_renders}/{total_planned} | "
                              f"ETA: {hours}h {minutes}m {seconds}s")

                    entry = {
                        "filename":          img_name,
                        "hdri_name":         hdri_name,
                        "hdri_rank":         rank_idx + 1,
                        "hdri_tier":         hdri_info['tier'],
                        "hdri_tier_name":    hdri_info['tier_name'],
                        "hdri_score":        hdri_info['score'],
                        "hdri_avg_lum":      hdri_info['avg_lum'],
                        "hdri_peak_lum":     hdri_info['peak_lum'],
                        "hdri_dyn_range":    hdri_info['dyn_range'],
                        "hdri_strength":     hdri_info['strength'],
                        "hdri_saturation":   hdri_info.get('saturation', 1.0),
                        "hdri_rotation_deg": round(rot_deg, 2),
                        "hdri_rotation_idx": rot_idx,
                        "camera_position":   pos_name,
                        "camera_yaw_deg":    yaw,
                        "exposure_used":     round(rot_exposure, 2),
                    }
                    meta_list.append(entry)
                    done_set.add((hdri_name, pos_name, round(rot_deg, 2)))

                    if obj_new % 20 == 0:
                        with open(meta_path, 'w') as f:
                            json.dump(meta_list, f, indent=2)
                        print(f"      ✓ {obj_new} rendered | {obj_skip} skipped")

                except Exception as e:
                    print(f"    RENDER ERROR: {e}")

                scene.view_settings.exposure = base_exposure

        try:
            bpy.data.images.remove(hdri_img)
        except:
            pass

    with open(meta_path, 'w') as f:
        json.dump(meta_list, f, indent=2)
    print(f"\n  Subject done: {obj_new} new | {obj_skip} skipped")

print(f"\n{'='*60}")
print(f"COMPLETE — total new renders: {total_renders}")
print(f"{'='*60}")
bpy.ops.wm.quit_blender()
