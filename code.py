import bpy
import os
import math
from mathutils import Vector, Euler

# ======================================================
# PATHS (EDIT IF NEEDED)
# ======================================================

FACE_ROOT_INPUT = r"D:\dataset_photo"
HDRI_FOLDER = r"C:\Users\Vaibhav singh\Desktop\HDRI"
OUTPUT_ROOT = r"D:\test"

# ======================================================
# SETTINGS
# ======================================================

ANGLES = [-25, 0, 25]
RES = 512
SAMPLES = 128

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
    """Add a realistic skin material to the face mesh"""
    
    # Clear existing materials
    obj.data.materials.clear()
    
    # Create new material
    mat = bpy.data.materials.new(name="SkinMaterial")
    mat.use_nodes = True
    
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear default nodes
    nodes.clear()
    
    # Create shader nodes
    output = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    
    # Skin color (peachy tone)
    bsdf.inputs['Base Color'].default_value = (0.8, 0.6, 0.5, 1.0)
    
    # Subsurface scattering for realistic skin
    bsdf.inputs['Subsurface Weight'].default_value = 0.1
    bsdf.inputs['Subsurface Radius'].default_value = (1.0, 0.5, 0.25)
    bsdf.inputs['Subsurface Scale'].default_value = 0.01
    
    # Roughness
    bsdf.inputs['Roughness'].default_value = 0.4
    
    # Specular
    bsdf.inputs['Specular IOR Level'].default_value = 0.5
    
    # Connect nodes
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    # Assign material to object
    obj.data.materials.append(mat)
    
    print(f"    Added skin material to {obj.name}")

def normalize_and_orient_face():
    """Normalize face mesh and ensure it faces the camera"""
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    
    if not meshes:
        print("  ❌ No meshes found!")
        return 1.0, Vector((0, 0, 0))
    
    print(f"  Found {len(meshes)} mesh object(s)")
    
    # Apply transformations to each mesh
    for obj in meshes:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        obj.select_set(False)
        
        # Add material
        add_skin_material(obj)
    
    print(f"  ✓ Applied transforms and materials")
    
    bpy.context.view_layer.update()
    
    # Get all vertices in world space
    all_verts = []
    for obj in meshes:
        for v in obj.data.vertices:
            all_verts.append(obj.matrix_world @ v.co)
    
    if not all_verts:
        print("  ❌ No vertices found!")
        return 1.0, Vector((0, 0, 0))
    
    # Calculate bounds
    min_co = Vector((
        min(v.x for v in all_verts),
        min(v.y for v in all_verts),
        min(v.z for v in all_verts)
    ))
    
    max_co = Vector((
        max(v.x for v in all_verts),
        max(v.y for v in all_verts),
        max(v.z for v in all_verts)
    ))
    
    center = (min_co + max_co) / 2
    size = max_co - min_co
    max_dim = max(size)
    
    print(f"  Size: {size}")
    print(f"  Center: {center}")
    
    if max_dim < 0.001:
        print("  ❌ Object too small!")
        return 1.0, Vector((0, 0, 0))
    
    # Scale to fit in 2-unit cube
    target_size = 2.0
    scale_factor = target_size / max_dim
    
    # Center and scale all objects
    for obj in meshes:
        obj.location = -center
        obj.scale = Vector((scale_factor, scale_factor, scale_factor))
    
    bpy.context.view_layer.update()
    
    # Calculate final size for camera distance
    all_verts = []
    for obj in meshes:
        for v in obj.data.vertices:
            all_verts.append(obj.matrix_world @ v.co)
    
    min_co = Vector((min(v.x for v in all_verts), min(v.y for v in all_verts), min(v.z for v in all_verts)))
    max_co = Vector((max(v.x for v in all_verts), max(v.y for v in all_verts), max(v.z for v in all_verts)))
    final_size = max_co - min_co
    final_center = (min_co + max_co) / 2
    
    print(f"  Final size: {final_size}")
    print(f"  Final center: {final_center}")
    
    return max(final_size) / 2, final_center

def create_camera_and_lights():
    """Create camera and 3-point lighting setup"""
    
    # Camera
    bpy.ops.object.camera_add(location=(0, -5, 0))
    cam = bpy.context.active_object
    cam.data.lens = 50
    cam.data.clip_start = 0.01
    cam.data.clip_end = 100
    scene.camera = cam
    
    # Key light (main, bright, from upper right)
    bpy.ops.object.light_add(type='AREA', location=(2, -3, 3))
    light = bpy.context.active_object
    light.data.energy = 200
    light.data.size = 3
    
    # Fill light (softer, from left)
    bpy.ops.object.light_add(type='AREA', location=(-2, -3, 2))
    light = bpy.context.active_object
    light.data.energy = 100
    light.data.size = 3
    
    # Rim/back light
    bpy.ops.object.light_add(type='AREA', location=(0, 3, 2))
    light = bpy.context.active_object
    light.data.energy = 120
    light.data.size = 3
    
    return cam

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
        cam = create_camera_and_lights()

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

        # Render with different HDRIs and angles
        for hdri_idx, hdri in enumerate(HDRIS):
            env.image = hdri

            for ang in ANGLES:
                # Calculate camera distance
                fov = cam.data.angle
                dist = (radius * 1.3) / math.tan(fov / 2)  # Reduced from 1.8 to 1.3 for much larger face
                
                # Position camera around face center
                cam.location = face_center + Vector((
                    math.sin(math.radians(ang)) * dist,
                    -math.cos(math.radians(ang)) * dist,
                    dist * 0.1
                ))
                
                # Look at face center
                look_at(cam, face_center)
                
                print(f"  📷 ang={ang}°, hdri={hdri_idx+1}/{len(HDRIS)}")

                # Output
                out_dir = os.path.join(OUTPUT_ROOT, f"face_{face_id}", obj_file[:-4])
                os.makedirs(out_dir, exist_ok=True)
                
                idx = len([f for f in os.listdir(out_dir) if f.endswith('.png')])
                out_path = os.path.join(out_dir, f"img_{idx:04d}.png")
                scene.render.filepath = out_path

                # Render
                bpy.ops.render.render(write_still=True)
                render_count += 1

print(f"\n{'='*60}")
print(f"✅ COMPLETE: {render_count} renders")
print(f"{'='*60}")

bpy.ops.wm.quit_blender()
