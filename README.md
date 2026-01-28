Overall Purpose
This script creates a large dataset of face photos for training AI models. It takes 3D face models and renders them from multiple angles with different lighting environments to create realistic variations.
Step-by-Step Process
1. Setup Phase

Configures Blender to use GPU rendering for speed
Sets image resolution to 512×512 pixels
Enables denoising for cleaner images
Applies color grading (AgX - Medium High Contrast) for better visual quality

2. Environment Preparation

Loads up to 40 random HDRI images (High Dynamic Range Images) from your HDRI folder
These HDRIs provide realistic lighting environments (outdoor fields, studios, streets, etc.)
Each HDRI gets a random rotation so the same environment looks different each time

3. Camera Setup

Creates a camera with 100mm lens (portrait lens)
Enables depth-of-field blur (like real cameras, keeps face sharp, background soft)
Sets aperture to f/2.2 with 6 blades for realistic bokeh

4. Lighting Setup

Creates 3 studio lights in a professional "three-point lighting" arrangement:

Key Light: Main bright light (front-right, above)
Fill Light: Softer light to reduce harsh shadows (front-left)
Rim Light: Back light to create edge glow and separate face from background



5. Skin Shader Creation
The script creates a realistic skin material with:

Color variation: Subtle reddish and yellowish tones across the face (not uniform)
Roughness variation: Some areas shinier (forehead), some more matte (cheeks)
Subsurface scattering: Light penetrates and scatters inside the skin like real human skin, creating a soft glow
Three-layer texture:

Large-scale: Subtle facial contours and wrinkles
Medium-scale: General skin texture
Fine-scale: Individual pores


Proper physical properties: Uses real-world values for skin reflection and light behavior

6. Rendering Loop
For each 3D face model (.obj file) in your folder:
A. Load the model

Deletes any previous models
Imports the new face model

B. Apply realistic skin

Assigns the skin shader to all parts of the face

C. Analyze the face

Calculates the face's size and center point
Positions the 3 lights around the face based on its location

D. Render from multiple perspectives
For each of the 40 HDRIs:

For each of 3 camera angles (-30°, 0°, +30°):

Position camera at calculated distance
Rotate camera to the angle
Auto-focus camera on the face center
Save metadata about this render (which model, which HDRI, camera position, angle, timestamp)
Take the photo and save as PNG



7. Output
Per face model, you get:

40 HDRIs × 3 angles = 120 images

What you receive:

Numbered PNG images: img_0000.png, img_0001.png, etc.
One JSON file: metadata.json containing complete information about every single render

Metadata includes:

Which face model was used
Which HDRI background was used
How the HDRI was rotated
Camera angle (-30°, 0°, or 30°)
Camera distance and exact 3D position
When it was rendered

Key Features
Variation Sources

40 different HDRI environments (outdoor, indoor, studio, etc.)
Random HDRI rotations
3 camera angles per HDRI
Procedural skin variations (color, roughness, texture)
Adaptive lighting that moves with each face

Realism Techniques

Subsurface scattering for translucent skin glow
Depth-of-field camera blur
Three-point professional lighting
Procedural pores and skin texture
Color and roughness variation across the face
Physically-accurate light behavior

Speed Optimizations

GPU acceleration
Adaptive sampling (stops early if image is clean)
Denoising (allows fewer samples)
512×512 resolution (fast but detailed enough)
128 samples (balanced quality/speed)

Summary
You provide: 3D face models + HDRI environments
Script produces: Professional-looking face photos from multiple angles and lighting conditions, each documented with complete metadata for training AI models.
Invoke-WebRequest `
  -Uri "http://172.17.85.172:8000/yt.zip" `
  -OutFile "yt.zip"
  Invoke-WebRequest http://172.17.83.41:8000/hdri1.zip -OutFile hdri1.zip
  Invoke-WebRequest "http://172.17.83.41:8000/100-500%2B.zip" -OutFile "100-500+.zip"


