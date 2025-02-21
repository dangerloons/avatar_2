import bpy
import os

def create_base_mesh(gender="male"):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()
    
    # Create a base mesh with a head and body
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1.0, location=(0, 0, 1.8))
    head = bpy.context.object
    head.name = f"{gender}_head"
    
    bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=0.5, depth=1.2, location=(0, 0, 1.2))
    neck = bpy.context.object
    neck.name = f"{gender}_neck"
    
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.6, depth=3, location=(0, 0, 0.6))
    torso = bpy.context.object
    torso.name = f"{gender}_torso"
    
    # Join the objects into a single mesh
    bpy.ops.object.select_all(action='DESELECT')
    head.select_set(True)
    neck.select_set(True)
    torso.select_set(True)
    bpy.context.view_layer.objects.active = head
    bpy.ops.object.join()
    
    # Apply smooth shading
    bpy.ops.object.shade_smooth()
    
    # Apply transformations
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

def export_model(output_dir="base_models", gender="male"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_path = os.path.join(output_dir, f"{gender}_base.obj")
    bpy.ops.export_scene.obj(filepath=file_path, use_selection=True)
    print(f"Model exported: {file_path}")

def main():
    for gender in ["male", "female"]:
        create_base_mesh(gender)
        export_model(gender=gender)
    print("Base models generated successfully.")

if __name__ == "__main__":
    main()
