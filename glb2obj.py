import os
import shutil
import sys

import bpy

# ./blender --background --python glb2obj.py

def glb2obj(glb_path, obj_path):
	for obj in bpy.data.objects:
		obj.select_set(True)
	bpy.ops.object.delete()

	# import glb file
	bpy.ops.import_scene.gltf(filepath=glb_path)

	# unpack all images
	for img in bpy.data.images:
		bpy.ops.image.unpack(method = "WRITE_LOCAL", id=img.name)

	# change ShaderNodeTexImage & ShaderNodeBsdfDiffuse node when there is no ShaderNodeBsdfPrincipled
	save_ply = False
	for mat in bpy.data.materials:
		print("---------------")
		print("matrial: ", mat.name)
		if mat.node_tree:
			found_principled_BSDF = False
			found_vertex_color = False
			diffuse_BSDF_cnt = 0
			diffuse_BSDF_node = None
			tex_image_cnt = 0
			tex_image_node = None
			for node in mat.node_tree.nodes:
				print(node.bl_idname)
				if node.bl_idname == 'ShaderNodeBsdfPrincipled':
					found_principled_BSDF = True
				elif node.bl_idname == 'ShaderNodeBsdfDiffuse':
					diffuse_BSDF_cnt += 1
					diffuse_BSDF_node = node
				elif node.bl_idname == 'ShaderNodeTexImage':
					tex_image_cnt += 1
					tex_image_node = node
				elif node.bl_idname == 'ShaderNodeVertexColor':
					found_vertex_color = True
					# print(Fore.RED + "Warning! Currently do not support export per-vertex color.")
					# print(Style.RESET_ALL)
			
			if found_vertex_color: #and len(mat.node_tree.nodes) == 3:
				save_ply = True

			if found_principled_BSDF:
				print("found principled_BSDF node, no change")
				continue

			if diffuse_BSDF_cnt == 1:
				principled_node = mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
				output_node = mat.node_tree.nodes.get("Material Output")
				mat.node_tree.links.new(principled_node.outputs[0], output_node.inputs[0])
				if diffuse_BSDF_node.inputs["Color"].is_linked == False:
					principled_node.inputs["Base Color"].default_value = diffuse_BSDF_node.inputs["Color"].default_value
				else:
					from_socket = diffuse_BSDF_node.inputs["Color"].links[0].from_socket
					mat.node_tree.links.new(from_socket, principled_node.inputs["Base Color"])
				# print(Fore.RED + "change a diffuse_BSDF node to a principled_BSDF node")
				# print(Style.RESET_ALL)
				continue

			if diffuse_BSDF_cnt == 0 and tex_image_cnt == 1:
				principled_node = mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
				output_node = mat.node_tree.nodes.get("Material Output")
				mat.node_tree.links.new(principled_node.outputs[0], output_node.inputs[0])
				mat.node_tree.links.new(tex_image_node.outputs['Color'], principled_node.inputs["Base Color"])
				# print(Fore.RED + "change a TexImage node to a principledBSDF node")
				# print(Style.RESET_ALL)
				continue

	if save_ply:
		# print(Fore.RED + "save a ply file for per-vertex colors.")
		# print(Style.RESET_ALL)
		ply_path = obj_path.replace(".obj", ".ply")
		bpy.ops.export_mesh.ply(filepath=ply_path, axis_forward='Z', axis_up='Y', use_ascii=True)

	bpy.ops.export_scene.obj(filepath=obj_path, use_triangles=True, path_mode="COPY", axis_forward='Z', axis_up='Y')


glb_path = str(sys.argv[4])
obj_path = str(sys.argv[5])

obj_folder = os.path.dirname(obj_path)
os.makedirs(os.path.join(obj_folder, "textures"), exist_ok=True)
os.chdir(obj_folder)
glb2obj(glb_path, obj_path)
shutil.rmtree(os.path.join(obj_folder, "textures"))