import bpy
from mathutils import Matrix, Vector
import sys
import json
import os

import bpy
import os

# This is the script used to render depth of field images with a set gaze used in the paper.
# Before using this script, first set up the scene with the desired camera pose, resolution etc.

# To change the focal point, edit the code at the end of this script.

class RenderOutputs:
    def __init__(self, output_base_path):
        bpy.context.scene.use_nodes = True
        self.nodes = bpy.context.scene.node_tree.nodes
        self.links = bpy.context.scene.node_tree.links
        self.links.clear()
        self.nodes.clear()

        #Enable extra render passes
        bpy.context.scene.view_layers[0].use_pass_normal = True
        self.render_layers = self.nodes.new("CompositorNodeRLayers")

        #Create output nodes
        self.file_output = self.nodes.new("CompositorNodeOutputFile")
        self.file_output.file_slots.new("color")
        self.color_input = self.file_output.inputs["color"]
        self.color_file_slot = self.file_output.file_slots["color"]
        self.color_file_slot.use_node_format = False
        self.color_file_slot.format.file_format = "PNG"
        self.color_file_slot.format.color_mode = "RGB"
        self.color_file_slot.format.color_depth = "8"

        self.file_output.file_slots.new("color_hdr")
        self.color_hdr_input = self.file_output.inputs["color_hdr"]
        self.color_hdr_file_slot = self.file_output.file_slots["color_hdr"]
        self.color_hdr_file_slot.use_node_format = False
        self.color_hdr_file_slot.format.file_format = "OPEN_EXR"
        self.color_hdr_file_slot.format.color_mode = "RGB"
        self.color_hdr_file_slot.format.color_depth = "32"
        
        self.file_output.file_slots.new("normal")
        self.normal_input = self.file_output.inputs["normal"]
        self.normal_file_slot = self.file_output.file_slots["normal"]
        self.normal_file_slot.use_node_format = False
        self.normal_file_slot.format.file_format = "OPEN_EXR"
        self.normal_file_slot.format.color_mode = "RGB"
        self.normal_file_slot.format.color_depth = "32"

        self.file_output.base_path = os.path.join(bpy.path.abspath("//"), output_base_path)

    def connect_color_output(self, output_path, save_hdr):
        self.links.clear()
        self.color_file_slot.path = output_path
        self.links.new(self.render_layers.outputs["Image"], self.color_input)
        if save_hdr:
            self.color_hdr_file_slot.path = output_path
            self.links.new(self.render_layers.outputs["Image"], self.color_hdr_input)


    def connect_normal_output(self, output_path):
        self.links.clear()
        self.normal_file_slot.path = output_path
        self.links.new(self.render_layers.outputs["Normal"], self.normal_input)

"""
Given a normalized image location, casts a ray through this location into the scene.
Returns collision data for any collided objects.

Returns:
    collision: Bool indicating if collision occurred
    origin: Starting point of ray
    location: Collision location
    distance: Collision distance
    name: Collided object name
    
"""
def raycast_image_loc(image_loc):
    camera = bpy.context.scene.camera
    origin = camera.location
    render = bpy.context.scene.render
    
    projection_matrix = camera.calc_matrix_camera(
        bpy.data.scenes[0].view_layers[0].depsgraph,
        x = render.resolution_x,
        y = render.resolution_y,
        scale_x = render.pixel_aspect_x,
        scale_y = render.pixel_aspect_y,
    )
    inv_projection_matrix = projection_matrix.inverted()
    cam_to_world = camera.matrix_world
    
    norm_loc = Vector([2*image_loc[0] - 1, (1 - 2*image_loc[1]), 1])
    proj_point_cam = inv_projection_matrix @ norm_loc
    proj_point = cam_to_world @ proj_point_cam
    direction = proj_point - origin
    
    collision, location, normal, index, object, matrix = bpy.context.scene.ray_cast(bpy.context.scene.view_layers[0].depsgraph, origin, direction)
    
    return collision, origin, location, (location - origin).length, object.name


def render_image_with_focus(image_loc, render_outputs, base_filename, fstop=1.8):
    collision, origin, location, distance, name = raycast_image_loc(image_loc)
    if not collision:
        raise Exception("Cannot render: no object to focus on for this image location")
    
    camera = bpy.context.scene.camera
    camera.data.dof.use_dof = True
    camera.data.dof.focus_distance = distance
    camera.data.dof.aperture_fstop = fstop
    
    json_data = {
        "gaze location": image_loc,
        "distance": distance,
        "camera world location": [origin[0], origin[1], origin[2]],
        "gaze world location": [location[0], location[1], location[2]],
        "object in focus": name,
        "camera aperture f stop": fstop
    }
    
    json_filename = "%s%04d.json" % (base_filename, bpy.context.scene.frame_current)
    jsonfile = open(json_filename, "w")
    json.dump(json_data, jsonfile, indent=4)
    render_outputs.connect_color_output(base_filename, save_hdr=False)
    bpy.ops.render.render()

min_focus_x = 0.1
max_focus_x = 0.9
min_focus_y = 0.4
max_focus_y = 0.6
n_focus_x = 5
n_focus_y = 2

fstop = 1.4

render_outputs = RenderOutputs("OutputImages/")

# Use the below code to render a single image with a chosen gaze location. Change the location as desired.
render_image_with_focus([0.5, 0.3], render_outputs, "image_focus_%04d_frame_" % (0), fstop=1.4)

# Uncomment and use the code below to render a series of image with focus points located at grid points in 
# the image. Chnage n_focus_x and n_focus_y to vary the number of rows & columns in the grid.
# For a focus sweep as in the paper, set n_focus_y to 1 and n_focus_x to the desired frame count.
"""
for i in range(n_focus_x):
    for j in range(n_focus_y):
        focus_x = min_focus_x + (i / (n_focus_x-1))*(max_focus_x - min_focus_x)
        focus_y = min_focus_y + (j / (n_focus_y-1))*(max_focus_y - min_focus_y)
        
        render_image_with_focus([focus_x, focus_y], render_outputs, "image_focus_%04d_frame_" % (i*n_focus_y + j), fstop=1.4)
"""
