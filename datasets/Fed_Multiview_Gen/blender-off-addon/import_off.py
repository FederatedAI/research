#####
#
# Copyright 2014 Alex Tsui
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#####

#
# http://wiki.blender.org/index.php/Dev:2.5/Py/Scripts/Guidelines/Addons
#
import os
import bpy
import mathutils
from bpy.props import (BoolProperty,
    FloatProperty,
    StringProperty,
    EnumProperty,
    )
from bpy_extras.io_utils import (ImportHelper,
    ExportHelper,
    unpack_list,
    unpack_face_list,
    axis_conversion,
    )

#if "bpy" in locals():
#    import imp
#    if "import_off" in

bl_info = {
    "name": "OFF format",
    "description": "Import-Export OFF, Import/export simple OFF mesh.",
    "author": "Alex Tsui, Mateusz Kłoczko",
    "version": (0, 3),
    "blender": (2, 74, 0),
    "location": "File > Import-Export",
    "warning": "", # used for warning icon and text in addons panel
    "wiki_url": "http://wiki.blender.org/index.php/Extensions:2.5/Py/"
                "Scripts/My_Script",
    "category": "Import-Export"}

class ImportOFF(bpy.types.Operator, ImportHelper):
    """Load an OFF Mesh file"""
    bl_idname = "import_mesh.off"
    bl_label = "Import OFF Mesh"
    filename_ext = ".off"
    filter_glob = StringProperty(
        default="*.off",
        options={'HIDDEN'},
    )

    axis_forward = EnumProperty(
            name="Forward",
            items=(('X', "X Forward", ""),
                   ('Y', "Y Forward", ""),
                   ('Z', "Z Forward", ""),
                   ('-X', "-X Forward", ""),
                   ('-Y', "-Y Forward", ""),
                   ('-Z', "-Z Forward", ""),
                   ),
            default='Y',
            )
    axis_up = EnumProperty(
            name="Up",
            items=(('X', "X Up", ""),
                   ('Y', "Y Up", ""),
                   ('Z', "Z Up", ""),
                   ('-X', "-X Up", ""),
                   ('-Y', "-Y Up", ""),
                   ('-Z', "-Z Up", ""),
                   ),
            default='Z',
            )

    def execute(self, context):
        #from . import import_off

        keywords = self.as_keywords(ignore=('axis_forward',
            'axis_up',
            'filter_glob',
        ))
        global_matrix = axis_conversion(from_forward=self.axis_forward,
            from_up=self.axis_up,
            ).to_4x4()

        mesh = load(self, context, **keywords)
        if not mesh:
            return {'CANCELLED'}

        scene = bpy.context.scene
        obj = bpy.data.objects.new(mesh.name, mesh)
        scene.objects.link(obj)
        scene.objects.active = obj
        obj.select = True

        obj.matrix_world = global_matrix

        scene.update()

        return {'FINISHED'}

class ExportOFF(bpy.types.Operator, ExportHelper):
    """Save an OFF Mesh file"""
    bl_idname = "export_mesh.off"
    bl_label = "Export OFF Mesh"
    filter_glob = StringProperty(
        default="*.off",
        options={'HIDDEN'},
    )
    check_extension = True
    filename_ext = ".off"

    axis_forward = EnumProperty(
            name="Forward",
            items=(('X', "X Forward", ""),
                   ('Y', "Y Forward", ""),
                   ('Z', "Z Forward", ""),
                   ('-X', "-X Forward", ""),
                   ('-Y', "-Y Forward", ""),
                   ('-Z', "-Z Forward", ""),
                   ),
            default='Y',
            )
    axis_up = EnumProperty(
            name="Up",
            items=(('X', "X Up", ""),
                   ('Y', "Y Up", ""),
                   ('Z', "Z Up", ""),
                   ('-X', "-X Up", ""),
                   ('-Y', "-Y Up", ""),
                   ('-Z', "-Z Up", ""),
                   ),
            default='Z',
            )
    use_colors = BoolProperty(
            name="Vertex Colors",
            description="Export the active vertex color layer",
            default=False,
            )

    def execute(self, context):
        keywords = self.as_keywords(ignore=('axis_forward',
            'axis_up',
            'filter_glob',
            'check_existing',
        ))
        global_matrix = axis_conversion(to_forward=self.axis_forward,
            to_up=self.axis_up,
            ).to_4x4()
        keywords['global_matrix'] = global_matrix
        return save(self, context, **keywords)

def menu_func_import(self, context):
    self.layout.operator(ImportOFF.bl_idname, text="OFF Mesh (.off)")

def menu_func_export(self, context):
    self.layout.operator(ExportOFF.bl_idname, text="OFF Mesh (.off)")

def register():
    bpy.utils.register_module(__name__)
    bpy.types.INFO_MT_file_import.append(menu_func_import)
    bpy.types.INFO_MT_file_export.append(menu_func_export)

def unregister():
    bpy.utils.unregister_module(__name__)
    bpy.types.INFO_MT_file_import.remove(menu_func_import)
    bpy.types.INFO_MT_file_export.remove(menu_func_export)

def load(operator, context, filepath):
    # Parse mesh from OFF file
    # TODO: Add support for NOFF and COFF
    filepath = os.fsencode(filepath)
    file = open(filepath, 'r')
    first_line = file.readline().rstrip()
    use_colors = (first_line == 'COFF')
    colors = []
    
    # handle blank and comment lines after the first line
    line = file.readline()
    while line.isspace() or line[0]=='#':
        line = file.readline()

    vcount, fcount, ecount = [int(x) for x in line.split()]
    verts = []
    facets = []
    edges = []
    i=0;
    while i<vcount:
        line = file.readline()
        if line.isspace():
            continue    # skip empty lines
        try:
             bits = [float(x) for x in line.split()]
             px = bits[0]
             py = bits[1]
             pz = bits[2]
             if use_colors:
                 colors.append([float(bits[3]) / 255, float(bits[4]) / 255, float(bits[5]) / 255])

        except ValueError:
            i=i+1
            continue
        verts.append((px, py, pz))
        i=i+1

    i=0;
    while i<fcount:
        line = file.readline()
        if line.isspace():
            continue    # skip empty lines
        try:
            splitted  = line.split()
            ids   = list(map(int, splitted))
            if len(ids) > 3:
                facets.append(tuple(ids[1:]))
            elif len(ids) == 3:
                edges.append(tuple(ids[1:]))
        except ValueError:
            i=i+1
            continue
        i=i+1

    # Assemble mesh
    off_name = bpy.path.display_name_from_filepath(filepath)
    mesh = bpy.data.meshes.new(name=off_name)
    mesh.from_pydata(verts,edges,facets)
    # mesh.vertices.add(len(verts))
    # mesh.vertices.foreach_set("co", unpack_list(verts))

    # mesh.faces.add(len(facets))
    # mesh.faces.foreach_set("vertices", unpack_face_list(facets))

    mesh.validate()
    mesh.update()

    if use_colors:
        color_data = mesh.vertex_colors.new()
        for i, facet in enumerate(mesh.polygons):
            for j, vidx in enumerate(facet.vertices):
                color_data.data[3*i + j].color = colors[vidx]

    return mesh

def save(operator, context, filepath,
    global_matrix = None,
    use_colors = False):
    # Export the selected mesh
    APPLY_MODIFIERS = True # TODO: Make this configurable
    if global_matrix is None:
        global_matrix = mathutils.Matrix()
    scene = context.scene
    obj = scene.objects.active
    mesh = obj.to_mesh(scene, APPLY_MODIFIERS, 'PREVIEW')

    # Apply the inverse transformation
    obj_mat = obj.matrix_world
    mesh.transform(global_matrix * obj_mat)

    verts = mesh.vertices[:]
    facets = [ f for f in mesh.tessfaces ]
    # Collect colors by vertex id
    colors = False
    vertex_colors = None
    if use_colors:
        colors = mesh.tessface_vertex_colors.active
    if colors:
        colors = colors.data
        vertex_colors = {}
        for i, facet in enumerate(mesh.tessfaces):
            color = colors[i]
            color = color.color1[:], color.color2[:], color.color3[:], color.color4[:]
            for j, vidx in enumerate(facet.vertices):
                if vidx not in vertex_colors:
                    vertex_colors[vidx] = (int(color[j][0] * 255.0),
                                            int(color[j][1] * 255.0),
                                            int(color[j][2] * 255.0))
    else:
        use_colors = False

    # Write geometry to file
    filepath = os.fsencode(filepath)
    fp = open(filepath, 'w')

    if use_colors:
        fp.write('COFF\n')
    else:
        fp.write('OFF\n')

    fp.write('%d %d 0\n' % (len(verts), len(facets)))

    for i, vert in enumerate(mesh.vertices):
        fp.write('%.16f %.16f %.16f' % vert.co[:])
        if use_colors:
            fp.write(' %d %d %d 255' % vertex_colors[i])
        fp.write('\n')

    #for facet in facets:
    for i, facet in enumerate(mesh.tessfaces):
        fp.write('%d' % len(facet.vertices))
        for vid in facet.vertices:
            fp.write(' %d' % vid)
        fp.write('\n')

    fp.close()

    return {'FINISHED'}

if __name__ == "__main__":
    register()
