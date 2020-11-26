'''
**********************************************************************************
THANKS for the work from WeiTang114 (Github:https://github.com/WeiTang114/BlenderPhong)
and
the work from zeaggler (Github:https://github.com/zeaggler/ModelNet_Blender_OFF2Multiview).

WeiTang114's work is for Linux system and zeaggler adapted it to Win10
**********************************************************************************

This file includes main operations inside blender:
1. load .off model
2. center model
3. normalize model
4. take a picture

'''

import bpy
import os.path
import math
import sys

C = bpy.context
D = bpy.data
scene = D.scenes['Scene']

# cameras: a list of camera positions
# a camera position is defined by two parameters: (theta, phi),
# where we fix the "r" of (r, theta, phi) in spherical coordinate system.

# 5 orientations: front, right, back, left, top
# cameras = [(60, 0), (60, 90), (60, 180), (60, 270),(0, 0)]


render_setting = scene.cycles

w = 1080
h = 1080

render_setting.resolution_x = w*2
render_setting.resolution_y = h*2
render_setting.resolution_percentage = 100

# output image size = (W, H)


RESIZE_FACTOR = 0.8

'''****************************************************************'''


def main():
    argv = sys.argv
    argv = argv[argv.index('--') + 1:]

    model_path = argv[0]    # input: path of single .off or dataset.txt
    image_dir = argv[1]     # input: path to save multiview images
    if len(argv) == 6:
        phi = int(argv[2])
        theta_interval = int(argv[3])
        phi_offset = int(argv[4])
        theta_offset = int(argv[5])
    else:
        phi = 60
        theta_interval = 30
        phi_offset = 0
        theta_offset = 0
    print(phi_offset, theta_offset)
    # multiview  with inter-deg elevation
    fixed_view = phi + phi_offset
    inter = theta_interval
    cameras = [(fixed_view, i) for i in range(0 + theta_offset, 360 + theta_offset, inter)]  # output 12(360/30=12) multiview images

    # blender has no native support for off files
    install_off_addon()
    init_camera()
    fix_camera_to_origin()

    '''*************************************************'''
    if model_path.split('.')[-1] == 'off':
        print('model path is ********', model_path) # model_path:'./airplane.off'
        do_model(model_path, image_dir, cameras)
    elif model_path.split('.')[-1] == 'txt':
        with open(model_path) as f:
            models = f.read().splitlines()
        for model in models:
            print('model path is ********', model) # model_path:'F:\DATA3D\ModelNet10\monitor\train\monitor_0003.off'
            do_model(model, image_dir, cameras)
    else:
        print('......Please input correct parameters......')
        exit(-1)
'''****************************************************************'''


def install_off_addon():
    print(os.path.dirname(__file__) +
            '/blender-off-addon/import_off.py')
    try:
        bpy.ops.wm.addon_install(
            overwrite=False,
            filepath=os.path.dirname(__file__) +
            '/blender-off-addon/import_off.py'
        )
        bpy.ops.wm.addon_enable(module='import_off')
    except Exception:
        print("""Import blender-off-addon failed.
              Did you pull the blender-off-addon submodule?
              $ git submodule update --recursive --remote
              """)
        exit(-1)


def init_camera():
    cam = D.objects['Camera']
    # select the camera object
    scene.objects.active = cam
    cam.select = True

    # set the rendering mode to orthogonal and scale
    C.object.data.type = 'ORTHO'
    C.object.data.ortho_scale = 2.


def fix_camera_to_origin():
    origin_name = 'Origin'

    # create origin
    try:
        origin = D.objects[origin_name]
    except KeyError:
        bpy.ops.object.empty_add(type='SPHERE')
        D.objects['Empty'].name = origin_name
        origin = D.objects[origin_name]

    origin.location = (0, 0, 0)

    cam = D.objects['Camera']
    scene.objects.active = cam
    cam.select = True

    if 'Track To' not in cam.constraints:
        bpy.ops.object.constraint_add(type='TRACK_TO')

    cam.constraints['Track To'].target = origin
    cam.constraints['Track To'].track_axis = 'TRACK_NEGATIVE_Z'
    cam.constraints['Track To'].up_axis = 'UP_Y'


def do_model(model_path, image_dir, cameras):
    name = load_model(model_path)
    center_model(name)
    normalize_model(name)

    image_subdir = os.path.join(image_dir)
    for i, c in enumerate(cameras):
        move_camera(c)
        render()
        save(image_subdir, '{}_{:03d}_{:03d}_{:03d}'.format(name, c[0], c[1], i+1))

    delete_model(name)


def load_model(model_path):
    # single .off: model_path='./airplane.off'
    # dataset.txt: model_path= 'F:\\DATA3D\ModelNet10_MV\\bathtub\\train\\bathtub_0003.off'
    d = os.path.dirname(model_path) # invalide for .off file
    ext = model_path.split('.')[-1] # ext: 'off'

    # Attention!  win10: ..path.split('\\')  linux: ..path.split('/')
    _model_path_tmp = model_path.split('\\')[-1] # _model_path_tmp: 'bathtub_0003.off'
    name = os.path.basename(_model_path_tmp).split('.')[0] # bathtub_0003
    # handle weird object naming by Blender for stl files
    if ext == 'stl':
        name = name.title().replace('_', ' ')

    if name not in D.objects:
        print('loading :' + name)
        if ext == 'stl':
            bpy.ops.import_mesh.stl(filepath=model_path, directory=d,
                                    filter_glob='*.stl')
        elif ext == 'off':
            bpy.ops.import_mesh.off(filepath=model_path, filter_glob='*.off')
        elif ext == 'obj':
            bpy.ops.import_scene.obj(filepath=model_path, filter_glob='*.obj')
        else:
            print('Currently .{} file type is not supported.'.format(ext))
            exit(-1)
    return name # name='airplane' -> 'bathtub_0003'


def delete_model(name):
    for ob in scene.objects:
        if ob.type == 'MESH' and ob.name.startswith(name):
            ob.select = True
        else:
            ob.select = False
    bpy.ops.object.delete()


def center_model(name):
    bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN')
    D.objects[name].location = (0, 0, 0)


def normalize_model(name):
    obj = D.objects[name]
    dim = obj.dimensions
    print('original dim:' + str(dim))
    if max(dim) > 0:
        dim = RESIZE_FACTOR * dim / max(dim)
    obj.dimensions = dim

    print('new dim:' + str(dim))


def move_camera(coord):
    def deg2rad(deg):
        return deg * math.pi / 180.

    r = 3.
    theta, phi = deg2rad(coord[0]), deg2rad(coord[1])
    loc_x = r * math.sin(theta) * math.cos(phi)
    loc_y = r * math.sin(theta) * math.sin(phi)
    loc_z = r * math.cos(theta)

    D.objects['Camera'].location = (loc_x, loc_y, loc_z)


def render():
    bpy.ops.render.render()


def save(image_dir, name):
    path = os.path.join(image_dir, name + '.png')
    D.images['Render Result'].save_render(filepath=path)
    print('save to ' + path)


if __name__ == '__main__':
    main()
