"""
Interactive visualization of kidney.

if event.text == '1':
    volume_visual.visible = not volume_visual.visible
    seg_visual.visible = not seg_visual.visible
elif event.text == '2':
    next_datafile()
elif event.text == '3':
    next_dataset()     
elif event.text == '4':
    swap_xy()
elif event.text == '5':
    swap_xz()
elif event.text == '6':
    mirror_x()
elif event.text == '7':
    mirror_y()
elif event.text == '8':
    image_visual.visible = not image_visual.visible
"""
from __future__ import print_function
from itertools import cycle

import numpy as np

from vispy import app, scene, io, visuals
from vispy.color import get_colormaps, BaseColormap, Colormap
import json

import nibabel
from vispy.visuals.transforms import (MatrixTransform, STTransform,
                                      arg_to_array, LogTransform, 
                                      PolarTransform, BaseTransform)
import cv2
import skimage.transform
import glob
import os
import h5py
import scipy.io

from MLFImage import load_mlf
from custom_image_visual import CustomImageVisual

#mlf_files = glob.glob("U:/Cupples/data/*.mlf")
markers_json = "./markers.json"

mlf_files = glob.glob("U:/Cupples/data/2014K26/*.mlf")
mat_files = glob.glob("U:/Cupples/data/2014K26/*.mat")
kidney_vol_file = "U:/Cupples/data/2014K26/2014K26_marker.img"
kidney_seg_file = "U:/Cupples/data/2014K26/2014K26_marker_fiducial.img"

canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(parent=view.scene, fov=60., name='Turntable')

frame_index = 0
animate = True


# create colormaps that work well for translucent and additive volume rendering
class TransFire(BaseColormap):
    glsl_map = """
    vec4 translucent_fire(float t) {
        return vec4(pow(t, 0.5), t, t*t, max(0, t*1.05 - 0.05));
    }
    """


class TransGrays(BaseColormap):
    glsl_map = """
    vec4 translucent_grays(float t) {
        return vec4(t, t, t, t*0.05);
    }
    """

# Setup colormap iterators
opaque_cmaps = cycle(get_colormaps())
translucent_cmaps = cycle([TransFire(), TransGrays()])
opaque_cmap = next(opaque_cmaps)
translucent_cmap = next(translucent_cmaps)

# Implement key presses
@canvas.events.key_press.connect
def on_key_press(event):
    global opaque_cmap, translucent_cmap, active_image
    if event.text == '1':
        volume_visual.visible = not volume_visual.visible
        seg_visual.visible = not seg_visual.visible
    elif event.text == '2':
        next_datafile()
    elif event.text == '3':
        next_dataset()     
    elif event.text == '4':
        swap_xy()
    elif event.text == '5':
        swap_xz()
    elif event.text == '6':
        mirror_x()
    elif event.text == '7':
        mirror_y()
    elif event.text == '8':
        image_visual.visible = not image_visual.visible

    print_active_info()

"""
Handles various formats.
A dictionary containing all 3d and 2d data in the .mat or .mlf file
"""
def load_lsci_source(filename):
    _, file_extension = os.path.splitext(filename)
    data = {}
    if 'mat' in file_extension:
        try:
            mat = h5py.File(filename, 'r')
        except Exception as e:
            mat = scipy.io.loadmat(filename)
        for k,v in mat.iteritems():
            try:
                conditions = [
                    v.ndim in [2,3],
                    v.min() != v.max(),
                    all(dim > 1 for dim in v.shape),
                    not np.isnan(v.max())
                ]
                if all(conditions):
                    data[k] = v
                else:
                    print("    %s rejected for conditions: %s" % (k, str(conditions)))                    
            except:
                pass

    elif 'mlf' in file_extension:
        mlf = load_mlf(filename)
        data['raw_mlf'] = mlf

    return data


def print_active_info():
    print("active file: %s" %  active_file)
    print("available datasets: %s" % active_dataset.keys())
    print("active dataset: %s" % active_data_key)
    print("data dimensions: %s" % str(active_image.shape)) 
    print("min, max: %s, %s" % (active_image.min(), active_image.max()))

def next_datafile():
    global active_dataset, data_keys, active_file
    active_file = data_files.next()
    active_dataset = load_lsci_source(active_file)
    data_keys = cycle(active_dataset.keys())
    next_dataset()

def next_dataset():
    global active_image, frame_index, active_data_key
    active_data_key = data_keys.next()
    active_image = active_dataset[active_data_key]
    if active_image.ndim == 2:
        image_visual.set_data(active_image)
    elif active_image.ndim == 3:
        frame_index = 0
        image_visual.set_data(active_image[frame_index])
    else:
        raise(Exception("incompatible dataset"))
    image_visual.clim = [active_image.min(), active_image.max()]
    view.camera.view_changed()

def swap_xz():
    global active_image, frame_index
    if active_image.ndim == 3:
        active_image = active_image.swapaxes(0,2)
        frame_index = 0
        set_visual_data(active_image[frame_index])
    else:
        print("Can't let you do that Star Fox.")


def set_visual_data(data):
    image_visual.set_data(data)
    view.camera.view_changed()

def swap_xy():
    global active_image
    active_image = active_image.swapaxes(0,1)
    set_visual_data(active_image)

def mirror_x():
    global active_image
    if active_image.ndim == 2:
        active_image = active_image[::-1,:]
    elif active_image.ndim == 3:
        active_image = active_image[::-1,:,:]
    set_visual_data(active_image)

def mirror_y():    
    global active_image
    if active_image.ndim == 2:
        active_image = active_image[:, ::-1]
    elif active_image.ndim == 3:
        active_image = active_image[:, ::-1,:]
    set_visual_data(active_image)

def toggle_animation():
    global animation_active
    animation_active = not animation_active

def edge_lengths(markers):
    points = np.vstack((markers, markers[0,:]))
    diff = (points[1:] - points[0:-1])
    return np.linalg.norm(diff,axis=1)

def get_centroids(vol, labels):
    points = []
    for label in labels:
        arr = np.array(np.where(vol==label))
        centroid = arr.sum(axis=1)/float(arr.shape[1])
        points.append(centroid)
    return np.array(points)

def animation_update(ev):
    global frame_index
    if active_image.ndim == 3 and animation_active:
        frame_index = (frame_index+1) % len(active_image)
        image_visual.set_data(active_image[:, :, frame_index])
        view.camera.view_changed()

def clean_source_files(files):
    clean_files = []
    for file in files:
        data = load_lsci_source(file)
        if data:
            clean_files.append(file)
    return clean_files

def show_files(header_string, files):
    print(header_string)
    for f in files:
        print(f)

if __name__ == '__main__':
    source_files = mat_files
    usable_source_files = clean_source_files(source_files)
    show_files("removed these files after clearing", [file for file in source_files if file not in usable_source_files])

    data_files = cycle(usable_source_files)
    active_file = data_files.next()
    active_dataset = load_lsci_source(active_file)

    data_keys = cycle(active_dataset.keys())
    active_data_key = data_keys.next()
    active_image = active_dataset[active_data_key]

    animation_active = True

    print("loading segmentation")
    seg_vol = np.array(nibabel.load(kidney_seg_file).get_data()).squeeze() #[::10,::10,::10]
    print("    segmentation shape: %s " % str(seg_vol.shape))
    

    print("loading volume")
    volume = np.array(nibabel.load(kidney_vol_file).get_data()).squeeze() #[::10,::10,::10]
    print("    volume shape: %s" % str(volume.shape))

    if os.path.exists(markers_json):
        markers = np.array(json.load(open(markers_json)))
    else:
        markers = get_centroids(vol=seg_vol, labels=[3,2,4,5])
        with open(markers_json, 'w') as outfile:
            json.dump(markers.tolist(), outfile)
    print(markers)

    # markers = np.array([
    #     [176,186,34],
    #     [200,685,43],
    #     [577,684,34],
    #     [485,241,43]
    # ])
    markers_zyx = markers[:,::-1]

    print(8.9*edge_lengths(markers))

    im_shape = np.array(active_image.shape)
    vertices = np.array([
        (0, 0, 1),
        (im_shape[0], 0, 1),
        (im_shape[0], im_shape[1], 1),
        (0, im_shape[1], 1)
    ])
    _, transform, _ = cv2.estimateAffine3D(vertices, markers, False)
    transform = np.vstack((transform, [0,0,0,1])).transpose()
    vertices_aug = np.hstack((vertices,np.array([[1],[1],[1],[1]])))
    rotated = np.dot(vertices_aug, transform)

    CustomIm = scene.visuals.create_visual_node(CustomImageVisual)
    image_visual = CustomIm(active_image, markers, parent=view.scene, cmap='grays')
    image_visual.clim = [active_image.min(), active_image.max()]

    volume_visual = scene.visuals.Volume(volume.swapaxes(0,2), parent=view.scene, threshold=0.225,
                                   emulate_texture=False)
    volume_visual.visible = False

    seg_visual = scene.visuals.Volume(seg_vol.swapaxes(0,2), parent=view.scene, threshold=0.225,
                                   emulate_texture=False)
    seg_visual.visible = True
    seg_visual.cmap = Colormap([
            (0., 0., 0., 1.),
            (1., 1., 1., 1.),
            (1., 0., 0., 1.),
            (0., 1., 0., 1.),
            (0., 0., 1., 1.),
            (0., 1., 1., 1.)
            ])

    timer = app.Timer(interval=0.02, connect=animation_update, start=True)

    app.run()