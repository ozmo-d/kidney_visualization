"""
Interactive visualization of kidney.

Click/drag mouse to rotate view.
Shift + drag to move view linearly.
Mouse wheel to zoom.

Keys control the images
'1': toggle segmentation / mip
'2': next fiducial data file
'3': next data source within fiducial data file
'4': swap fiducial data xy
'5': swap fiducial data yz
'6': mirror x
'7': mirror y
'8': toggle fiducial image visibility

'h': show this help

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
import glob
import os
import h5py
import scipy.io

from MLFImage import load_mlf
from custom_image_visual import CustomImageVisual

import kidney_data


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

class FiducialViewer(object):
    def __init__(self, markers_json, lsci_source_files, kidney_vol_file, kidney_seg_file):
        self.markers_json = markers_json

        usable_source_files = self.clean_source_files(lsci_source_files)
        self.show_files("removed these files after clearing", [file for file in lsci_source_files if file not in usable_source_files])

        self.data_files = cycle(usable_source_files)
        self.active_file = next(self.data_files)
        self.active_dataset = self.load_lsci_source(self.active_file)

        self.data_keys = cycle(self.active_dataset.keys())
        self.active_data_key = next(self.data_keys)
        self.active_image = self.active_dataset[self.active_data_key]

        self.animation_active = True


        self.canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
        self.canvas.events.key_press.connect(self.on_key_press)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(parent=self.view.scene, fov=60., name='Turntable')

        self.frame_index = 0
        self.animate = True        

        print("loading segmentation")
        seg_vol = np.array(nibabel.load(kidney_seg_file).get_data()).squeeze() #[::10,::10,::10]
        print("    segmentation shape: %s " % str(seg_vol.shape))
        
        print("loading volume")
        volume = np.array(nibabel.load(kidney_vol_file).get_data()).squeeze() #[::10,::10,::10]
        print("    volume shape: %s" % str(volume.shape))

        if os.path.exists(self.markers_json):
            markers = np.array(json.load(open(self.markers_json)))
        else:
            markers = self.get_centroids(vol=seg_vol, labels=[3,2,4,5])
            with open(self.markers_json, 'w') as outfile:
                json.dump(markers.tolist(), outfile)
        print(markers)

        CustomIm = scene.visuals.create_visual_node(CustomImageVisual)
        self.image_visual = CustomIm(self.active_image, markers, parent=self.view.scene, cmap='grays')
        self.image_visual.clim = [self.active_image.min(), self.active_image.max()]

        self.volume_visual = scene.visuals.Volume(volume.swapaxes(0,2), parent=self.view.scene, threshold=0.225,
                                    emulate_texture=False)
        self.volume_visual.visible = False

        self.seg_visual = scene.visuals.Volume(seg_vol.swapaxes(0,2), parent=self.view.scene, threshold=0.225,
                                    emulate_texture=False)
        self.seg_visual.visible = True
        self.seg_visual.cmap = Colormap([
                (0., 0., 0., 1.),
                (1., 1., 1., 1.),
                (1., 0., 0., 1.),
                (0., 1., 0., 1.),
                (0., 0., 1., 1.),
                (0., 1., 1., 1.)
                ])

        print(__doc__)

        self.timer = app.Timer(interval=0.02, connect=self.animation_update, start=True)

        app.run()

    def on_key_press(self, event):
        if event.text == '1':
            self.volume_visual.visible = not self.volume_visual.visible
            self.seg_visual.visible = not self.seg_visual.visible
        elif event.text == '2':
            self.next_datafile()
        elif event.text == '3':
            self.next_dataset()     
        elif event.text == '4':
            self.swap_xy()
        elif event.text == '5':
            self.swap_xz()
        elif event.text == '6':
            self.mirror_x()
        elif event.text == '7':
            self.mirror_y()
        elif event.text == '8':
            self.image_visual.visible = not self.image_visual.visible
        elif event.text == 'h':
            print(__doc__)

        if event.text in ['2', '3', '4','5']:
            self.print_active_info()


    """
    Handles various formats.
    A dictionary containing all 3d and 2d data in the .mat or .mlf file
    """
    def load_lsci_source(self, filename):
        _, file_extension = os.path.splitext(filename)
        data = {}
        if 'mat' in file_extension:
            try:
                mat = h5py.File(filename, 'r')
            except Exception as e:
                mat = scipy.io.loadmat(filename)
            for k,v in mat.items():
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
                        #print("    %s rejected for conditions: %s" % (k, str(conditions)))
                        pass
                except:
                    pass

        elif 'mlf' in file_extension:
            mlf = load_mlf(filename)
            data['raw_mlf'] = mlf

        return data


    def print_active_info(self):
        print("active file: %s" %  self.active_file)
        print("available datasets: %s" % self.active_dataset.keys())
        print("active dataset: %s" % self.active_data_key)
        print("data dimensions: %s" % str(self.active_image.shape)) 
        print("min, max: %s, %s" % (self.active_image.min(), self.active_image.max()))

    def next_datafile(self):
        self.active_file = next(self.data_files)
        self.active_dataset = self.load_lsci_source(self.active_file)
        self.data_keys = cycle(self.active_dataset.keys())
        self.next_dataset()

    def next_dataset(self):
        self.active_data_key = next(self.data_keys)
        self.active_image = self.active_dataset[self.active_data_key]
        if self.active_image.ndim == 2:
            self.image_visual.set_data(self.active_image)
        elif self.active_image.ndim == 3:
            self.frame_index = 0
            self.image_visual.set_data(self.active_image[self.frame_index])
        else:
            raise(Exception("incompatible dataset"))
        self.image_visual.clim = [self.active_image.min(), self.active_image.max()]
        self.view.camera.view_changed()

    def animation_update(self, ev):
        if self.active_image.ndim == 3 and self.animation_active:
            self.frame_index = (self.frame_index+1) % len(self.active_image)
            self.set_visual_data(self.active_image[:, :, self.frame_index])

    def swap_xz(self):
        if self.active_image.ndim == 3:
            self.active_image = self.active_image.swapaxes(0,2)
            self.frame_index = 0
            self.set_visual_data(self.active_image[self.frame_index])
        else:
            print("Can't let you do that Star Fox.")

    def set_visual_data(self, data):
        self.image_visual.set_data(data)
        self.view.camera.view_changed()

    def swap_xy(self):
        self.active_image = self.active_image.swapaxes(0,1)
        self.set_visual_data(self.active_image)

    def mirror_x(self):
        if self.active_image.ndim == 2:
            self.active_image = self.active_image[::-1,:]
        elif self.active_image.ndim == 3:
            self.active_image = self.active_image[::-1,:,:]
        self.set_visual_data(self.active_image)

    def mirror_y(self):    
        if self.active_image.ndim == 2:
            self.active_image = self.active_image[:, ::-1]
        elif self.active_image.ndim == 3:
            self.active_image = self.active_image[:, ::-1,:]
        self.set_visual_data(self.active_image)

    def toggle_animation(self):
        self.animation_active = not self.animation_active

    def edge_lengths(self, markers):
        points = np.vstack((markers, markers[0,:]))
        diff = (points[1:] - points[0:-1])
        return np.linalg.norm(diff,axis=1)

    def get_centroids(self, vol, labels):
        points = []
        for label in labels:
            arr = np.array(np.where(vol==label))
            centroid = arr.sum(axis=1)/float(arr.shape[1])
            points.append(centroid)
        return np.array(points)

    def clean_source_files(self, files):
        clean_files = []
        for file in files:
            data = self.load_lsci_source(file)
            if data:
                clean_files.append(file)
        return clean_files

    def show_files(self, header_string, files):
        print(header_string)
        for f in files:
            print(f)


def fiducial_visualization(dataset):
    assert(dataset in kidney_data.get_datasets())
    markers_json = "./%s_markers.json" % dataset
    FiducialViewer(markers_json, kidney_data.get_lsci_files(dataset), kidney_data.get_kidney_vol(dataset), kidney_data.get_kidney_seg(dataset))

if __name__ == '__main__':
    dataset = "2014K26"
    fiducial_visualization(dataset)