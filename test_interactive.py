# python test.py --dataroot ./inp/ --which_direction BtoA --model pix2pix --name facades_label2photo_pretrained --dataset_mode aligned --which_model_netG unet_256 --norm batch --loadSize=256

import time
import os
import glob
import shutil
import util
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
import numpy as np

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from PIL import Image

import copy
import _thread
from util.fit_boxes import fit_boxes, LabelClass, LabelFit
from util.fit_circles import fit_circles

import traceback

# class_names

# the order if the classes is their z order (first one is at the back)
cmp_classes = [
    LabelClass('other', [0, 0, 0], 0),  # black borders or sky (id 0)
    LabelClass('background', [0, 0, 170], 1),  # background (id 1)
    LabelClass('facade', [0, 0, 255], 2),  # facade (id 2)
    LabelClass('molding', [255, 85, 0], 3),  # molding (id 3)
    LabelClass('cornice', [0, 255, 255], 4),  # cornice (id 4)
    LabelClass('pillar', [255, 0, 0], 5),  # pillar (id 5)
    LabelClass('window', [0, 85, 255], 6),  # window (id 6)
    LabelClass('door', [0, 170, 255], 7),  # door (id 7)
    LabelClass('sill', [85, 255, 170], 8),  # sill (id 8)
    LabelClass('blind', [255, 255, 0], 9),  # blind (id 9)
    LabelClass('balcony', [170, 255, 85], 10),  # balcony (id 10)
    LabelClass('shop', [170, 0, 0], 11),  # shop (id 11)
    LabelClass('deco', [255, 170, 0], 12),  # deco (id 12)
]
fit_cmp_labels = {'window':LabelFit(-1), 'door':LabelFit(-1), 'sill':LabelFit(-1), 'balcony':LabelFit(-1), 'shop':LabelFit(-1)}
fit_cmp_labels_extended = {'window':LabelFit(-1), 'door':LabelFit(-1), 'sill':LabelFit(-1), 'balcony':LabelFit(-1), 'shop':LabelFit(-1), 'molding':LabelFit(-1), 'cornice':LabelFit(-1)}

roof_classes = [
    LabelClass('other', [0, 0, 0], 0),
    LabelClass('flat_roof', [255, 0, 0], 1),
    LabelClass('slanted_roof', [0, 255, 255], 2),
    LabelClass('edge', [255, 0, 255], 3),
    LabelClass('chimney', [255, 200, 0], 4),
    LabelClass('velux', [0, 0, 255], 5),
]
fit_roof_labels = {'velux':LabelFit(max_count=3), 'chimney':LabelFit(max_count=3)}

blank_classes = [
    LabelClass('other', [0,   0,   0], 0),
    LabelClass('wall', [0,   0, 255], 1),
    LabelClass('window', [0, 255,   0], 2),
]
fit_blank_labels = {'wall':LabelFit(-1), 'window':LabelFit(-1)}

pane_classes = [
    LabelClass('other', [0, 0, 0], 0),  # black borders or sky (id 0)
    LabelClass('frame', [255,   0,   0], 1),
    LabelClass('pane', [0,   0, 255], 2),
    LabelClass('object', [0, 255, 0], 3),
]
fit_pane_labels = {'frame':LabelFit(-1), 'pane':LabelFit(-1), 'object':LabelFit(-1)}

def save_image(image_numpy, image_path):

    try:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
    except:
        pass

    image_pil = Image.fromarray(image_numpy)

    image_pil.save(image_path, 'PNG', quality=100)

def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)

def rmrf (file):
    try:
        files = glob.glob(file)
        for f in files:
            if os.path.isfile(f):
                os.remove(f)
    except:
        pass

class RunG(FileSystemEventHandler):
    def __init__(self, model, opt, fit_boxes, fit_circles, directory):
        self.model = model
        self.opt = opt
        self.fit_boxes = fit_boxes
        self.fit_circles = fit_circles
        self.directory = directory

    def on_created(self, event): # when file is created
        # do something, eg. call your function to process the image
        print ("Got G event for file %s" % event.src_path)

        try:
            go = os.path.abspath(os.path.join (event.src_path, os.pardir, "go"))

            if not os.path.isfile(go):
                return

            with open(go) as f:
                name = f.readlines()[0]

            print("starting to process %s" % name)

            if self.opt.mlabel_condition:
                self.opt.mlabel_dataroot = self.opt.dataroot.rstrip('/\\')+'_mlabels'

            if self.opt.metrics_condition or self.opt.empty_condition:
                self.opt.empty_dataroot = self.opt.dataroot.rstrip('/\\')+'_empty'

            self.model.opt = self.opt

            data_loader = CreateDataLoader(self.opt)
            dataset = data_loader.load_data()


            for i, data in enumerate(dataset):
                # try:

                zs = os.path.basename(data['A_paths'][0])[:-4].split("_")[1:]
                z = np.array([float(i) for i in zs], dtype=np.float32)

                self.model.set_input(data)

                _, real_A, fake_B, real_B, _ = self.model.test_simple(z, encode_real_B=False)

                img_path = self.model.get_image_paths()
                print('%04d: process image... %s' % (i, img_path))

                save_image(fake_B, "./output/%s/%s/%s" % (self.directory, name, os.path.basename(img_path[0])))
                save_image(real_A, "./output/%s/%s/%s_label" % (self.directory, name, os.path.basename(img_path[0])))

                if self.fit_boxes is not None:
                    fit_boxes(
                        img=fake_B, classes=self.fit_boxes[0], fit_labels=self.fit_boxes[1],
                        json_path="./output/%s/%s/%s_boxes" % (self.directory, name, os.path.basename(img_path[0])))

                if self.fit_circles is not None:
                    fit_circles(
                        img=fake_B, classes=self.fit_circles[0], fit_labels=self.fit_circles[1],
                        json_path="./output/%s/%s/%s_circles" % (self.directory, name, os.path.basename(img_path[0])))

        except Exception as e:
            traceback.print_exc()
            print(e)

        try:
            rmrf('./input/%s/val/*' % self.directory)
            rmrf('./input/%s_empty/val/*' % self.directory)
            rmrf('./input/%s_mlabel/val/*' % self.directory)

            if os.path.isfile(go):
                os.remove(go)
        except Exception as e:
            traceback.print_exc()
            print(e)


class RunE(FileSystemEventHandler):

    def __init__(self, model, opt, directory):
        self.model = model
        self.opt = opt
        self.directory = directory

    def on_created(self, event):  # when file is created
        # do something, eg. call your function to process the image
        print("Got E event for file %s" % event.src_path)


        try:
            go = os.path.abspath(os.path.join(event.src_path, os.pardir, "go"))

            if not os.path.isfile(go):
                return

            with open(go) as f:
                name = f.readlines()[0]

            print("starting to process %s" % self.opt.name)

            self.model.opt = self.opt

            data_loader = CreateDataLoader(self.opt)
            dataset = data_loader.load_data()


            for i, data in enumerate(dataset):
                self.model.set_input(data)

                z = self.model.encode_real_B()

                img_path = self.model.get_image_paths()
                print('%04d: process image... %s' % (i, img_path))

                outfile = "./output/%s/%s/%s" % (self.directory, name, "_".join([str (s) for s in z[0]]) )
                try:
                    os.makedirs(os.path.dirname(outfile), exist_ok=True)
                except:
                    pass

                touch (outfile)

        except Exception as e:
            traceback.print_exc()
            print(e)

        try:
            rmrf('./input/%s_e/val/*' % self.directory)

            if os.path.isfile(go):
                os.remove(go)
        except Exception as e:
            traceback.print_exc()
            print(e)


class Interactive():
    def __init__(self, directory, name, size=256, which_model_netE='resnet_256', which_direction="BtoA",
                 fit_boxes=None, fit_circles=None, lbl_classes=None,
                 walldist_condition=False, imgpos_condition=False, noise_condition=False,
                 empty_condition=False, mlabel_condition=False, metrics_condition=False,
                 metrics_mask_color=None, norm='instance', nz=8, pytorch_v2=False, dataset_mode='aligned',
                 normalize_metrics=False, normalize_metrics2=False):

        # options
        optG = TestOptions().parse()
        optG.name = name
        optG.loadSize = size
        optG.fineSize = size
        optG.nThreads = 0 # min(1, optG.nThreads)  # test code only supports nThreads=1
        optG.batchSize = 1  # test code only supports batchSize=1
        optG.serial_batches = True  # no shuffle
        optG.which_model_netE = which_model_netE
        optG.which_direction = which_direction
        optG.pytorch_v2 = pytorch_v2

        optG.G_path = "./checkpoints/%s/latest_net_G.pth" % optG.name
        optG.E_path = "./checkpoints/%s/latest_net_E.pth" % optG.name

        optG.dataroot = "./input/%s/" % directory
        optG.no_flip = True
        optG.lbl_classes = lbl_classes
        optG.walldist_condition = walldist_condition
        optG.imgpos_condition = imgpos_condition
        optG.noise_condition = noise_condition
        optG.mlabel_condition = mlabel_condition
        optG.metrics_condition = metrics_condition
        optG.empty_condition = empty_condition
        optG.metrics_mask_color = metrics_mask_color
        optG.normalize_metrics = normalize_metrics
        optG.normalize_metrics2 = normalize_metrics2
        optG.norm = norm
        optG.nz = nz
        optG.dataset_mode = dataset_mode
        optG.use_dropout = False

        if optG.imgpos_condition:
            optG.input_nc += 2 # 2 image position x,y channels

        if optG.walldist_condition:
            optG.input_nc += 1 # 1 wall distance channel

        if optG.noise_condition:
            optG.input_nc += 1 # 1 wall noise channel

        if optG.mlabel_condition:
            optG.input_nc += 3 # 3 additional channels: RGB

        if optG.metrics_condition:
            optG.input_nc += 6 # 6 additional channels

        if optG.empty_condition:
            optG.input_nc += 3 # 3 additional channels: RGB

        optE = copy.deepcopy(optG)
        optE.dataroot = "./input/%s_e/" % directory
        optE.name = optG.name + "_e"

        optE.walldist_condition = False
        optE.imgpos_condition = False
        optE.noise_condition = False
        optE.mlabel_condition = False
        optE.metrics_condition = False
        optE.empty_condition = False

        model = create_model(optG)
        model.eval()

        self.optG = optG
        self.optE = optE
        self.model = model

        _thread.start_new_thread(self.go, (directory, name, size, fit_boxes, fit_circles))

    def go(self, directory, name, size, fit_boxes, fit_circles):

        observer = Observer()

        input_folder = './input/%s/' % directory
        os.makedirs(input_folder + "val", exist_ok=True)
        observer.schedule(
            RunG(model=self.model, opt=self.optG, fit_boxes=fit_boxes, fit_circles=fit_circles, directory=directory),
            path=input_folder+"val/")

        input_folder_e = './input/%s_e/' % directory
        os.makedirs(input_folder_e+"val", exist_ok=True)
        observer.schedule(
            RunE(self.model, self.optE, directory+"_e"),
            path=input_folder_e+"val/")
        observer.start()

        print('[network %s is awaiting input]' % name)

        # sleep until keyboard interrupt, then stop + rejoin the observer
        # try:
        #     while True:
        #         time.sleep(1)
        # except KeyboardInterrupt:
        #     observer.stop()

        observer.join()

# #------------------------------------------#
# # original set:
# #------------------------------------------#
# # Interactive ("bike_2", pytorch_v2 = True)
# Interactive ( "roof", "roofs6", 512, 'resnet_512', pytorch_v2 = True)
# Interactive ( "facade super", "super6", pytorch_v2 = True) # walls
# # Interactive ("", ""super9", pytorch_v2 = True) # facades
# Interactive ( "pane labels","dows2", pytorch_v2 = True)
# Interactive ( "pane textures", "dows1", pytorch_v2 = True)
# # Interactive ("blank", fit_boxes=blank_classes )
# Interactive ( "facade labels", "empty2windows_f005", lbl_classes=cmp_classes, imgpos_condition=True, walldist_condition=True, norm='instance_track', fit_boxes=blank_classes, dataset_mode='multi')
# # Interactive ("empty2windows_f005", lbl_classes=cmp_classes, imgpos_condition=True, walldist_condition=True, norm='instance_track', fit_boxes=blank_classes, dataset_mode='multi')
# Interactive ( "facade textures", "facade_windows_f000", norm='instance_track', fit_boxes=blank_classes, dataset_mode='multi')
# # Interactive ("image2clabels_f001", norm='instance_track', fit_boxes=blank_classes, nz=0, dataset_mode='multi')
# #------------------------------------------#


# #------------------------------------------#
# # original set with new facade texture -> greebles network:
# #------------------------------------------#
# Interactive("roof", "roofs6", 512, 'resnet_512', pytorch_v2 = True)
# Interactive("facade super", "super6", pytorch_v2 = True) # walls
# Interactive("pane labels","dows2", pytorch_v2 = True)
# Interactive("pane textures", "dows1", pytorch_v2 = True)
# Interactive("facade labels", "empty2windows_f005", lbl_classes=cmp_classes, imgpos_condition=True, walldist_condition=True, norm='instance_track', fit_boxes=blank_classes, dataset_mode='multi')
# Interactive("facade textures", "facade_windows_f000", norm='instance_track', fit_boxes=blank_classes, dataset_mode='multi')
# Interactive("facade greeble labels", "image2clabels_f005_200",
#             dataset_mode='multi', fit_boxes=cmp_greeble_classes,
#             empty_condition=True, mlabel_condition=True, metrics_condition=True,
#             metrics_mask_color=[0, 0, 255], nz=0)
# #------------------------------------------#


#------------------------------------------#
# latest set: (29 May):
#------------------------------------------#
Interactive("roof greebles", "r3_clabels2labels_f001_400",
            size=512, which_model_netE='resnet_512',
            dataset_mode='multi', fit_circles=(roof_classes, fit_roof_labels),
            empty_condition=True, metrics_condition=True, imgpos_condition=True,
            noise_condition=True,
            metrics_mask_color=[0, 0, 255], normalize_metrics=True)

Interactive("roof", "r3_labels2image_f001_400",
            size=512, which_model_netE='resnet_512',
            dataset_mode='multi',
            empty_condition=True, metrics_condition=True, imgpos_condition=True,
            metrics_mask_color=[0, 0, 255], normalize_metrics=True)

Interactive("pane labels", "w3_empty2labels_f009_200",
            dataset_mode='multi', fit_boxes=(pane_classes, fit_pane_labels),
            empty_condition=True, metrics_condition=True, imgpos_condition=True,
            metrics_mask_color=[255, 0, 0])

Interactive("pane textures", "w3_labels2image_f013_400",
            dataset_mode='multi',
            empty_condition=True, metrics_condition=True, imgpos_condition=True,
            metrics_mask_color=[255, 0, 0])

Interactive("facade labels", "empty2windows_f009v2_400",
            dataset_mode='multi', fit_boxes=(blank_classes, fit_blank_labels),
            empty_condition=True, metrics_condition=True, imgpos_condition=True,
            metrics_mask_color=[0, 0, 255])

Interactive("facade textures", "facade_windows_f013v2_150",
            dataset_mode='multi',
            empty_condition=True, metrics_condition=True, imgpos_condition=True,
            metrics_mask_color=[0, 0, 255])

Interactive("facade greebles", "image2clabels_f005_200",
            dataset_mode='multi', fit_boxes=(cmp_classes, fit_cmp_labels),
            empty_condition=True, metrics_condition=True, mlabel_condition=True,
            metrics_mask_color=[0, 0, 255], nz=0)

# Interactive("facade greebles", "image2celabels_f001_335",
#             dataset_mode='multi', fit_boxes=(cmp_classes, fit_cmp_labels),
#             empty_condition=True, metrics_condition=True, mlabel_condition=True,
#             metrics_mask_color=[0, 0, 255], nz=0)

Interactive("facade super", "super6", pytorch_v2=True)

Interactive("roof super", "super6", pytorch_v2=True)

#------------------------------------------#

print("all nets up")

while True:
    time.sleep(600)
