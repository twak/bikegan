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
from util.fit_boxes import fit_boxes, LabelClass

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

blank_classes = [
    LabelClass('other' , [0,   0,   0], 0),
    LabelClass('wall'  , [0,   0, 255], 0),
    LabelClass('window', [0, 255,   0], 0),
]

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
    files = glob.glob(file)
    for f in files:
        if os.path.isfile(f):
            os.remove(f)

class RunG(FileSystemEventHandler):
    def __init__(self, model, opt, fit_boxes=None):
        self.model = model
        self.opt = opt
        self.fit_boxes = fit_boxes

    def on_created(self, event): # when file is created
        # do something, eg. call your function to process the image
        print ("Got G event for file %s" % event.src_path)

        go = os.path.abspath( os.path.join (event.src_path, os.pardir, "go") )
        
        if not os.path.isfile( go ):
            return

        with open (go) as f:
            name = f.readlines()[0]
        
        print("starting to process %s" % name)

        data_loader = CreateDataLoader(self.opt)
        dataset = data_loader.load_data()

        for i, data in enumerate(dataset):
            try:

                zs = os.path.basename ( data['A_paths'][0] )[:-4].split("_") [1:]
                z = np.array ( [float(i) for i in zs], dtype = np.float32 )

                self.model.set_input(data)

                _, real_A, fake_B, real_B, _ = self.model.test_simple( z, encode_real_B=False)

                img_path = self.model.get_image_paths()
                print('%04d: process image... %s' % (i, img_path))

                save_image( fake_B, "./output/%s/%s/%s" % (self.opt.name,name, os.path.basename(img_path[0]) ) )
                save_image(real_A, "./output/%s/%s/%s_label" % (self.opt.name, name, os.path.basename(img_path[0])))

                if self.fit_boxes is not None:
                    fit_boxes(fake_B, self.fit_boxes, "./output/%s/%s/%s_boxes" % (self.opt.name, name, os.path.basename(img_path[0])))

            except Exception as e:
                print(e)

        os.remove(go)

        rmrf('./input/%s/val/*' % self.opt.name)


class RunE(FileSystemEventHandler):

    def __init__(self, model, opt):
        self.model = model
        self.opt = opt

    def on_created(self, event):  # when file is created
        # do something, eg. call your function to process the image
        print("Got E event for file %s" % event.src_path)

        go = os.path.abspath(os.path.join(event.src_path, os.pardir, "go"))

        if not os.path.isfile(go):
            return

        with open(go) as f:
            name = f.readlines()[0]

        print("starting to process %s" % self.opt.name)

        self.opt.dataroot = "./input/%s/" % self.opt.name

        data_loader = CreateDataLoader(self.opt)
        dataset = data_loader.load_data()

        for i, data in enumerate(dataset):
            self.model.set_input(data)

            z = self.model.encode_real_B()

            img_path = self.model.get_image_paths()
            print('%04d: process image... %s' % (i, img_path))

            outfile = "./output/%s/%s/%s" % (self.opt.name, name, "_".join([str (s) for s in z[0]]) )
            try:
                os.makedirs(os.path.dirname(outfile), exist_ok=True)
            except:
                pass

            touch (outfile)

        os.remove(go)

        rmrf('./input/%s/val/*' % self.opt.name)


class Interactive():
    def __init__(self, name, size=256, which_model_netE='resnet_256', which_direction="BtoA", fit_boxes = None):

        # options
        optG = TestOptions().parse()
        optG.name = name
        optG.loadSize = size
        optG.fineSize = size
        optG.nThreads = 1  # test code only supports nThreads=1
        optG.batchSize = 1  # test code only supports batchSize=1
        optG.serial_batches = True  # no shuffle
        optG.which_model_netE = which_model_netE
        optG.which_direction = which_direction

        optG.G_path = "./checkpoints/%s/latest_net_G.pth" % optG.name
        optG.E_path = "./checkpoints/%s/latest_net_E.pth" % optG.name

        optG.dataroot = "./input/%s/" % optG.name
        optG.no_flip = True

        optE = copy.deepcopy(optG)
        optE.dataroot = "./input/%s_e/" % optG.name
        optE.name = optG.name + "_e"

        model = create_model(optG)
        model.eval()

        self.optG = optG
        self.optE = optE
        self.model = model

        _thread.start_new_thread (self.go, (name, size, fit_boxes ) )

    def go (self, name, size, fit_boxes):

        observer = Observer()

        input_folder = './input/%s/' % self.optG.name
        os.makedirs(input_folder + "val", exist_ok=True)
        observer.schedule(RunG(self.model, self.optG, fit_boxes), path=input_folder+"val/")

        input_folder_e = './input/%s/' % self.optE.name
        os.makedirs(input_folder_e+"val", exist_ok=True)
        observer.schedule(RunE(self.model, self.optE), path=input_folder_e+"val/")
        observer.start()

        # sleep until keyboard interrupt, then stop + rejoin the observer
        # try:
        #     while True:
        #         time.sleep(1)
        # except KeyboardInterrupt:
        #     observer.stop()

        observer.join()

Interactive ("bike_2")
# Interactive ("roofs2", 512, 'resnet_512')
Interactive ("roofs4", 512, 'resnet_512')
Interactive ("super4")
Interactive ("dows2")
Interactive ("dows1")
Interactive ("blank", fit_boxes=blank_classes )

while True:
    time.sleep(600)
