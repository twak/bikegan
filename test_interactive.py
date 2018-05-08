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

# options
opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads=1
opt.batchSize = 1   # test code only supports batchSize=1
opt.serial_batches = True  # no shuffle

opt.G_path = "./checkpoints/%s/latest_net_G.pth" % opt.name
opt.E_path = "./checkpoints/%s/latest_net_E.pth" % opt.name

opt.dataroot = "./input/%s/" % opt.name
opt.no_flip = True

optE = copy.deepcopy(opt)
optE.dataroot = "./input/%s_e/" % opt.name
optE.name = opt.name +"_e"

model = create_model(opt)
model.eval()

def save_image(image_numpy, image_path):

    try:
        os.mkdir(os.path.dirname(image_path))
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
    def on_created(self, event): # when file is created
        # do something, eg. call your function to process the image
        print ("Got G event for file %s" % event.src_path)

        go = os.path.abspath( os.path.join (event.src_path, os.pardir, "go") )
        
        if not os.path.isfile( go ):
            return

        with open (go) as f:
            name = f.readlines()[0]
        
        print("starting to process %s" % name)

        zs = name.split("_")[2:]
        z = np.array ( [float(i) for i in zs], dtype = np.float32 )

        data_loader = CreateDataLoader(opt)
        dataset = data_loader.load_data()

        for i, data in enumerate(dataset):
            model.set_input(data)

            _, real_A, fake_B, real_B, _ = model.test_simple( z, encode_real_B=False)

            img_path = model.get_image_paths()
            print('%04d: process image... %s' % (i, img_path))

            save_image( fake_B, "./output/%s/%s/%s" % (opt.name,name, os.path.basename(img_path[0]) ) )
            save_image( real_A, "./output/%s/%s/%s_label" % (opt.name,name, os.path.basename(img_path[0]) ) )

        os.remove(go)

        rmrf('./input/%s/val/*' % opt.name)


class RunE(FileSystemEventHandler):
    def on_created(self, event):  # when file is created
        # do something, eg. call your function to process the image
        print("Got E event for file %s" % event.src_path)

        go = os.path.abspath(os.path.join(event.src_path, os.pardir, "go"))

        if not os.path.isfile(go):
            return

        with open(go) as f:
            name = f.readlines()[0]

        print("starting to process %s" % optE.name)

        optE.dataroot = "./input/%s/" % optE.name

        data_loader = CreateDataLoader(optE)
        dataset = data_loader.load_data()

        for i, data in enumerate(dataset):
            model.set_input(data)

            z = model.encode_real_B()

            img_path = model.get_image_paths()
            print('%04d: process image... %s' % (i, img_path))

            outfile = "./output/%s/%s/%s" % (optE.name, name, "_".join([str (s) for s in z[0]]) )
            try:
                os.makedirs(os.path.dirname(outfile), exist_ok=True)
            except:
                pass

            touch (outfile)

        os.remove(go)

        rmrf('./input/%s/val/*' % optE.name)

observer = Observer()

input_folder = './input/%s/' % opt.name
os.makedirs(input_folder + "val", exist_ok=True)
observer.schedule(RunG(), path=input_folder+"val/")

input_folder_e = './input/%s_e/' % opt.name
os.makedirs(input_folder_e+"val", exist_ok=True)
observer.schedule(RunE(), path=input_folder_e+"val/")
observer.start()

# sleep until keyboard interrupt, then stop + rejoin the observer
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()

observer.join()

