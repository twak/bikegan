import os
import numpy as np
import cv2
import torch
import util.util as util
from torch.autograd import Variable
from . import networks


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        # create image position channel
        if self.opt.imgpos_condition:
            real_batchSize = self.opt.batchSize // 2 if self.opt.isTrain else self.opt.batchSize
            self.imgpos = torch.stack([
                torch.linspace(0, 1, self.opt.fineSize).view(-1, 1).repeat(1, self.opt.fineSize),
                torch.linspace(0, 1, self.opt.fineSize).view(1, -1).repeat(self.opt.fineSize, 1)]).expand(real_batchSize, -1, -1, -1)
            if len(self.opt.gpu_ids) > 0 and self.opt.gpu_ids[0] >= 0:
                self.imgpos = self.imgpos.cuda(self.opt.gpu_ids[0])

        # create wall mask (to find walls in the label image)
        if self.opt.walldist_condition:
            wall_color = [c.color for c in opt.lbl_classes if c.name == 'facade']
            if len(wall_color) != 1:
                raise ValueError('There should be a single label \'facade\' in the list of labels.')
            wall_color = wall_color[0]
            self.mask_wall = torch.FloatTensor(wall_color).view(1, 3, 1, 1) * 2 - 1
            if len(self.opt.gpu_ids) > 0 and self.opt.gpu_ids[0] >= 0:
                self.mask_wall = self.mask_wall.cuda(self.opt.gpu_ids[0])

    def init_data(self, opt, use_D=True, use_D2=True, use_E=True, use_vae=True):
        print('---------- Networks initialized -------------')
        # load/define networks: define G
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.nz, opt.ngf,
                                      which_model_netG=opt.which_model_netG,
                                      norm=opt.norm, nl=opt.nl, use_dropout=opt.use_dropout, init_type=opt.init_type,
                                      gpu_ids=self.gpu_ids, where_add=self.opt.where_add, upsample=opt.upsample)

        # networks.print_network(self.netG)
        self.netD, self.netD2, self.netE = None, None, None

        use_sigmoid = opt.gan_mode == 'dcgan'
        D_output_nc = opt.input_nc + opt.output_nc if self.opt.conditional_D else opt.output_nc
        # define D
        if not opt.isTrain:
            use_D = False
            use_D2 = False

        if use_D:
            self.netD = networks.define_D(D_output_nc, opt.ndf,
                                          which_model_netD=opt.which_model_netD,
                                          norm=opt.norm, nl=opt.nl,
                                          use_sigmoid=use_sigmoid, init_type=opt.init_type, num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)
            # networks.print_network(self.netD)
        if use_D2:
            self.netD2 = networks.define_D(D_output_nc, opt.ndf,
                                           which_model_netD=opt.which_model_netD2,
                                           norm=opt.norm, nl=opt.nl,
                                           use_sigmoid=use_sigmoid, init_type=opt.init_type, num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)
            # networks.print_network(self.netD2)

        # define E
        if use_E:
            self.netE = networks.define_E(opt.output_nc, opt.nz, opt.nef,
                                          which_model_netE=opt.which_model_netE,
                                          norm=opt.norm, nl=opt.nl,
                                          init_type=opt.init_type, gpu_ids=self.gpu_ids,
                                          vaeLike=use_vae)
            # networks.print_network(self.netE)

        if not opt.isTrain:
            self.load_network_test(self.netG, opt.G_path, opt.pytorch_v2)

            if use_E:
                self.load_network_test(self.netE, opt.E_path, opt.pytorch_v2)

        if opt.isTrain and opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch, opt.pytorch_v2)

            if use_D:
                self.load_network(self.netD, 'D', opt.which_epoch, opt.pytorch_v2)
            if use_D2:
                self.load_network(self.netD, 'D2', opt.which_epoch, opt.pytorch_v2)

            if use_E:
                self.load_network(self.netE, 'E', opt.which_epoch, opt.pytorch_v2)
        print('-----------------------------------------------')

        # define loss functions
        self.criterionGAN = networks.GANLoss(
            mse_loss=not use_sigmoid, tensor=self.Tensor)
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionL2 = torch.nn.MSELoss()
        self.criterionZ = torch.nn.L1Loss()

        if opt.isTrain:
            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            if use_E:
                self.optimizer_E = torch.optim.Adam(
                    self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_E)

            if use_D:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)
            if use_D2:
                self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(),
                                                     lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D2)

            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

    def is_skip(self):
        return False

    def forward(self):
        pass

    def eval(self):
        pass

    def set_requires_grad(self, net, requires_grad=False):
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad  # to avoid computation

    def balance(self):
        pass

    def update_D(self, data):
        pass

    def update_G(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, pytorch2=False):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        dict = torch.load(save_path)
        if pytorch2:
            self.importPytorch2(dict)
        network.load_state_dict()

    def load_network_test(self, network, network_path, pytorch2=False):
        dict = torch.load(network_path)
        if pytorch2:
            self.importPytorch2(dict)
        network.load_state_dict(dict)

    def importPytorch2(self, foo):
        for k in list(foo.keys()):
            if "running" in k:
                del foo[k]

    def update_learning_rate(self):
        loss = self.get_measurement()
        for scheduler in self.schedulers:
            scheduler.step(loss)
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_measurement(self):
        return None

    def get_z_random(self, batchSize, nz, random_type='gauss'):
        z = self.Tensor(batchSize, nz)
        if random_type == 'uni':
            z.copy_(torch.rand(batchSize, nz) * 2.0 - 1.0)
        elif random_type == 'gauss':
            z.copy_(torch.randn(batchSize, nz))
        z = Variable(z)
        return z

    def compute_walldist(self, x):
        mask = ((x - self.mask_wall).abs_().sum(dim=1, keepdim=True) <= 0.001)
        mask[:, :, [0, -1], :] = 0 # remove the image border, assume the wall stops at the image border
        mask[:, :, :, [0, -1]] = 0 # remove the image border, assume the wall stops at the image border ()
        mask = mask.cpu().numpy().transpose(0, 2, 3, 1)
        walldist = np.zeros_like(mask, dtype='float32')
        for i in range(mask.shape[0]):
            img_mask = mask[i, :, :, :]
            if img_mask.all():
                raise ValueError('No wall boundary detected, cannot compute distance transform.')
            walldist[i, :, :, 0] = cv2.distanceTransform(img_mask, cv2.DIST_L2, cv2.DIST_MASK_3)

        walldist = torch.from_numpy(walldist.transpose(0, 3, 1, 2))

        if len(self.gpu_ids) > 0:
            walldist = walldist.cuda(self.gpu_ids[0])

        return walldist

    # testing models
    def set_input(self, input):
        # get direciton
        AtoB = self.opt.which_direction == 'AtoB'
        # set input images
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
        self.input_A = input_A
        self.input_B = input_B

        # get image paths
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        if self.opt.walldist_condition:
            self.input_walldist = self.compute_walldist(self.input_A)
        if self.opt.mlabel_condition:
            self.input_mlabel = input['mlabel']
            if len(self.gpu_ids) > 0:
                self.input_mlabel = self.input_mlabel.cuda(self.gpu_ids[0])
        if self.opt.noise_condition:
            self.input_noise = torch.randn(self.input_A.size(0), 1, self.input_A.size(2), self.input_A.size(3))
            if len(self.gpu_ids) > 0:
                self.input_noise = self.input_noise.cuda(self.gpu_ids[0])
        if self.opt.metrics_condition:
            self.input_facade_metrics = input['metrics']
            if len(self.gpu_ids) > 0:
                self.input_facade_metrics = self.input_facade_metrics.cuda(self.gpu_ids[0])
        if self.opt.empty_condition:
            self.input_empty_facade = input['empty']
            if len(self.gpu_ids) > 0:
                self.input_empty_facade = self.input_empty_facade.cuda(self.gpu_ids[0])

    def get_image_paths(self):
        return self.image_paths

    def test(self, z_sample):  # need to have input set already
        with torch.no_grad():
            self.real_A = self.input_A
            batchSize = self.input_A.size(0)

            self.G_input = self.real_A
            if self.opt.imgpos_condition:
                self.G_input = torch.cat([self.imgpos, self.G_input], dim=1)
            if self.opt.walldist_condition:
                self.G_input = torch.cat([self.input_walldist, self.G_input], dim=1)
            if self.opt.mlabel_condition:
                self.G_input = torch.cat([self.input_mlabel, self.G_input], dim=1)
            if self.opt.noise_condition:
                self.G_input = torch.cat([self.input_noise, self.G_input], dim=1)
            if self.opt.metrics_condition:
                self.G_input = torch.cat([self.input_facade_metrics, self.G_input], dim=1)
            if self.opt.empty_condition:
                self.G_input = torch.cat([self.input_empty_facade, self.G_input], dim=1)

            if self.opt.nz > 0:
                self.z = self.Tensor(batchSize, self.opt.nz)
                z_torch = torch.from_numpy(z_sample)
                self.z.copy_(z_torch)
            else:
                self.z = None

            self.fake_B = self.netG.forward(self.G_input, self.z)
            self.real_B = self.input_B

    def encode(self, input_data):
        return self.netE.forward(Variable(input_data, volatile=True))

    def encode_real_B(self):
        # self.z_encoded = self.encode(self.input_B)
        self.z_encoded, _ = self.netE.forward(self.input_B)
        return util.tensor2vec(self.z_encoded)

    def real_data(self, input=None):
        if input is not None:
            self.set_input(input)
        return util.tensor2im(self.input_A), util.tensor2im(self.input_B)

    def test_simple(self, z_sample=None, input=None, encode_real_B=False):
        if input is not None:
            self.set_input(input)

        if self.opt.nz > 0 and encode_real_B:  # use encoded z
            z_sample = self.encode_real_B()

        self.test(z_sample)

        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return self.image_paths, real_A, fake_B, real_B, z_sample
