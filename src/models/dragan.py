import torch, time, os, pickle
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.autograd import grad
from ..data.dataset import dataloader
from .. import  utils
from .layers.snconv2d import SNConv2d
    
class Generator(nn.Module):
    '''
    Generator Class
    Values:
        input_dim: the dimension of the noise vector, a scalar
        output_dim: the number of channels in the images, fitted for the dataset used, a scalar
              (in this case it is 3, since the image is in RGB)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, input_dim=100, output_dim=3, hidden_dim=32):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.z_dim = input_dim
        self.gen = nn.Sequential(
            self.make_gen_block(input_dim, hidden_dim * 8, 4, 1, 0),
            self.make_gen_block(hidden_dim * 8, hidden_dim * 4, 4, 2, 1),
            self.make_gen_block(hidden_dim * 4, hidden_dim*2, 4, 2, 1),
            self.make_gen_block(hidden_dim * 2, hidden_dim, 4, 2, 1),
            self.make_gen_block(hidden_dim, output_dim, kernel_size=4, stride=2, padding=1, final_layer=True),
        )

        utils.initialize_weights(self)

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, padding=1, final_layer=False):
            '''
            Function to return a sequence of operations corresponding to a generator block of DCGAN, 
            corresponding to a transposed convolution, a batchnorm (except for in the last layer), and an activation.
            Parameters:
                input_channels: how many channels the input feature representation has
                output_channels: how many channels the output feature representation should have
                kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
                stride: the stride of the convolution
                final_layer: a boolean, true if it is the final layer and false otherwise 
                        (affects activation and batchnorm)
            '''

            if not final_layer:
                return nn.Sequential(
                    nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True)
                )
            else: # Final Layer
                return nn.Sequential(
                    nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                    nn.Tanh()
                )
                                                                              
    def unsqueeze_noise(self, noise):
            '''
            Function for completing a forward pass of the generator: Given a noise tensor, 
            returns a copy of that noise with width and height = 1 and channels = z_dim.
            Parameters:
                noise: a noise tensor with dimensions (n_samples, z_dim)
            '''
            return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        x = self.unsqueeze_noise(noise)
        return self.gen(x)

class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
        input_dim: the number of channels in the images, fitted for the dataset used, a scalar
              (in this case it is 3, since the image is in RGB)
    hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, input_dim=1, hidden_dim=32):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.disc = nn.Sequential(
            self.make_disc_block(input_dim, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim*2),
            self.make_disc_block(hidden_dim*2, hidden_dim*4),
            self.make_disc_block(hidden_dim*4, hidden_dim*8),
            self.make_disc_block(hidden_dim*8, 1, 4, 1, 0, final_layer=True),
        )

        utils.initialize_weights(self)

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding=1, final_layer=False):
            '''
            Function to return a sequence of operations corresponding to a Discriminator block of DCGAN, 
            corresponding to a convolution, a batchnorm (except for in the last layer), and an activation.
            Parameters:
                input_channels: how many channels the input feature representation has
                output_channels: how many channels the output feature representation should have
                kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
                stride: the stride of the convolution
                final_layer: a boolean, true if it is the final layer and false otherwise 
                        (affects activation and batchnorm)
            '''
            # Build the neural block
            if not final_layer:
                return nn.Sequential(
                    SNConv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                    nn.BatchNorm2d(output_channels),
                    nn.LeakyReLU(0.2)
                )
            else: # Final Layer
                return nn.Sequential(
                    SNConv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
                )

    def forward(self, image):
        '''
        Function for completing a forward pass of the Discriminator: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_dim)
        '''
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)

class DRAGAN(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 64
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.hdg = args.hdg
        self.hdd = args.hdd
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.input_size = args.input_size
        self.data_root = args.data_root
        self.z_dim = 100
        self.lambda_ = 0.25

        # load dataset
        self.data_loader = dataloader(self.data_root, self.input_size, self.batch_size, 2)
        data = self.data_loader.__iter__().__next__()[0]

        # networks init
        self.G = Generator(input_dim=self.z_dim, output_dim=data.shape[1], hidden_dim=self.hdg)
        self.D = Discriminator(input_dim=data.shape[1], hidden_dim=self.hdd)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCEWithLogitsLoss().cuda()
        else:
            self.BCE_loss = nn.BCEWithLogitsLoss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')

        # fixed noise
        self.sample_z_ = torch.randn((self.batch_size, self.z_dim))
        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            epoch_start_time = time.time()
            self.G.train()
            with tqdm(self.data_loader, unit="batch") as tepoch:
                for iter, (x_, _) in enumerate(tepoch):
                    tepoch.set_description(f'Epoch {epoch+1}')
                    if iter == self.data_loader.dataset.__len__() // self.batch_size:
                        break

                    z_ = torch.randn((self.batch_size, self.z_dim))
                    if self.gpu_mode:
                        x_, z_ = x_.cuda(), z_.cuda()

                    # update D network
                    self.D_optimizer.zero_grad()

                    D_real = self.D(x_)
                    D_real_loss = self.BCE_loss(D_real, self.y_real_)

                    G_ = self.G(z_)
                    D_fake = self.D(G_.detach())
                    D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

                    """ DRAGAN Loss (Gradient penalty) """
                    # This is borrowed from https://github.com/kodalinaveen3/DRAGAN/blob/master/DRAGAN.ipynb
                    alpha = torch.rand(self.batch_size, 1, 1, 1).cuda()
                    if self.gpu_mode:
                        alpha = alpha.cuda()
                        x_p = x_ + 0.5 * x_.std() * torch.rand(x_.size()).cuda()
                    else:
                        x_p = x_ + 0.5 * x_.std() * torch.rand(x_.size())

                    differences = x_p - x_
                    interpolates = x_ + (alpha * differences)
                    interpolates.requires_grad = True
                    pred_hat = self.D(interpolates)

                    if self.gpu_mode:
                        gradients = grad(outputs=pred_hat, inputs=interpolates, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
                    else:
                        gradients = grad(outputs=pred_hat, inputs=interpolates, grad_outputs=torch.ones(pred_hat.size()),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

                    gradient_penalty = self.lambda_ * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

                    D_loss = D_real_loss + D_fake_loss + gradient_penalty
                    self.train_hist['D_loss'].append(D_loss.item())
                    D_loss.backward()
                    self.D_optimizer.step()

                    # update G network
                    self.G_optimizer.zero_grad()

                    G_ = self.G(z_)
                    D_fake = self.D(G_)
                    G_loss = self.BCE_loss(D_fake, self.y_real_)
                    self.train_hist['G_loss'].append(G_loss.item())

                    G_loss.backward()
                    self.G_optimizer.step()

                    tepoch.set_postfix(D_loss= D_loss.item(),  G_loss=G_loss.item())

                self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
                with torch.no_grad():
                    self.visualize_results((epoch+1))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        utils.generate_animation(self.result_dir + '/' + self.model_name + '/' + self.model_name,
                                 self.epoch)
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.model_name), self.model_name)

    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.model_name)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random noise """
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                sample_z_ = sample_z_.cuda()

            samples = self.G(sample_z_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def save(self):
        save_dir = os.path.join(self.save_dir, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))