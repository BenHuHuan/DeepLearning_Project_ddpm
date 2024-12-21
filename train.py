import torch
from denoising_diffusion.ddpm import DenoiseDiffusion
from denoising_diffusion.unet import UNet
import torchvision
from torchvision.utils import save_image
import os
import argparse
from denoising_diffusion.dataset import MNISTDataset,CelebADataset


n_channels=64
channel_multipliers=[1,2,2,4]
is_attention= [True, True, True, True]
image_size=32
learning_rate=2e-5

def run():
    for epoch in range(epochs):
        for batch_ndx, data in enumerate(data_loader):
            data = data.to("cuda")
            optimizer.zero_grad()
            loss = diffusion.loss(data)
            loss.backward()
            optimizer.step()
            if batch_ndx % 100==0:
                print ('Epoch [{}] Step [{}/{}], Loss: {:.4f}'.format(epoch, batch_ndx, len(data_loader), loss.item()))
                
        if epoch % sample_epoch==0:
            with torch.no_grad():
                x = torch.randn([n_samples, image_channels, image_size, image_size], device="cuda")
                for t_ in range(n_steps):
                    t = n_steps - t_ - 1
                    x = diffusion.p_sample(x, x.new_full((n_samples,), t, dtype=torch.long))
                path = os.path.join("outputs/{}/epoch_{}".format(dataset_style, epoch))
                if not os.path.exists(path) :
                    os.makedirs(path)
                    
                for idx in range(n_samples):
                    sample = x[idx]
                    save_image(sample, path+'/sample_{}.png'.format(idx))

                print("save image to :{}".format(path))



parser = argparse.ArgumentParser(description='Get the data info')
parser.add_argument('--dataset', default='mnist', type=str, help='mnist or CelebA')
parser.add_argument('--epochs',  default=5, type=int, help='mnist=10, CelebA=50')
parser.add_argument('--sample_epoch',  default=5, type=int, help='mnist=1, CelebA=5')
parser.add_argument('--channels',  default=1, type=int, help='mnist=1, CelebA=3')
parser.add_argument('--n_sample',  default=200, type=int, help='the number of generation image')
parser.add_argument('--T',  default=1000, type=int, help='time step')
parser.add_argument('--batch_size',  default=64, type=int, help='batch size')
args = parser.parse_args()

image_channels = args.channels
epochs = args.epochs
dataset_style = args.dataset  
sample_epoch = args.sample_epoch
n_samples = args.n_sample
n_steps = args.T
batch_size = args.batch_size

if __name__ == '__main__':
    eps_model = UNet(
                image_channels=image_channels,
                n_channels=n_channels,
                ch_mults=channel_multipliers,
                is_attn=is_attention,
            ).to("cuda")

    diffusion = DenoiseDiffusion(
        eps_model=eps_model,
        n_steps=n_steps,
        device ="cuda",
    )

    if dataset_style == 'mnist':
        dataset = MNISTDataset(image_size)
    elif dataset_style == 'CelebA':
        dataset = CelebADataset(image_size)
        
    data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, pin_memory=True)
    optimizer = torch.optim.Adam(eps_model.parameters(), lr=learning_rate)

    run()
                