import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn.functional as F
from cleanfid import fid
import matplotlib.pyplot as plt
import os


# Define VAE architecture
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, latent_dim)  # Mean
        self.fc22 = nn.Linear(400, latent_dim)  # Log-variance
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Loss function for VAE
def loss_function(recon_x, x, mu, logvar):
    # Denormalize data back to [0, 1]
    x = (x + 1) / 2
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE, KLD, BCE + KLD


# Train function
def train(epoch, model, optimizer, train_loader):
    model.train()
    train_loss = 0
    train_bce = 0
    train_kld = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        BCE, KLD, loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        train_bce += BCE.item()
        train_kld += KLD.item()
        optimizer.step()
    return train_loss / len(train_loader.dataset), train_bce / len(train_loader.dataset), train_kld / len(
        train_loader.dataset)


# Validation function
def validate(epoch, model, valid_loader):
    model.eval()
    val_loss = 0
    val_bce = 0
    val_kld = 0
    with torch.no_grad():
        for data, _ in valid_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            BCE, KLD, loss = loss_function(recon_batch, data, mu, logvar)
            val_loss += loss.item()
            val_bce += BCE.item()
            val_kld += KLD.item()
    return val_loss / len(valid_loader.dataset), val_bce / len(valid_loader.dataset), val_kld / len(
        valid_loader.dataset)


# Save real images for FID calculation
def save_real_images(dataset, output_dir, num_images=500):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_images):
        img, _ = dataset[i]
        img = img.squeeze(0).numpy()
        plt.imsave(f"{output_dir}/real_{i}.png", img, cmap='gray')


# Generate images for FID calculation
def generate_images(model, num_images, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for i in range(num_images):
            z = torch.randn(1, 20).to(device)
            sample = model.decode(z).cpu().view(28, 28)
            plt.imsave(f"{output_dir}/gen_{i}.png", sample, cmap='gray')


# Main function
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parameters
    latent_dim = 20
    epochs = 100
    batch_size = 128
    fid_output_dir = "generated_images"
    real_images_dir = "real_images"
    log_file = "training_log.txt"
    os.makedirs(fid_output_dir, exist_ok=True)

    # Create or clear log file
    with open(log_file, 'w') as f:
        f.write("Epoch, Train Loss, Train BCE, Train KLD, Valid Loss, Valid BCE, Valid KLD, FID\n")

    # Dataset and DataLoader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    valid_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Save real images for FID calculation
    save_real_images(train_dataset, real_images_dir, num_images=500)

    # Model, optimizer
    model = VAE(latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training and validation loop
    train_losses, valid_losses, fid_scores = [], [], []
    for epoch in range(1, epochs + 1):
        train_loss, train_bce, train_kld = train(epoch, model, optimizer, train_loader)
        val_loss, val_bce, val_kld = validate(epoch, model, valid_loader)

        # Generate images and compute FID
        epoch_fid_dir = os.path.join(fid_output_dir, f"epoch_{epoch}")
        generate_images(model, num_images=500, output_dir=epoch_fid_dir)
        fid_score = fid.compute_fid(fdir1=real_images_dir, fdir2=epoch_fid_dir, device=device)
        fid_scores.append(fid_score)

        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}, FID: {fid_score:.4f}")

        # Log the losses and FID score
        with open(log_file, 'a') as f:
            f.write(
                f"{epoch}, {train_loss:.4f}, {train_bce:.4f}, {train_kld:.4f}, {val_loss:.4f}, {val_bce:.4f}, {val_kld:.4f}, {fid_score:.4f}\n")

        train_losses.append(train_loss)
        valid_losses.append(val_loss)

    # Plot the learning curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Train')
    plt.plot(range(1, epochs + 1), valid_losses, label='Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.savefig("learning_curve.png")
    plt.show()

    # Plot the FID scores
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), fid_scores, label='FID')
    plt.xlabel('Epoch')
    plt.ylabel('FID Score')
    plt.title('FID Score per Epoch')
    plt.legend()
    plt.savefig("fid_curve.png")
    plt.show()

