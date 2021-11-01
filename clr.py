import os
import random
from PIL import Image
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import Classifier
import schedulers

class CLRDataset(Dataset):
    """torch.utils.data.Dataset subclass for holding image data for contrastive
    learning
    Attributes:
        X: torch.tensor of images with shape (num_samples, num_channels, height, width)
        transform: callable transform for data augmentation
    """
    def __init__(self, X, transform):
        self.X = X
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x_0 = self.X[idx]
        x_1 = x_0.clone()
        x_0 = self.transform(x_0)
        x_1 = self.transform(x_1)
        return (x_0, x_1)
    
class FourfoldRotation():
    """Rotates image by 0, 90, 180 or 270 degrees, with equal probability.
    """
    def __call__(self, x):
        k = int(torch.randint(0,4,(1,)))
        return torch.rot90(x, k = k, dims = [-2,-1])
        
class VerticalFlip():
    """Flips the image vertically or doesn't, with equal probability. 
    """
    def __call__(self, x):
        k = int(torch.randint(0,2,(1,)))
        if k == 1:
            return torch.flip(x, dims = [-2])
        else:
            return x

class ColorShift():
    """Mutiply image channels by random factors in a range
    Attributes:
        amplitude: maximum distance of multiplication factors from one
    """
    def __init__(self, amplitude):
        self.amplitude = amplitude
    
    def __call__(self, x):
        factors = 1+2*self.amplitude*(torch.rand((3,1,1))-0.5)
        x = torch.clamp(factors*x, 0, 255).type(torch.uint8)
        return x
    
def load_images(images_dir, images, loader_transform): 
    """Loads images. 
    Args:
        images_dir: directory of images
        images: filenames of images to load
        loader_transform: callable transform to apply to images after loading
    Returns:
        X: (num_samples, num_channels, height, width) torch.tensor of images
    """
    X = []
    for image in images:
        image = Image.open(images_dir+'/'+image)
        image = loader_transform(image)
        image_array = np.array(image)
        image_array = image_array.transpose((2,0,1))
        X.append(image_array)
    X = np.array(X)
    X = torch.from_numpy(X)
    return X
   
def create_datasets(images_dir, batch_size, num_workers, input_size, 
                    loader_transform, f_split, f_use):
    """Creates DataSets and DataLoaders for the training and validation sets
    """
    interpolation = transforms.InterpolationMode.BILINEAR
    rotation =  transforms.RandomApply([transforms.RandomRotation(20.0, 
                                    interpolation = interpolation)], p = 0.2)
    crop = transforms.RandomResizedCrop(input_size, scale = (0.25,1.0),
                                        ratio = (1.0,1.0),
                                        interpolation = interpolation)
    color_shift = transforms.RandomApply([ColorShift(0.5)],
                                          p = 0.8)
    gaussian_blur = transforms.RandomApply([transforms.GaussianBlur(5)],
                                           p = 0.2)
    data_transform = transforms.Compose([FourfoldRotation(), 
                                             VerticalFlip(),
                                             color_shift,
                                             rotation,
                                             gaussian_blur,
                                             crop])
    extensions = ('.jpg', '.png', '.jpeg', '.tif')
    images = os.listdir(images_dir)
    images = [image for image in images if image.endswith(extensions)]
    random.shuffle(images)
    
    N_samples = int(np.ceil(f_use*len(images)))
    N_train = int(np.ceil(f_split*N_samples))
    print('Loading training data')
    X = load_images(images_dir, images[:N_train], loader_transform)
    print('Loaded '+str(len(X))+' training images')
    train_dataset = CLRDataset(X, transform = data_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                              shuffle = True, pin_memory = True, 
                              num_workers = num_workers)
    
    
    print('Loading validation data')
    X = load_images(images_dir, images[N_train:N_samples], loader_transform)
    print('Loaded '+str(len(X))+' validation images')
    val_dataset = CLRDataset(X, transform = data_transform)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, 
                              shuffle = True, pin_memory = True, 
                              num_workers = num_workers)

    return train_dataset, train_loader, val_dataset, val_loader

class ContrastiveLoss():
    """Loss function for contrastive learning
    Attributes:
        temperature: scaling factor for softmax
        logsoftmax: nn.Logsoftmax instance
        epislon: cut-off for cosine similarity normalization
    """
    def __init__(self, temperature, epsilon = 1e-8):
        self.temperature = temperature
        self.logsoftmax = torch.nn.LogSoftmax(dim = 1)
        self.epsilon = epsilon
        
    def __call__(self, z_0, z_1):
        z_0 = z_0/torch.clamp(torch.sqrt(torch.sum(z_0*z_0, axis = 1, keepdim = True)), 
                              min = self.epsilon)
        z_1 = z_1/torch.clamp(torch.sqrt(torch.sum(z_1*z_1, axis = 1, keepdim = True)), 
                              min = self.epsilon)
        dots = torch.matmul(z_0, torch.transpose(z_1, 0, 1))/self.temperature
        loss = -torch.diagonal(self.logsoftmax(dots)).mean()
        return loss
    
def train(model, loss_fn, optimizer, scheduler, train_loader, val_loader, params):
    """Train the model
    Args:
        model: nn.Module
        loss_fn: callable loss function
        optimizer: torch.optim.Optimizer
        scheduler: learning rate scheduler with .step() method
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        params: dict of params
    Returns:
        ndarray of training losses, ndarray of validation losses
    """
    print('Starting training')
    device = params['Device']
    epochs = params['Epochs']
    output_frequency = params['Output Frequency']
    save_frequency = params['Save Frequency']
    output_dir = params['Output Directory']
    
    losses = []
    val_losses = []
    for epoch in range(epochs):
        epoch_losses = []
        model.train()
        for x_0, x_1 in train_loader:
            x = torch.cat((x_0, x_1), dim = 0)
            x = x.to(device)
            z = model(x)
            z_0, z_1 = z.split([len(z)//2, len(z)//2])
            loss = loss_fn(z_0, z_1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.detach().to('cpu').numpy())
        epoch_loss = np.mean(epoch_losses)
        losses.append(epoch_loss)
    
        if (epoch+1)%output_frequency == 0:
            print('Training loss on epoch '+str(epoch+1)+': '+ '{:.2f}'.format(epoch_loss))
            
        epoch_val_losses = []
        model.eval()
        for x_0, x_1 in val_loader:
            x = torch.cat((x_0, x_1), dim = 0)
            x = x.to(device)
            with torch.no_grad():
                 z = model(x)
                 z_0, z_1 = z.split([len(z)//2, len(z)//2])
                 val_loss = loss_fn(z_0, z_1)
            epoch_val_losses.append(val_loss.to('cpu').numpy())
        epoch_val_loss = np.mean(np.array(epoch_val_losses))
        val_losses.append(epoch_val_loss)
        
        if (epoch+1)%output_frequency == 0:
            print('Validation loss on epoch '+str(epoch+1)+': '+ '{:.2f}'.format(epoch_val_loss))
        
        if (epoch+1)%save_frequency == 0:
            results = np.column_stack((np.arange(1,epoch+2), np.array(losses), np.array(val_losses)))
            np.savetxt(output_dir+'/losses.csv', results, delimiter = ',')
            torch.save(model.state_dict(), output_dir+'/'+'model_weights-'+str(epoch+1)+'.pth')
            torch.save(model.backbone.state_dict(), output_dir+'/'+'backbone_weights-'+str(epoch+1)+'.pth')
        
        if scheduler is not None:
            scheduler.step()
        
    return np.array(losses), np.array(val_losses)

def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars = '@')
    
    parser.add_argument('--cuda', action = 'store_true', 
                        help = 'use GPU if available')
    
    parser.add_argument('--weight_decay', type = float, default = 0.3,
                        help = 'weight decay')
    
    parser.add_argument('--lr', type = float, default = 3e-4,
                        help = 'learning rate')
    
    parser.add_argument('--momentum', type = float, default = 0.9,
                        help = 'momentum')
    
    parser.add_argument('--optimizer', default = 'Adam', choices = ['Adam', 'SGD'],
                        help = 'optimizer type')
    
    parser.add_argument('--scheduler', default = None, choices = ['OneCycle', 'StepDecay'],
                        help = 'scheduler type')
    
    parser.add_argument('--epochs', type = int, default = 20, 
                        help = 'training epochs')
    
    parser.add_argument('--output_frequency', type = int, default = 1, 
                        help = 'epochs between loss output')
    
    parser.add_argument('--save_frequency', type = int, default = 10, 
                        help = 'epochs between saving weights')
    
    parser.add_argument('--output_dir', type = str, default = 'output', 
                        help = 'output directory')
    
    parser.add_argument('--images_dir', type = str, default = 'images',
                        help = 'path to dataset directory')
    
    parser.add_argument('--f_split', type = float, default = 0.8, 
                        help = 'fraction of examples to use for training set')
    
    parser.add_argument('--f_use', type = float, default = 1.0, 
                        help = 'fraction of data to use')
    
    parser.add_argument('--temperature', type = float, default = 0.1, 
                        help = 'contrastive loss temperature')
    
    parser.add_argument('--output_features', type = int, default = 256,
                        help = 'embedding dimension')
    
    parser.add_argument('--weights_path', type = str, default = None,
                        help = 'path to initial weights')
    
    parser.add_argument('--batch_size', type = int, default = 256, 
                        help = 'batch size')
    
    parser.add_argument('--num_workers', type = int, default = 2, 
                        help = 'number of workers for DataLoader')
    
    parser.add_argument('--load_size', type = int, default = 200,
                        help = 'crop images to this size on loading')
    
    parser.add_argument('--input_size', type = int, default = 100,
                        help = 'size of images input to model')
        
    parser.add_argument('--scheduler_params', type = float, nargs = '*', default = None,
                        help = 'parameters passed to learning rate scheduler')
    
    parser.add_argument('--seed', type = int, default = 2319,
                        help = 'seed for random number generators')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    loader_transform = transforms.CenterCrop(args.load_size)
    train_dataset, train_loader, val_dataset, val_loader = create_datasets(args.images_dir,
                                                                           args.batch_size, 
                                                                           args.num_workers, 
                                                                           args.input_size,
                                                                           loader_transform,
                                                                           args.f_split,
                                                                           args.f_use)
    
    params = dict()
    params['Device'] = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    params['Epochs'] = args.epochs
    params['Output Frequency'] = args.output_frequency
    params['Save Frequency'] = args.save_frequency
    params['Output Directory'] = args.output_dir
    
    model = Classifier(args.output_features)
    model.to(params['Device'])
    
    if args.weights_path:
        model.load_state_dict(torch.load(args.weights_path))
    
    if args.optimizer == 'Adam':
        optimizer = torch.optim.AdamW(model.parameters(), weight_decay = args.weight_decay, 
                                      betas = (args.momentum, 0.99), 
                                      lr = args.lr)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), weight_decay = args.weight_decay, 
                                      betas = args.momentum, 
                                      lr = args.lr)
    
    if args.scheduler == 'OneCycle':
        scheduler = schedulers.OneCycle(optimizer, args.scheduler_params)
    elif args.scheduler == 'StepDecay':
        scheduler = schedulers.StepDecay(optimizer, args.scheduler_params)
    else:
        scheduler = None

    loss_fn = ContrastiveLoss(args.temperature)
    
    losses, val_losses = train(model, loss_fn, optimizer, scheduler, train_loader, val_loader, params)
    torch.save(model.state_dict(), args.output_dir+'/model_weights-final.pth')
    torch.save(model.backbone.state_dict(), args.output_dir+'/'+'backbone_weights-final.pth')
    
    results = np.column_stack((np.arange(1,args.epochs+1), losses, val_losses))
    np.savetxt(args.output_dir+'/losses.csv', results, delimiter = ',')
    
    plt.figure()
    plt.plot(np.arange(1,args.epochs+1), val_losses)
    plt.plot(np.arange(1,args.epochs+1), losses)
    plt.xlim([1, args.epochs+1])
    plt.xticks(fontsize = 14);
    plt.yticks(fontsize = 14);
    plt.ylabel('Loss', fontsize = 15);
    plt.xlabel('Epoch', fontsize = 15);
    plt.legend(['Validation', 'Training'], fontsize = 15, loc = 'best')
    plt.tight_layout()
    plt.savefig(args.output_dir+'/losses.pdf')
    plt.close()    
    print('Finished training')
    
if __name__ == '__main__':
    main()