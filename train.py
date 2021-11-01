import os
import random
import shutil
from PIL import Image
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import (Dataset, DataLoader, WeightedRandomSampler, 
                              RandomSampler)
from torchvision import transforms
from model import Classifier
import test_model
import schedulers

class FlakeDataset(Dataset):
    """torch.utils.data.Dataset subclass for holding image and label data
    Attributes:
        X: torch.tensor of images with shape (num_samples, num_channels, height, width)
        y: torch.tensor of labels with shape (num_samples, num_classes);
        the value y[i,j] is 1 if class j is present in image i and zero otherwise
        transform: callable object for data augmentation
    """
    def __init__(self, X, y, transform):
        self.X = X
        self.y = y
        self.transform = transform
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (self.transform(self.X[idx]), self.y[idx])
    
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

def dataset_from_dir(dataset_dir, f_split):
    """Creates a binary classification dataset of the form read by 
    load_images_and_labels from a directory with two subfolders of images, 
    one labeled "Good" (Class 1) and the other labeled "Bad" (Class 0)
    Args:
        dataset_dir: path to dataset directory
        f_split: fraction of data to use for training set
    """
    dataset_folders = ('training_images','training_labels','validation_images', 'validation_labels')
    target_paths = [dataset_dir+'/'+x for x in dataset_folders]
    for folder in target_paths:
        if os.path.isdir(folder):
            shutil.rmtree(folder)
            os.mkdir(folder)
        else:
            os.mkdir(folder)
    class_folders = ['Bad', 'Good']
    class_paths = [dataset_dir+'/'+folder for folder in class_folders]
    for i, class_path in enumerate(class_paths):
        images = os.listdir(class_path)
        random.shuffle(images)
        N_train = int(np.ceil(f_split*len(images)))
        for j, image in enumerate(images):
            target_image_dir = target_paths[0] if j<N_train else target_paths[2]
            target_label_dir = target_paths[1] if j<N_train else target_paths[3]
            shutil.copy(class_path+'/'+image, target_image_dir+'/'+\
                        class_folders[i]+'_'+image)
            label = np.array([i])
            np.savetxt(target_label_dir+'/'+class_folders[i]+'_'+\
                       image.split('.')[0]+'-labels.csv', label)
    
def load_images_and_labels(images_dir, labels_dir, any_class, loader_transform,
                           f_use = 1.0): 
    """Loads labels and corresponding images. 
    Args:
        image_dir: directory of images
        labels_dir: directory of labels
        any_class: include a class that indicates if any class is present
        loader_transform: callable transform to apply to images after loading
        f_use: fraction of the data to load
    Returns: X, y, sampler_weights
        X: (num_samples, num_channels, height, width) torch.tensor of data
        y (num_samples, num_classes) torch.tensor of labels
        sampler_weights: (num_samples,) ndarray of weights that will balance 
        the fraction of samples with and without any class present.
    """

    images = os.listdir(images_dir)
    random.shuffle(images)
    N_samples = int(np.ceil(f_use*len(images)))
    images = images[:N_samples]
    
    X = []
    y = []
    for image in images:
        try:
            label = image.split('.')[0]+('-labels.csv')
            image = Image.open(images_dir+'/'+image)
            image = loader_transform(image)
            image_array = np.array(image)
            image_array = image_array.transpose((2,0,1))
            X.append(image_array)
            y_i = np.loadtxt(labels_dir+'/'+label, delimiter = ',')
            if not y_i.shape:#convert binary classifier labels from 0d to 1d
                y_i = np.array([y_i])
            y.append(y_i)
        except FileNotFoundError:
            print('Missing label file', label)
    
    X = np.array(X)
    X = torch.from_numpy(X)
    y = np.array(y)
    any_class_present = np.max(y, axis = 1)
    sampler_weights = ((any_class_present==1)/np.sum(any_class_present==1)+\
    (any_class_present==0)/np.sum(any_class_present==0))*len(y)
    if any_class: y = np.concatenate((y, np.expand_dims(any_class_present, axis = 1)), axis = 1)
    y = torch.tensor(y, dtype = torch.float)
    return X, y, sampler_weights
   
def create_datasets(dataset_dir, batch_size, num_workers, input_size, 
                    any_class, rebalance_classes, loader_transform, f_use):
    """Creates DataSets and DataLoaders for the training and validation sets
    """
    interpolation = transforms.InterpolationMode.BILINEAR
    rotation =  transforms.RandomApply([transforms.RandomRotation(20.0, 
                                    interpolation = interpolation)], p = 0.25)
    crop = transforms.RandomCrop(input_size)
    color_shift = transforms.RandomApply([ColorShift(0.15)],
                                          p = 0.5)
    data_transform = transforms.Compose([FourfoldRotation(), 
                                             VerticalFlip(),
                                             color_shift,
                                             rotation,
                                             crop])
    
    training_images = dataset_dir+'/training_images'
    training_labels = dataset_dir+'/training_labels'
    print('Loading training data')
    X, y, sample_weights = load_images_and_labels(training_images, 
                                                  training_labels,
                                                  any_class,
                                                  loader_transform,
                                                  f_use)
    if rebalance_classes:
        train_sampler = WeightedRandomSampler(weights = sample_weights,
                                              num_samples = len(y),
                                              replacement = True)
    else:
        train_sampler = RandomSampler(y, replacement = False)
    train_dataset = FlakeDataset(X, y, transform = data_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                              sampler = train_sampler, pin_memory = True, 
                              num_workers = num_workers)
    
    validation_images = dataset_dir+'/validation_images'
    validation_labels = dataset_dir+'/validation_labels'
    print('Loading validation data')
    X, y, sample_weights = load_images_and_labels(validation_images, 
                                                  validation_labels,
                                                  any_class,
                                                  loader_transform,
                                                  1.0)
    if rebalance_classes:
        val_sampler = WeightedRandomSampler(weights = sample_weights,
                                              num_samples = len(y),
                                              replacement = True)
    else:
        val_sampler = RandomSampler(y, replacement = False)
    val_dataset = FlakeDataset(X, y, transform = data_transform)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                            sampler = val_sampler, pin_memory = True,
                            num_workers = num_workers)

    return train_dataset, train_loader, val_dataset, val_loader

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
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            loss = loss_fn(model(x_batch), y_batch)
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
        for x_val, y_val in val_loader:
            with torch.no_grad():
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                val_loss = loss_fn(model(x_val), y_val)
            epoch_val_losses.append(val_loss.to('cpu').numpy())
        epoch_val_loss = np.mean(np.array(epoch_val_losses))
        val_losses.append(epoch_val_loss)
        
        if (epoch+1)%output_frequency == 0:
            print('Validation loss on epoch '+str(epoch+1)+': '+ '{:.2f}'.format(epoch_val_loss))
        
        if (epoch+1)%save_frequency == 0:
            results = np.column_stack((np.arange(1,epoch+2), np.array(losses), np.array(val_losses)))
            np.savetxt(output_dir+'/losses.csv', results, delimiter = ',')
            torch.save(model.state_dict(), output_dir+'/'+'model_weights-'+str(epoch+1)+'.pth')
        
        if scheduler is not None:
            scheduler.step()
        
    return np.array(losses), np.array(val_losses)

def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars = '@')
    
    parser.add_argument('--cuda', action = 'store_true', 
                        help = 'use GPU if available')
    
    parser.add_argument('--dataset_from_dir', action = 'store_true',
                        help = 'create labels and validation set using directory structure')
    
    parser.add_argument('--f_split', type = float, default = 0.8,
                        help = 'fraction of data used for training set if creating validation set')
    
    parser.add_argument('--weight_decay', type = float, default = 0.05,
                        help = 'weight decay')
    
    parser.add_argument('--lr', type = float, default = 3e-4,
                        help = 'learning rate')
    
    parser.add_argument('--momentum', type = float, default = 0.9,
                        help = 'momentum')
    
    parser.add_argument('--optimizer', default = 'Adam', choices = ['Adam', 'SGD'],
                        help = 'optimizer type')
    
    parser.add_argument('--scheduler', default = None, choices = ['OneCycle', 'StepDecay'],
                        help = 'scheduler type')
    
    parser.add_argument('--epochs', type = int, default = 10, 
                        help = 'training epochs')
    
    parser.add_argument('--output_frequency', type = int, default = 1, 
                        help = 'epochs between loss output')
    
    parser.add_argument('--save_frequency', type = int, default = 10, 
                        help = 'epochs between saving weights')
    
    parser.add_argument('--output_dir', type = str, default = 'output', 
                        help = 'output directory')
    
    parser.add_argument('--dataset_dir', type = str, default = 'dataset',
                        help = 'path to dataset directory')
    
    parser.add_argument('--f_use', type = float, default = 1.0,
                        help = 'fraction of data to use')
    
    parser.add_argument('--weights_path', type = str, default = None,
                        help = 'path to initial weights')
    
    parser.add_argument('--backbone_weights_path', type = str, default = None,
                        help = 'path to backbone weights')
    
    parser.add_argument('--freeze_backbone', action = 'store_true',
                        help = 'freeze backbone weights')
    
    parser.add_argument('--batch_size', type = int, default = 8, 
                        help = 'batch size')
    
    parser.add_argument('--num_workers', type = int, default = 0, 
                        help = 'number of workers for DataLoader')
    
    parser.add_argument('--load_size', type = int, default = 400,
                        help = 'crop images to this size on loading')
    
    parser.add_argument('--input_size', type = int, default = 300,
                        help = 'size of images input to model')
        
    parser.add_argument('--scheduler_params', type = float, nargs = '*', default = None,
                        help = 'parameters passed to learning rate scheduler')
    
    parser.add_argument('--any_class', action = 'store_true', 
                        help = 'add a class indicating the prescence of any class')
    
    parser.add_argument('--rebalance_classes', action = 'store_true',
                        help = 'use weighted sampling to rebalance classes')
    
    parser.add_argument('--seed', type = int, default = 2319,
                        help = 'seed for random number generators')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    if args.dataset_from_dir:
        print('Creating dataset')
        dataset_from_dir(args.dataset_dir, args.f_split)
    
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    
    loader_transform = transforms.CenterCrop(args.load_size)
    train_dataset, train_loader, val_dataset, val_loader = create_datasets(args.dataset_dir,
                                                                           args.batch_size, 
                                                                           args.num_workers, 
                                                                           args.input_size,
                                                                           args.any_class,
                                                                           args.rebalance_classes,
                                                                           loader_transform,
                                                                           args.f_use)
    
    params = dict()
    params['Device'] = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    params['Epochs'] = args.epochs
    params['Output Frequency'] = args.output_frequency
    params['Save Frequency'] = args.save_frequency
    params['Output Directory'] = args.output_dir
    
    num_classes = train_dataset.y.shape[1]
    model = Classifier(num_classes)
    model.to(params['Device'])
    
    if args.weights_path:
        model.load_state_dict(torch.load(args.weights_path))
        
    if args.backbone_weights_path:
        model.load_backbone(torch.load(args.backbone_weights_path))
        if args.weights_path:
            print('Warning: backbone weights and model weights specified')
    
    if args.freeze_backbone:
        model.freeze_backbone()
    
    #Exclude output layer biases from weight decay
    all_params = []
    all_params = set(model.parameters())
    no_wd = set([model.head[-1].bias])
    wd = all_params-no_wd
    no_wd = list(no_wd)
    wd = list(wd)
    if args.optimizer == 'Adam':
        optimizer = torch.optim.AdamW([{'params': no_wd, 'weight_decay':0}, {'params': wd, 'weight_decay':args.weight_decay}], 
                                      betas = (args.momentum, 0.99), 
                                      lr = args.lr)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD([{'params': no_wd, 'weight_decay':0}, {'params': wd, 'weight_decay':args.weight_decay}], 
                                      betas = args.momentum, 
                                      lr = args.lr)
    
    if args.scheduler == 'OneCycle':
        scheduler = schedulers.OneCycle(optimizer, args.scheduler_params)
    elif args.scheduler == 'StepDecay':
        scheduler = schedulers.StepDecay(optimizer, args.scheduler_params)
    else:
        scheduler = None

    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_fn = loss_fn.to(params['Device'])
    
    losses, val_losses = train(model, loss_fn, optimizer, scheduler, train_loader, val_loader, params)
    model.unfreeze_backbone()
    torch.save(model.state_dict(), args.output_dir+'/model_weights-final.pth')
    
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
    
    print('Testing model')
    model.eval()
    y = val_dataset.y.numpy()
    thresholds, probabilities, precisions, recalls = test_model.test_model(val_dataset.X, 
                                                                y, 
                                                                model, 
                                                                params['Device'], 
                                                                args.input_size, 
                                                                19)
    np.savetxt(args.output_dir+'/probabilities.csv', 
               np.concatenate((probabilities, y), axis = 1),
                   delimiter = ',')
    plt.figure()
    plt.plot(thresholds, precisions,'o')
    plt.xticks(fontsize = 14);
    plt.yticks(fontsize = 14);
    plt.ylabel('Precision', fontsize = 15);
    plt.xlabel('Threshold', fontsize = 15);
    plt.ylim([0,1])
    plt.tight_layout()
    plt.savefig(args.output_dir+'/precision.pdf')
    plt.close() 
    
    plt.figure()
    plt.plot(thresholds, recalls,'o')
    plt.xticks(fontsize = 14);
    plt.yticks(fontsize = 14);
    plt.ylabel('Recall', fontsize = 15);
    plt.xlabel('Threshold', fontsize = 15);
    plt.ylim([0,1])
    plt.tight_layout()
    plt.savefig(args.output_dir+'/recall.pdf')
    plt.close() 
    
    plt.figure()
    plt.plot(thresholds, 2*precisions*recalls/np.clip(precisions+recalls,1 , None), 'o')
    plt.xticks(fontsize = 14);
    plt.yticks(fontsize = 14);
    plt.ylabel('F score', fontsize = 15);
    plt.xlabel('Threshold', fontsize = 15);
    plt.ylim([0,1])
    plt.tight_layout()
    plt.savefig(args.output_dir+'/fscore.pdf')
    plt.close() 
    print('Finished testing')
    
if __name__ == '__main__':
    main()