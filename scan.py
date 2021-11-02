import os
from PIL import Image, ImageDraw
import numpy as np
import torch
from model import Classifier
import argparse

class ImageData(torch.utils.data.Dataset):
    """Holds image data
    Attributes:
        X: (N, w, h, 3) uint8 tensor of image data
    """
    def __init__(self, X):
        self.X = X
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx]
    
def tile_image(image, tile_width, tile_height):
    """Convert and image into an array of overlapping tiles
    Args:
        image: PIL.Image
        tile_width: width of tiles
        tile_height: height of tiles
    Returns:
        (N*M, tile_height, tile_width, 3) uint8 ndarray. N*M is the number
    of tiles which is chosen to create the minimum overlap size. 
    """
    w, h = image.size
    N = int(np.ceil(w/tile_width))
    M = int(np.ceil(h/tile_height))
    o_w = int(np.floor((N*tile_width-w)/(N-1)))
    o_h = int(np.floor((M*tile_height-h)/(M-1)))
    image_tiles = []
    locations = []
    for i in range(M):
        for j in range(N):
            left = tile_width*j-o_w*j
            top = tile_height*i-o_h*i
            right = tile_width*(j+1)-o_w*j
            bottom = tile_height*(i+1)-o_h*i
            locations.append((left, top, right, bottom))
            tile = image.crop((left, top, right, bottom))
            image_tiles.append(np.array(tile).transpose((2,0,1)))
    image_tiles = np.array(image_tiles)
    return image_tiles, locations


def scan_image(model, device, batch_size, threshold, source_path, output_dir, 
               tile_width, tile_height):
    """Uses a model to scan an array of image tiles and draw rectangles around
    detections
    Args:
        model: torch.nn.module
        device: device for computation
        batch_size: inference batch_size
        threshold: detection threshold
        source_path: path to the image
        output_dir: output directory
        tile_width: width of tiles
        tile_height: height of tiles
    """
    image = Image.open(source_path)
    image_tiles, locations = tile_image(image, tile_width, tile_height)
    image_tensor = torch.from_numpy(image_tiles)
    if batch_size is None:
        batch_size = len(image_tensor)
    data = ImageData(image_tensor)
    sampler = torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(range(len(data))), 
                                                                        batch_size = batch_size,
                                                                        drop_last = False)
    loader = torch.utils.data.DataLoader(data, batch_sampler = sampler, 
                                         num_workers = 0)
    predictions = []
    for x in loader:
        x = x.to(device)
        with torch.no_grad():
            predictions.append(torch.nn.Sigmoid()(model(x)).cpu().numpy())
    predictions = np.concatenate(predictions, axis = 0)
    """
    Simpler code for inference on single batch using CPU
    with torch.no_grad():
        predictions = torch.nn.Sigmoid()(model(image_tensor)).numpy()
    """
    localization_image = image.copy()
    draw = ImageDraw.Draw(localization_image)
    for i, tile in enumerate(image_tiles):
        if np.float(predictions[i,-1])>threshold:
            image_name = source_path.split('/')[-1].split('.')[0]+'_'+str(i)+'.png'
            Image.fromarray(tile.transpose((1,2,0))).save(output_dir+'/'+image_name)
            draw.rectangle(locations[i], outline = 'red', width = 3)
    image_name = source_path.split('/')[-1]
    localization_image.save(output_dir+'/'+image_name)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars = '@')
    
    parser.add_argument('--cuda', action = 'store_true', 
                        help = 'use GPU if available')
    
    parser.add_argument('--source_path', type = str, default = None,
                        help = 'path to image')
    
    parser.add_argument('--weights_path', type = str, default = None,
                        help = 'path to model weights')
    
    parser.add_argument('--tile_width', type = int, default = 300,
                        help = 'width of images input to model')
    
    parser.add_argument('--tile_height', type = int, default = 300,
                        help = 'height of images input to model')
    
    parser.add_argument('--num_classes', type = int, default = 1,
                        help = 'dimension of model output')
    
    parser.add_argument('--threshold', type = float, default = 0.5,
                        help = 'detection threshold')
    
    parser.add_argument('--output_dir', type = str, default = 'output', 
                        help = 'output directory')
    
    parser.add_argument('--batch_size', type = int, default = None, 
                        help = 'batch size')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    
    model = Classifier(args.num_classes)
    model.load_state_dict(torch.load(args.weights_path))
    model.eval()
    model.to(device)
    
    scan_image(model, device, args.batch_size, args.threshold, args.source_path, 
               args.output_dir, args.tile_width, args.tile_height)
    
if __name__=='__main__':
    main()