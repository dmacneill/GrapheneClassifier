import os
import torch
from torchvision import transforms
import train
import argparse
import numpy as np
import matplotlib.pyplot as plt
from model import Classifier

def predict(model, X, device, input_size):
    """Use the model to predict class probabilities for a dataset
    Args:
        model: torch.nn.module
        X: datapoints; (N, size, size) torch.tensor of uint8 images
        device: device for computation
        input_size: images are center-cropped to this size
    Return :
        (N, num_classes) ndarray of class probabilities
    """
    crop = transforms.CenterCrop(input_size)
    predictions = []
    for x in X:
        x = crop(x)
        x = x.to(device)
        with torch.no_grad():
            z = model(x.unsqueeze(dim = 0))
        prediction = torch.sigmoid(z).cpu().numpy()
        predictions.append(prediction[0])
    predictions = np.array(predictions)
    return predictions

def calculate_precision(predictions, labels):
    """Calculate the precision given classifier predictions and ground-truth
    labels
    Args:
        predictions: (num_thresholds, num_samples, num_classes) ndarray of 
        classifier labels
        labels: (num_samples, num_classes) ndarray of ground-truth labels
    Returns:
        (num_thresholds, num_classes) ndarray of precisions
    """
    true_positives = np.logical_and(predictions, np.expand_dims(labels, axis = 0))
    false_positives = np.logical_and(predictions, np.expand_dims(1-labels, axis = 0))
    true_positives = np.sum(true_positives, axis = 1)
    false_positives = np.sum(false_positives, axis = 1)
    return true_positives/np.clip((true_positives+false_positives), 1, None)

def calculate_recall(predictions, labels):
    """Calculate the recall given classifier predictions and ground-truth
    labels
    Args:
        predictions: (num_thresholds, num_samples, num_classes) ndarray of 
        classifier labels
        labels: (num_samples, num_classes) ndarray of ground-truth labels
    Returns:
        (num_thresholds, num_classes) ndarray of recalls
    """
    true_positives = np.logical_and(predictions, np.expand_dims(labels, axis = 0))
    true_positives = np.sum(true_positives, axis = 1)
    all_positives = np.expand_dims(np.sum(labels, axis = 0), axis = 0)
    return true_positives/np.clip(all_positives, 1, None)

def test_model(X, y, model, device, input_size, num_thresholds):
    """Calculates precision and recall for a model and dataset
    Args:
        X: (N, size, size) torch.tensor of uint8 images
        y: (N, num_classes) ndarray of ground-truth labels
        model: torch.nn.module for inference
        device: device for computation
        input_size: images are center-cropped to this size
        num_thresholds: number of threshold values to try
    Returns:
        thresholds, probabilities, precisions, recalls; thresholds is a 
    (num_thresholds,) ndarray, probabilities is a (N,1) ndarray, and 
    precisions and recalls are (num_thresholds, N, num_classes) ndarrays.
    """
    probabilities = predict(model, X, device, input_size)
    predictions = np.expand_dims(probabilities, axis = 0)
    thresholds = np.arange(1, 1+num_thresholds)/(1+num_thresholds)
    thresholds_arr = np.expand_dims(np.expand_dims(thresholds, axis = 1), axis = 2)
    predictions = np.heaviside(predictions-thresholds_arr, 0)
    precisions = calculate_precision(predictions, y)
    recalls = calculate_recall(predictions, y)
    
    return thresholds, probabilities, precisions, recalls

def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars = '@')
    
    parser.add_argument('--cuda', action = 'store_true', 
                        help = 'use GPU if available')
    
    parser.add_argument('--images_dir', type = str, default = 'dataset/validation_images',
                        help = 'path to image directory')
    
    parser.add_argument('--labels_dir', type = str, default = 'dataset/validation_labels',
                        help = 'path to label directory')
    
    parser.add_argument('--output_dir', type = str, default = 'output', 
                        help = 'output directory')
    
    parser.add_argument('--weights_path', type = str, default = 'model_weights.pth',
                        help = 'path to model weights')
    
    parser.add_argument('--input_size', type = int, default = 300,
                        help = 'images are cropped to this size')
    
    parser.add_argument('--num_thresholds', type = int, default = 19,
                        help = 'number of values to try for classification threshold')
    
    parser.add_argument('--any_class', action = 'store_true', 
                    help = 'add a class indicating the prescence of any class')
    
    args = parser.parse_args()   
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    
    print('Loading data')
    loader_transform = transforms.CenterCrop(args.input_size)
    X, y, _ = train.load_images_and_labels(args.images_dir,  
                                           args.labels_dir, 
                                           args.any_class,
                                           loader_transform)
    y = y.numpy()
    np.savetxt(args.output_dir+'/labels.csv', y, delimiter = ',')
    
    model = Classifier(y.shape[1])
    model.load_state_dict(torch.load(args.weights_path))
    model.to(device)
    model.eval()
    
    print('Testing model')
    thresholds, probabilities, precisions, recalls = test_model(X, y, model, 
                                                                device, 
                                                                args.input_size, 
                                                                args.num_thresholds)
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
        