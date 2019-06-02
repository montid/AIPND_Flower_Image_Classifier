# Imports here
import torch
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
import json

#check hidden_units input(if only one integer then convert it to a list)
def check_h(h):
    if isinstance(h, int):
        g = [h]
    else:
        g = h
    return g


def load_data(data_directory = "./flowers" ):
    '''
    Arguments : Dataset's path
    Returns : dataloaders and image_datasets dictionaries for training and validating the model.
    This function receives the source folder of the image files and performs the following action:

    Training data augmentation:  Torchvision transforms are used to augment the training data with random scaling, rotations, mirroring, and/or cropping
    Data normalization:          Training, validation, and testing data is appropriately cropped and normalized
    Data loading:                Image_datasets (train, validation, test) are loaded with torchvision's ImageFolder
    Data batching:               Datasets loaded with torchvision's DataLoader
    '''

    data_dir = data_directory
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'
    valid_dir = data_dir + '/valid'
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    image_datasets = dict()
    image_datasets['train'] = datasets.ImageFolder(train_dir, transform=train_transforms)
    image_datasets['valid'] = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    image_datasets['test'] = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the transforms, define the dataloaders
    dataloaders = dict()
    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True)
    dataloaders['valid'] = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64)
    dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=64)

    #return dataloaders['train'] , dataloaders['valid'], dataloaders['test']
    return image_datasets,dataloaders


# Process a PIL image for use in a PyTorch model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    pil_image = Image.open(image) #open the image


    img_tranformer = transforms.Compose([transforms.Resize(256), #resize the images where the shortest side is 256 pixels
                                     transforms.CenterCrop(224), #crop out the center 224x224 portion of the image
                                     transforms.ToTensor()])

    pil_image = img_tranformer(pil_image) #apply transformation

    #Color channels of images are typically encoded as integers 0-255
    #, but the model expected floats 0-1. You'll need to convert the values
    np_image = np.array(pil_image)

    #the network expects the images to be normalized in a specific way
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std
    np_image = np.transpose(np_image, (2, 0, 1))

    return np_image

# Predict function returning top probabilities, top labels and top classes
def predict(image_path, model, topk, isgpu, category_names):

    device = torch.device("cuda:0" if isgpu is True else "cpu")

    model.to(device)
    model.eval() # inference mode

    # Process image
    img = process_image(image_path)

    # Numpy -> Tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    image_tensor = image_tensor.to(device)


    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)

    # Probs
    with torch.no_grad():
        probs = torch.exp(model.forward(model_input))

    # Top probs
    top_probs, top_labs = probs.topk(topk)
    # can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
    top_probs = top_probs.detach().cpu().numpy().tolist()[0]
    top_labs = top_labs.detach().cpu().numpy().tolist()[0]

    # dictionary mapping the integer encoded categories to the actual names of the flowers
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]

    return top_probs, top_labels, top_flowers

# print prediction results
def print_results(checkpoint, image_path, topk, category_names, isgpu, top_probs, top_flowers):
    print("{}Checkpoint:             {} {} ".format('\n','\t', checkpoint))
    print("Image path:             {} {} ".format('\t', image_path))
    print("Top Probabilities:      {} {} ".format('\t',topk))
    print("Category Names:         {} {} ".format('\t',category_names))

    if isgpu:
        print("GPU:                    {} {} {}".format('\t', 'ON', '\n'))
    else:
        print("GPU:                    {} {} {}".format('\t', 'OFF', '\n'))

    print("The predicted flower is:{} {:*^20s} {}probability: {:.2%}".format('\t', top_flowers[0].upper(),'\t', top_probs[0] ))
    for pbs,flws in zip(top_probs[1:],top_flowers[1:]):
        print("                        {} {:*^20s} {}probability: {:.2%}".format('\t',flws.capitalize(),'\t', pbs))

    print("Prediction completed")
