import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, models
#import numpy as np
#import nn_model

# load a pre-trained network
def load_pretrained_model(arch = 'vgg'):

    if arch == 'vgg':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif arch == 'densenet':
        model = models.densenet121(pretrained=True)
        input_size = 1024

    # freeze the parameters so that the gradients are not computed in backward()
    for param in model.parameters():
        param.requires_grad = False

    return model, input_size





# Builds a feedforward network with arbitrary hidden layers
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.4):
        ''' Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        '''
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''

        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
            x = self.output(x)

        return F.log_softmax(x, dim=1)



# TODO: Save the checkpoint
def save_model_checkpoint(model, input_size, epochs, save_dir, arch, hidden_layers, learning_rate, drop, optimizer, output_size):
    """
    Save trained model as checkpoint file.
    Parameters:
        model - Previously trained and tested CNN model
        input_size - Input size of CNN model
        epochs - Nr of epochs used to train the CNN
        save_dir - Directory to save the checkpoint file(default- current path)
        arch - Architecture choosen (Vgg or AlexNet)
        hidden_layers - Nr of hidden units
        learning_rate
        drop
        optimizer
        output_size
    Returns:
        None
    """

    #model.class_to_idx = image_datasets['train'].class_to_idx

    # Save Checkpoint: input, output, hidden layer, epochs, learning rate, model, optimizer, arch,drop and state_dict sure.
    checkpoint = {'input_size': input_size,
                  'output_size': output_size,
                  'hidden_layers': hidden_layers,
                  'drop': drop,
                  'epochs': epochs,
                  'learning_rate': learning_rate,
                  'arch': arch,
                  'optimizer': optimizer.state_dict,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, save_dir + '/checkpoint.pth')
    print('Model checkpoint stored at {}'.format(save_dir))



# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_model_checkpoint(filepath, isgpu):
    device = torch.device("cuda:0" if isgpu is True else "cpu")

    if device == "cuda:0":
        map_loc = 'cuda:0'
    else:
        map_loc = 'cpu'

    checkpoint = torch.load(filepath, map_location=map_loc)

    # load a pretrained network
    arch = checkpoint['arch']
    if arch == 'vgg':
        model = getattr(models, "vgg16")(pretrained=True)
    elif arch == 'densenet':
        model = getattr(models, "densenet121")(pretrained=True)

    # Re-build the model

    classifier = Network(checkpoint['input_size'],
                         checkpoint['output_size'],
                         checkpoint['hidden_layers'],
                         checkpoint['drop'])

    model.classifier = classifier


    model.load_state_dict(checkpoint['state_dict'])

    model.optimizer = checkpoint['optimizer']
    model.epochs = checkpoint['epochs']
    model.learning_rate = checkpoint['learning_rate']
    model.class_to_idx = checkpoint['class_to_idx']
    model.to(device)
    return model
