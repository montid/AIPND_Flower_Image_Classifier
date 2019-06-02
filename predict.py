"""
This script will use  a trained network (checkpoint file) to predict the class for an input image
It returns the predicted class along with the class probability.

Required Inputs:    - image_path: Input Image's path
                    - checkpoint: checkpoint.pth path
Optional inputs:    - topk: top K most likely classes (default = 3)
                    - category_names: json file path (mapping classes to real classes)
                    - gpu: gpu ON (default = cpu)
"""
#Imports here
from nn_model import load_model_checkpoint
from myutils import predict, print_results
import argparse

parser = argparse.ArgumentParser(description='Predicts flower name for an input image')
parser.add_argument('image_path', type=str)
parser.add_argument('checkpoint', type=str)
parser.add_argument('--topk', help='Top K most likely classes - default=3', type=int, default=3)
parser.add_argument('--category_names', help='Map classes to real names - default="./cat_to_name.json"', type=str, default="./cat_to_name.json")
parser.add_argument('--gpu', help='Use GPU for training is strongly suggested - default=GPU', action='store_true')


# set input args to variables
parcy = parser.parse_args()
image_path = parcy.image_path
checkpoint = parcy.checkpoint
topk = parcy.topk
category_names = parcy.category_names
isgpu = parcy.gpu


# loading the pre trained model checkpoint
model = load_model_checkpoint(checkpoint, isgpu)


# Output the predicted class and top probabilities
top_probs, top_classes, top_flowers = predict(image_path, model, topk, isgpu, category_names)
#print(top_probs)
#print(top_classes)
#print(top_flowers)

# print results
print_results(checkpoint, image_path, topk, category_names, isgpu, top_probs, top_flowers)
