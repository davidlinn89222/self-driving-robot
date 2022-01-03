import os
import json
import torch
from tool import utils, preprocess, architecture
import numpy as np

def get_params(path):
    
    with open(os.path.join(path, 'params.json'), 'r') as f:
        params = json.load(f)
    
    return params


def get_test_loader(params):

    param = params['TREATMENT']

    if param == 'yes':
        loader = preprocess.data_loader(
            param, 1, [300,300], [224,224], [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    else:
        loader = preprocess.data_loader(param, 1)

    loader.define_preprocess()
    _, _, test_loader = loader.build_loader()

    images, labels = next(iter(test_loader)) # get the (width, height)
    _, h, w = images[0].shape

    return test_loader, (h, w)


def build_load_model(params, path, img_size):

    param = params['ARCHITECTURE']

    WEIGHT_PATH = os.path.join(path, 'training_checkpoints/weight.pth')

    if param == '1':
        model = architecture.CNN_Model1(img_size, 'model1')
    elif param == '2':
        model = architecture.CNN_Model2(img_size, 'model2')
    elif param == '3':
        model = architecture.CNN_Model3(img_size, 'model3')

    tmp = torch.load(WEIGHT_PATH, map_location=torch.device('cpu'))

    model.load_state_dict(tmp['model_state_dict'])

    return model

def predict(model, test_loader):
    predict = []
    true_label = []

    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        predict.append(preds)
        true_label.append(labels)
        
    return np.array(predict), np.array(true_label)

