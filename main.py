import argparse
import json
import torch
from torch import nn
from tool import utils, preprocess, architecture
import os
import logging

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

logging.basicConfig(level=logging.INFO, filename='./log.txt', filemode='w',
	format='[%(asctime)s %(levelname)-1s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def main():

    ### Arguements Parsing ###
    parser = argparse.ArgumentParser()
    parser.add_argument('ARCHITECTURE', help = 'Please specify the architeture of convolutional neural network (1/2/3)')
    parser.add_argument('TREATMENT', help = 'Please specify what perform data preprocessing ? (1: dncrop 2: normalization, 3: mixed)')
    args = parser.parse_args()

    ### Mapping user input ###
    if args.ARCHITECTURE == '1':
        MODEL_SPEC = 'Model_1_layer'
    elif args.ARCHITECTURE == '2':
        MODEL_SPEC = 'Model_3_layer_32_64_128'
    elif args.ARCHITECTURE == '3':
        MODEL_SPEC = 'Model_3_layer_64_128_256'

    if args.TREATMENT == '1':
        PREPROC_SPEC = 'dncrop'
        json_file = open('dncrop.json')
    elif args.TREATMENT == '2':
        PREPROC_SPEC = 'norm'
        json_file = open('normalization.json')
    elif args.TREATMENT == '3':
        PREPROC_SPEC = 'mixed'
        json_file = open('mixed.json')

    ### Create RUN folder with the combination of model and preprocess specifications ###
    RUN_DIR_PATH = './RUN'
    RUN_SUBDIR_PATH = os.path.join(RUN_DIR_PATH, MODEL_SPEC + '_' + PREPROC_SPEC)

    if not os.path.exists(RUN_SUBDIR_PATH):
        os.makedirs(RUN_SUBDIR_PATH)

    ### Dump user inputs for further analysis ###
    params = {"ARCHITECTURE": args.ARCHITECTURE, "TREATMENT": args.TREATMENT}
    with open(os.path.join(RUN_SUBDIR_PATH, 'params.json'), 'w') as f:
        json.dump(params, f)

    logging.info(f'Executed main file: #({args.TREATMENT}, {args.ARCHITECTURE})')

    ### Define batch_size (machine-dependent) ###
    batch_size = 64 # ...

    ### Build Dataloader ###
    preprocess_para = json.load(json_file)
    loader = preprocess.data_loader(args.TREATMENT, batch_size, preprocess_para)

    loader.define_preprocess()
    train_loader, valid_loader, test_loader = loader.build_loader()

    data_loaders = {'train': train_loader, 'val': valid_loader}

    images, labels = next(iter(test_loader))
    c, h, w = images[0].shape
    img_size = (h, w)

    logging.info(f'Built dataloader with image size #({c}, {h}, {w})')

    ### Build Model ###
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu") # cuda:0 is dead in my server

    if args.ARCHITECTURE == '1':
        model = architecture.CNN_Model1(img_size, MODEL_SPEC)
    elif args.ARCHITECTURE == '2':
        model = architecture.CNN_Model2(img_size, MODEL_SPEC)
    elif args.ARCHITECTURE == '3':
        model = architecture.CNN_Model3(img_size, MODEL_SPEC)

    print(model)

    model = model.to(device)

    logging.info('Built model')

    ### Build Optimizer ###

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    logging.info('Built optimizer')

    ### Training the model ###

    logging.info('Began to train the model')

    utils.train_model(model, data_loaders, criterion, optimizer, scheduler, device, RUN_SUBDIR_PATH, num_epochs=2)

    logging.info('Training ended')

if __name__ == '__main__':
    main()
