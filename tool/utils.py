import torch
import time
import copy
import os
import pandas as pd
import numpy as np

def train_model(model, data_loaders, criterion, optimizer, scheduler, device, run_folder_path, num_epochs=25):
    since = time.time()

    train_acc_history = []
    train_loss_history = []
    val_acc_history = []
    val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward (track history if only in train)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(data_loaders[phase].dataset)
            epoch_acc = epoch_acc.cpu().detach()

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # append the results
            if phase == 'val':
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
            else:
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    # Store checkpoints
    CHECKPOINT_DIR_PATH = os.path.join(run_folder_path, './training_checkpoints')

    if not os.path.exists(CHECKPOINT_DIR_PATH):
        os.makedirs(CHECKPOINT_DIR_PATH)

    PATH = os.path.join(CHECKPOINT_DIR_PATH, 'weight.pth')
    torch.save({'model_state_dict': best_model_wts}, PATH)

    # Store training_logger
    logger_path = os.path.join(run_folder_path, 'training_logger.csv')
    loss_metrics = np.array([
        train_loss_history, train_acc_history,
        val_loss_history, val_acc_history])
    loss_metrics_df = pd.DataFrame(loss_metrics.T, columns = ['train_loss', 'train_acc', 'val_loss', 'val_acc'])
    loss_metrics_df.to_csv(logger_path)

    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model
