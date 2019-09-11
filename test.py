import os
import argparse
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from data_loader.data_loaders import RCICDataLoader
# import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser

# Plate Leak implementation from https://www.kaggle.com/zaharch/keras-model-boosted-with-plates-leak
train_csv = pd.read_csv("./data/raw/train.csv")
test_csv = pd.read_csv("./data/raw/test.csv")
all_test_exp = test_csv.experiment.unique()
exp_to_group = [3, 1, 0, 0, 0, 0, 2, 2, 3, 0, 0, 3, 1, 0, 0, 0, 2, 3]

plate_groups = np.zeros((1108, 4), int)
for sirna in range(1108):
    grp = train_csv.loc[train_csv.sirna==sirna,:].plate.value_counts().index.values
    assert len(grp) == 3
    plate_groups[sirna,0:3] = grp
    plate_groups[sirna,3] = 10 - grp.sum()

def select_plate_group(pp_mult, idx):
    sub_test = test_csv.loc[test_csv.experiment == all_test_exp[idx], :]
    assert len(pp_mult) == len(sub_test)
    mask = np.repeat(plate_groups[np.newaxis, :, exp_to_group[idx]], len(pp_mult), axis = 0) != \
            np.repeat(sub_test.plate.values[:, np.newaxis], 1108, axis = 1)
    pp_mult[mask] = 0
    return pp_mult

def main(config, is_cropped=False, four_plates=False):
    logger = config.get_logger('test')

    # setup data_loader instances
    if is_cropped:
        data_dir = "./data/cropped"
        batch_size = 1
    else:
        data_dir = "./data/raw"
        batch_size = 8
    site1_data_loader = RCICDataLoader(data_dir = data_dir, batch_size = batch_size, 
                                       shuffle = False, validation_split = 0.0, 
                                       num_workers = 12, training = False, site = 1,
                                       is_cropped=is_cropped, four_plates=four_plates) 
    site2_data_loader = RCICDataLoader(data_dir = data_dir, batch_size = batch_size, 
                                       shuffle = False, validation_split = 0.0, 
                                       num_workers = 12, training = False, site = 2, 
                                       is_cropped=is_cropped)
    
    # build model architecture
    model = config.initialize('arch', module_arch)
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        # preds = np.empty(0)
        predicted = []
        for i, data in enumerate(tqdm(zip(site1_data_loader, site2_data_loader))):
            site1_data = data[0][0].view(-1,6,64,64).to(device)
            site2_data = data[1][0].view(-1,6,64,64).to(device)
            output1 = model(site1_data)
            output2 = model(site2_data)
            if is_cropped:
                output = torch.exp(output1).sum(dim=0) + torch.exp(output2).sum(dim=0)
            else:
                output = (torch.exp(output1) + torch.exp(output2))

            #
            # save sample images, or do something with output here
            #
            # idx = output.max(dim = -1)[1].cpu().numpy()
            # preds = np.append(preds, idx, axis = 0)
            predicted.append(output)

    # predicted = np.stack(predicted).squeeze()
    predicted = torch.cat(predicted)
    predicted = predicted.cpu().numpy().squeeze()

    for idx in range(len(all_test_exp)):
        indices = (test_csv.experiment == all_test_exp[idx])
        preds = predicted[indices, :].copy()
        preds = select_plate_group(preds, idx)
        test_csv.loc[indices, 'sirna'] = preds.argmax(1)

    test_csv['sirna'] = test_csv['sirna'].astype(int)
    save_dir = config.save_dir
    save_filename = '{}_submission.csv'.format(os.path.basename(os.path.dirname(config.resume)))
    test_csv.to_csv(os.path.join(save_dir, save_filename), index = False, columns = ['id_code', 'sirna'])
    logger.info("Saved submission file to {}".format(os.path.join(save_dir, save_filename)))

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-cr', '--is_cropped', dest= 'is_cropped', default=False, action='store_true',
                      help= ' running on cropped data (opts: True, False)') 
    args.add_argument('-fp', '--four_plates', dest= 'four_plates', default=False, action='store_true',
                      help= ' use plate specific loss (opts: True, False)') 
    config = ConfigParser(args)
    arguments = args.parse_args()
    main(config, arguments.is_cropped, arguments.four_plates)
