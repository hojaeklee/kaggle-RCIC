import os
import argparse
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from data_loader.data_loaders import RCICDataLoader
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import torch
import torch.nn.functional as F

def main(config):
    """Extracts and saves embedding for negative controls from trained models."""
    logger = config.get_logger('test')

    # setup data_loader instances
    negctrl1_dataloader = RCICDataLoader(data_dir = "./data/raw", batch_size = 8, shuffle = False, validation_split = 0.0, num_workers = 2, neg_ctrl = True, training = False, site = 1)
    negctrl2_dataloader = RCICDataLoader(data_dir = "./data/raw", batch_size = 8, shuffle = False, validation_split = 0.0, num_workers = 2, neg_ctrl = True, training = False, site = 2)

    # build feature model architecture
    feature_model = config.initialize('arch', module_arch)
    logger.info(feature_model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        feature_model = torch.nn.DataParallel(model)
    feature_model.load_state_dict(state_dict)

    # prepare model for extraction
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_model = feature_model.to(device)
    feature_model.eval()

    with torch.no_grad():
        negative_embeddings = []
        for i, data in enumerate(tqdm(zip(negctrl1_dataloader, negctrl2_dataloader))):
            negctrl1_data = data[0][0].to(device)
            negctrl2_data = data[1][0].to(device)
            output1, emb1 = feature_model(negctrl1_data)
            output2, emb2 = feature_model(negctrl2_data)
            emb1 = F.relu(emb1, inplace = True)
            emb2 = F.relu(emb2, inplace = True)
            emb1 = F.adaptive_avg_pool2d(emb1, (1, 1)).view(emb1.size(0), -1)
            emb2 = F.adaptive_avg_pool2d(emb2, (1, 1)).view(emb2.size(0), -1)

            negative_embeddings.append(emb1)
            negative_embeddings.append(emb2)

        negative_embeddings = torch.cat(negative_embeddings)

    save_dir = config.save_dir
    save_filename = '{}_neg_emb.pth'.format(os.path.basename(os.path.dirname(config.resume)))
    torch.save(negative_embeddings, os.path.join(save_dir, save_filename))
    logger.info("Saved negative embeddings to {}".format(os.path.join(save_dir, save_filename)))

if __name__ == "__main__":
    args = argparse.ArgumentParser(description = "PyTorch Template")
    args.add_argument("-r", "--resume", default = None, type=str, \
                      help = "path to latest checkpoint (default: None)")
    args.add_argument("-d", "--device", default = None, type = str, \
                      help = "indices of GPUs to enable (default: all)")
    config = ConfigParser(args)
    main(config)
