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


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    """
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=8,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )
    """
    site1_data_loader = RCICDataLoader(data_dir = "./data/raw", batch_size = 8, shuffle = False, validation_split = 0.0, num_workers = 2, training = False, site = 1) 
    site2_data_loader = RCICDataLoader(data_dir = "./data/raw", batch_size = 8, shuffle = False, validation_split = 0.0, num_workers = 2, training = False, site = 2)

    # build model architecture
    model = config.initialize('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    # loss_fn = getattr(module_loss, config['loss'])
    # metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # total_loss = 0.0
    # total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        preds = np.empty(0)
        for i, data in enumerate(tqdm(zip(site1_data_loader, site2_data_loader))):
            site1_data = data[0][0].to(device)
            site2_data = data[1][0].to(device)
            output1 = model(site1_data)
            output2 = model(site2_data)
            output = 0.5 * (output1 + output2)

            #
            # save sample images, or do something with output here
            #
            idx = output.max(dim = -1)[1].cpu().numpy()
            preds = np.append(preds, idx, axis = 0)


            """
            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size
            """
    """        
    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)
    """

    submission = pd.read_csv('./data/raw/test.csv')
    submission['sirna'] = preds.astype(int)
    save_dir = config.save_dir
    save_filename = '{}_submission.csv'.format(os.path.basename(os.path.dirname(config.resume)))
    submission.to_csv(os.path.join(save_dir, save_filename), index = False, columns = ['id_code', 'sirna'])
    logger.info("Saved submission file to {}".format(os.path.join(save_dir, save_filename)))

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser(args)
    main(config)
