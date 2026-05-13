from config import get_config, print_usage
config, unparsed = get_config()
import os
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
import torch.utils.data
import sys
from data import collate_fn, CorrespondencesDataset
from hatmatchplus import HATNetPlus as Model
from train import train
from test import test

print("======================== HAT-Match+ ========================")

def create_log_dir(config):
    if not os.path.isdir(config.log_base):
        os.makedirs(config.log_base)
    result_path = config.log_base
    for sub in ['', '/train', '/valid', '/test']:
        p = result_path + sub
        if not os.path.isdir(p):
            os.makedirs(p)
    if os.path.exists(result_path + '/config.th'):
        print('warning: will overwrite config file')
    torch.save(config, result_path + '/config.th')
    config.log_path = result_path + '/train'

def main(config):
    model = Model(config)
    print('Total parameters:', sum(p.numel() for p in model.parameters()))

    if config.run_mode == "train":
        create_log_dir(config)
        train_dataset = CorrespondencesDataset(config.data_tr, config)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.train_batch_size, shuffle=True,
            num_workers=16, pin_memory=False, collate_fn=collate_fn)
        valid_dataset = CorrespondencesDataset(config.data_va, config)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=config.train_batch_size, shuffle=False,
            num_workers=8, pin_memory=False, collate_fn=collate_fn)
        print('Start training HAT-Match+ ...')
        train(model, train_loader, valid_loader, config)

    elif config.run_mode == "test":
        test_dataset = CorrespondencesDataset(config.data_te, config)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False,
            num_workers=8, pin_memory=False, collate_fn=collate_fn)
        test(test_loader, model, config)

if __name__ == "__main__":
    config, unparsed = get_config()
    if len(unparsed) > 0:
        print_usage()
        exit(1)
    main(config)
