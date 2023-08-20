import os
import json
import time
import torch
from torch.utils.data import DataLoader

import networks
import preprocess
from preprocess import ImageList

def dump_params(config):
    dump_params = config.copy()
    for k, v in dump_params.items():
        dump_params[k] = str(v)
        write_logs(config, f"{k}: {v}")
    with open(os.path.join(config['output_path'], "params.json"), "wt") as f:
        f.write(json.dumps(dump_params, indent=4) + "\n")
        f.flush()


def inv_lr_scheduler(optimizer, iter_num, gamma, power, lr=0.001, weight_decay=0.0005):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = weight_decay * param_group['decay_mult']
        i += 1
    return optimizer


def build_config(args):
    config = {
        'method': args.method,
        # 'ndomains': 2,
        'output_path': args.output_dir,
        'threshold': args.threshold,
        'edge_features': args.edge_features,
        'source_iters': args.source_iters,
        'finetune_iters': args.finetune_iters,
        'adapt_iters': args.adapt_iters,
        'test_interval': args.test_interval,
        'num_workers': args.num_workers,
        'lambda_edge': args.lambda_edge,
        'lambda_node': args.lambda_node,
        'lambda_adv': args.lambda_adv,
        'random_dim': args.rand_proj,
        'use_cgct_mask': args.use_cgct_mask if 'use_cgct_mask' in args else False,
    }
    if args.alg_type in ['hyper_dcgct']:
        config['target_iters'] = args.target_iters
        config['same_id_adapt'] = args.same_id_adapt
    # preprocessing params
    config['prep'] = {
        'test_10crop': False,
        'params':
            {'resize_size': 256,
             'crop_size': 224,
             },
    }
    # backbone params
    config['encoder'] = {
        'name': networks.ResNetFc,
        'params': {'resnet_name': args.encoder,
                   'use_bottleneck': True,
                   'bottleneck_dim': 256,
                   'new_cls': True,
                   "hyper_embed_dim": args.hyper_embed_dim,
                   "hyper_hidden_dim": args.hyper_hidden_dim,
                   "hyper_hidden_num": args.hyper_hidden_num,
                   'domain_num': len(args.target.split("_")) + 1,
                   'use_hyper': args.use_hyper,
                   },
    }
    # optimizer params
    config['optimizer'] = {
        'type': torch.optim.SGD,
        'optim_params': {
            'lr': args.lr,
             'momentum': 0.9,
             'weight_decay': args.wd,
             'nesterov': True,
             },
        'lr_type': 'inv',
        'lr_param': {
            'lr': args.lr,
            'gamma': 0.001,
            'power': 0.75,
        },
    }
    # dataset params
    config['dataset'] = args.dataset
    config['data_root'] = os.path.expanduser(os.path.join(args.data_root, args.dataset))
    config['ndomains'] = len(args.target.split("_"))
    config['data'] = {
        'source': {
            'name': args.source,
            'batch_size': args.source_batch,
        },
        'target': {
            'name': args.target.split("_"),
            'batch_size': args.target_batch,
        },
        'test': {
            'name': args.target.split("_"),
            'batch_size': args.test_batch,
        },
    }
    config["domain_id"] = {name: i + 1 for i, name in enumerate(args.target.split("_"))}
    # set number of classes
    if config['dataset'] == 'office31':
        config['encoder']['params']['class_num'] = 31
        config['data']['image_list_root'] = 'data/office/'
    elif config['dataset'] == 'office-home':
        config['encoder']['params']['class_num'] = 65
        config['data']['image_list_root'] = 'data/office-home/'
    elif config['dataset'] == 'domain-net':
        config['encoder']['params']['class_num'] = 345
        config['data']['image_list_root'] = 'data/domain-net/'
    elif config['dataset'] == 'pacs':
        config['encoder']['params']['class_num'] = 7
        config['data']['image_list_root'] = 'data/pacs/'
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')

    # set pre-processing transformations
    config['prep'] = {
        'source': preprocess.image_train(**config['prep']['params']),
        'target': preprocess.image_train(**config['prep']['params']),
        'test': preprocess.image_test(**config["prep"]['params']),
    }
    # create output folder and log file
    time_tag = time.strftime("%Y%m%d%H%M%S", time.localtime())
    output_file =  f"{args.method}_{args.dataset}_{args.source}_rest_{args.seed}_{time_tag}"
    config['output_path'] = os.path.expanduser(os.path.join(args.output_dir, output_file))
    if not os.path.exists(config['output_path']):
        os.system('mkdir -p '+config['output_path'])
    config['out_file'] = open(os.path.join(config['output_path'], 'log.txt'), 'w')

    # print pout config values
    config['out_file'].write(str(config)+'\n')
    config['out_file'].flush()

    dump_params(config)
    return config


def build_data(config):
    dsets = {
        'target_train': {},
        'target_test': {},
    }
    dset_loaders = {
        'target_train': {},
        'target_test': {},
    }
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    target_bs = data_config["target"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]

    # source dataloader
    dsets['source'] = ImageList(image_root=config['data_root'], image_list_root=data_config['image_list_root'],
                                dataset=data_config['source']['name'], transform=config['prep']["source"],
                                domain_label=0, domain_id=0, dataset_name=config['dataset'], split='train')
    dset_loaders['source'] = DataLoader(dsets['source'], batch_size=train_bs, shuffle=True,
                                        num_workers=config['num_workers'], drop_last=True, pin_memory=False)

    # target dataloader
    for i, dset_name in enumerate(sorted(data_config['target']['name'])):
        domain_id = 0 if config['same_id_adapt'] else config["domain_id"][dset_name]
        # create train and test datasets for a target domain
        dsets['target_train'][dset_name] = ImageList(image_root=config['data_root'],
                                                     image_list_root=data_config['image_list_root'],
                                                     dataset=dset_name, transform=config['prep']['target'],
                                                     domain_label=1, domain_id=domain_id, dataset_name=config['dataset'], split='train',
                                                     use_cgct_mask=config['use_cgct_mask'])
        dsets['target_test'][dset_name] = ImageList(image_root=config['data_root'],
                                                    image_list_root=data_config['image_list_root'],
                                                    dataset=dset_name, transform=config['prep']['test'],
                                                    domain_label=1, domain_id=domain_id, dataset_name=config['dataset'], split='test',
                                                    use_cgct_mask=config['use_cgct_mask'])
        # create train and test dataloaders for a target domain
        dset_loaders['target_train'][dset_name] = DataLoader(dataset=dsets['target_train'][dset_name],
                                                             batch_size=target_bs, shuffle=True,
                                                             num_workers=config['num_workers'], drop_last=True)
        dset_loaders['target_test'][dset_name] = DataLoader(dataset=dsets['target_test'][dset_name],
                                                            batch_size=test_bs, num_workers=config['num_workers'],
                                                            pin_memory=False)
    return dsets, dset_loaders


def write_logs(config, log_str):
    config['out_file'].write(log_str + '\n')
    config['out_file'].flush()
    print(log_str)
