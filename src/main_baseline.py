import os
import random
import argparse
import torch
import numpy as np

import graph_net
import utils
import trainer

from logger import configure

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Graph Curriculum Domain Adaptaion')
# model args
parser.add_argument('--method', type=str, default='CDAN', choices=['CDAN', 'CDAN+E'])
parser.add_argument('--encoder', type=str, default='ResNet50', choices=['ResNet18', 'ResNet50'])
parser.add_argument('--hyper_embed_dim', type=int, default=128)
parser.add_argument('--hyper_hidden_dim', type=int, default=512)
parser.add_argument('--hyper_hidden_num', type=int, default=1)
parser.add_argument('--prompt_num', type=int, default=0)
parser.add_argument('--rand_proj', type=int, default=512, help='random projection dimension')
parser.add_argument('--edge_features', type=int, default=128, help='graph edge features dimension')
parser.add_argument('--save_models', action='store_false', help='whether to save encoder, mlp and gnn models')
# dataset args
parser.add_argument('--dataset', type=str, default='MTRS', choices=['MTRS', 'office31', 'office-home', 'pacs',
                                                                        'domain-net'], help='dataset used')
parser.add_argument('--source', default='AList', help='name of source domain')
parser.add_argument('--target', default='NList_PList_RList_UList', help='names of target domains')
# parser.add_argument('--target', nargs='+', default=['dslr', 'webcam'], help='names of target domains')
parser.add_argument('--data_root', type=str, default='/data/ztjiaweixu/Code/ZTing', help='path to dataset root')
# training args
parser.add_argument('--target_inner_iters', type=int, default=1, help='number of inner steps in train_target')
parser.add_argument('--target_iters', type=int, default=100, help='number of fine-tuning iters on pseudo target')
parser.add_argument('--source_iters', type=int, default=100, help='number of source pre-train iters')
parser.add_argument('--adapt_iters', type=int, default=100, help='number of iters for a curriculum adaptation')
parser.add_argument('--finetune_iters', type=int, default=10, help='number of fine-tuning iters')
parser.add_argument('--test_interval', type=int, default=100, help='interval of two continuous test phase')
parser.add_argument('--output_dir', type=str, default='~/results/ZTing', help='output directory')
parser.add_argument('--source_batch', type=int, default=16)
parser.add_argument('--target_batch', type=int, default=16)
parser.add_argument('--test_batch', type=int, default=32)
parser.add_argument('--same_id_adapt', type=int, default=1, choices=[0, 1])
parser.add_argument('--random_domain', type=int, default=0, choices=[0, 1])
parser.add_argument('--unable_gnn', type=int, default=0, choices=[0, 1])
# optimization args
parser.add_argument('--lr_type_hyper', type=str, default='none', choices=['none', 'inv'], help='type of learning rate scheduler')
parser.add_argument('--lr_type', type=str, default='none', choices=['none', 'inv'], help='type of learning rate scheduler')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
parser.add_argument('--lambda_edge', default=0.5, type=float, help='edge loss weight')
parser.add_argument('--lambda_node', default=0.5, type=float, help='node classification loss weight')
parser.add_argument('--lambda_adv', default=1.0, type=float, help='adversarial loss weight')
parser.add_argument('--threshold_progressive', type=float, default=0.7, help='threshold for progressive inference')
parser.add_argument('--threshold_target', type=float, default=0.9, help='threshold for pseudo labels in update target domain')
parser.add_argument('--threshold', type=float, default=0.7, help='threshold for pseudo labels')
parser.add_argument('--seed', type=int, default=2023, help='random seed for training')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloaders')
# other args
parser.add_argument("--alg_type", type=str, default=os.path.basename(__file__)[5:-3])


def main(args):
    # fix random seed
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    assert args.source not in args.target.split("_"), 'Source domain can not be one of the target domains'

    # create train configurations
    config = utils.build_config(args)
    # prepare data
    dsets, dset_loaders = utils.build_data(config)
    # set base network
    net_config = config['encoder']
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.to(DEVICE)
    utils.write_logs(config, str(base_network))
    # set GNN classifier
    classifier_gnn = graph_net.ClassifierGNN(in_features=base_network.bottleneck.out_features,
                                             edge_features=config['edge_features'],
                                             nclasses=config['encoder']['params']['class_num'],
                                             device=DEVICE)
    classifier_gnn = classifier_gnn.to(DEVICE)
    utils.write_logs(config, str(classifier_gnn))

    # train on source domain and compute domain inheritability
    log_str = '==> Step 1: Pre-training on the source dataset ...'
    utils.write_logs(config, log_str)

    logger = configure(config["output_path"], ["csv"], f"_step1")
    base_network, classifier_gnn = trainer.train_source(config, base_network, classifier_gnn, dset_loaders, logger)
    del logger
    log_str = '==> Finished pre-training on source!\n'
    utils.write_logs(config, log_str)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

