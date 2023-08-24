import os
import random
import argparse
import torch
import numpy as np

import graph_net
import utils
import trainer


#TODO 0 for source and target before step 4 1, 2.. for target in step 4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Graph Curriculum Domain Adaptaion')
# model args
parser.add_argument('--method', type=str, default='CDAN', choices=['CDAN', 'CDAN+E'])
parser.add_argument('--encoder', type=str, default='ResNet50', choices=['ResNet18', 'ResNet50'])
parser.add_argument('--use_hyper', type=int, default=1, choices=[0, 1])
parser.add_argument('--hyper_embed_dim', type=int, default=64)
parser.add_argument('--hyper_hidden_dim', type=int, default=128)
parser.add_argument('--hyper_hidden_num', type=int, default=1)
parser.add_argument('--rand_proj', type=int, default=1024, help='random projection dimension')
parser.add_argument('--edge_features', type=int, default=128, help='graph edge features dimension')
parser.add_argument('--save_models', action='store_false', help='whether to save encoder, mlp and gnn models')
# dataset args
parser.add_argument('--dataset', type=str, default='office31', choices=['office31', 'office-home', 'pacs',
                                                                        'domain-net'], help='dataset used')
parser.add_argument('--source', default='amazon', help='name of source domain')
parser.add_argument('--target', default='dslr_webcam', help='names of target domains')
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
# optimization args
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
parser.add_argument('--lambda_edge', default=1., type=float, help='edge loss weight')
parser.add_argument('--lambda_node', default=0.3, type=float, help='node classification loss weight')
parser.add_argument('--lambda_adv', default=1.0, type=float, help='adversarial loss weight')
parser.add_argument('--threshold_progressive', type=float, default=0.7, help='threshold for progressive inference')
parser.add_argument('--threshold_target', type=float, default=0.7, help='threshold for pseudo labels in update target domain')
parser.add_argument('--threshold_source', type=float, default=0.9, help='threshold for pseudo labels in update source domain')
parser.add_argument('--threshold', type=float, default=0.9, help='threshold for pseudo labels')
parser.add_argument('--seed', type=int, default=2023, help='random seed for training')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloaders')
# other args
parser.add_argument("--alg_type", type=str, default=os.path.basename(__file__).rstrip(".py").lstrip("main_"))


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
                                             nclasses=base_network.fc.out_features,
                                             device=DEVICE)
    classifier_gnn = classifier_gnn.to(DEVICE)
    utils.write_logs(config, str(classifier_gnn))

    # train on source domain and compute domain inheritability
    log_str = '==> Step 1: Pre-training on the source dataset ...'
    utils.write_logs(config, log_str)

    base_network, classifier_gnn = trainer.train_source(config, base_network, classifier_gnn, dset_loaders)
    log_str = '==> Finished pre-training on source!\n'
    utils.write_logs(config, log_str)

    log_str = '==> Step 2: Curriculum learning ...'
    utils.write_logs(config, log_str)

    ######## Stage 1: find the closest target domain ##########
    temp_test_loaders = dict(dset_loaders['target_test'])
    max_inherit_domain = trainer.select_closest_domain(config, base_network, classifier_gnn, temp_test_loaders)

    # iterate over all domains
    for _ in range(len(config['data']['target']['name'])):
        log_str = '==> Starting the adaptation on {} ...'.format(max_inherit_domain)
        utils.write_logs(config, log_str)
        ######## Stage 2: adapt to the chosen target domain having the maximum inheritance/similarity ##########
        base_network, classifier_gnn = trainer.adapt_target(config, base_network, classifier_gnn,
                                                            dset_loaders, max_inherit_domain)
        log_str = '==> Finishing the adaptation on {}!\n'.format(max_inherit_domain)
        utils.write_logs(config, log_str)

        ######### Stage 3: obtain the target pseudo labels and upgrade source and target domain ##########
        trainer.upgrade_source_domain(config, max_inherit_domain, dsets,
                                      dset_loaders, base_network, classifier_gnn)

        ######### Sage 1: recompute target domain inheritability/similarity ###########
        # remove already considered domain
        del temp_test_loaders[max_inherit_domain]
        # find the maximum inheritability/similarity domain
        if len(temp_test_loaders.keys()) > 0:
            max_inherit_domain = trainer.select_closest_domain(config, base_network,
                                                                       classifier_gnn, temp_test_loaders)


    ######### Step 3: fine-tuning stage on source ###########
    log_str = '==> Step 3: Fine-tuning on pseudo-source dataset ...'
    utils.write_logs(config, log_str)
    config['source_iters'] = config['finetune_iters']
    base_network, classifier_gnn = trainer.train_source(config, base_network, classifier_gnn, dset_loaders)
    log_str = 'Finished training and evaluation on source!\n'
    utils.write_logs(config, log_str)

    # save models
    if args.save_models:
        torch.save(base_network.state_dict(), os.path.join(config['output_path'], 'base_network_source.pth'))
        torch.save(classifier_gnn.state_dict(), os.path.join(config['output_path'], 'classifier_gnn_source.pth'))

    ######### Step 4: fine-tuning stage on target ###########
    log_str = '==> Step 4: Fine-tuning on pseudo-target dataset ...'
    utils.write_logs(config, log_str)

    for name in config['data']['target']['name']:
        log_str = f'==> Update target domian label on {name} ...'
        utils.write_logs(config, log_str)

        utils.write_logs(config, f"Dataset: {name}, {len(dsets['target_train'][name])}")
        trainer.upgrade_target_domain(config, name, dsets, dset_loaders, base_network, classifier_gnn)
        utils.write_logs(config, f"Dataset: {name}, {len(dsets['target_train'][name])}")

        log_str = f'==> Change domian id on {name} ...'
        utils.write_logs(config, log_str)
        dsets["target_test"][name].set_domain_id(config["domain_id"][name])
        dset_loaders["target_test"][name].dataset.set_domain_id(config["domain_id"][name])

    for name in config['data']['target']['name']:
        log_str = f'==> Starting fine-tuning on {name}'
        utils.write_logs(config, log_str)
        train_target = trainer.train_target if config['target_inner_iters'] == 1 else trainer.train_target_v2
        base_network, classifier_gnn = train_target(config, base_network, classifier_gnn, dset_loaders, name)
        log_str = f'==> Finishing fine-tuning on {name}\n'
        utils.write_logs(config, log_str)

    # save models
    if args.save_models:
        torch.save(base_network.state_dict(), os.path.join(config['output_path'], 'base_network_target.pth'))
        torch.save(classifier_gnn.state_dict(), os.path.join(config['output_path'], 'classifier_gnn_target.pth'))

    ######### Step 5: progressive inference stage on target ###########
    log_str = '==> Step 5: Progressive Inference on target dataset ...'
    utils.write_logs(config, log_str)

    log_str = 'Starting progressive inference on target!'
    utils.write_logs(config, log_str)
    trainer.evaluate_progressive(0, config, base_network, classifier_gnn, dset_loaders["target_test"], dset_loaders["source"])
    log_str = 'Finished progressive inference on target!'
    utils.write_logs(config, log_str)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

