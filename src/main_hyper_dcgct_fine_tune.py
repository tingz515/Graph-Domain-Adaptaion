import os
import random
import argparse
import torch
import json
import numpy as np

import graph_net
import utils
import trainer

from logger import configure, save_json

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Graph Curriculum Domain Adaptaion')
# model args
parser.add_argument('--method', type=str, default='CDAN', choices=['CDAN', 'CDAN+E'])
parser.add_argument('--encoder', type=str, default='ResNet50', choices=['ResNet18', 'ResNet50'])
parser.add_argument('--multi_mlp', type=int, default=0, choices=[0, 1])
parser.add_argument('--use_hyper', type=int, default=1, choices=[0, 1])
parser.add_argument('--hyper_embed_dim', type=int, default=128)
parser.add_argument('--hyper_hidden_dim', type=int, default=512)
parser.add_argument('--hyper_hidden_num', type=int, default=1)
parser.add_argument('--embedding_init', type=str, default='sphere', choices=['sphere', 'xavier', 'uniform'])
parser.add_argument('--prompt_num', type=int, default=0)
parser.add_argument('--rand_proj', type=int, default=512, help='random projection dimension')
parser.add_argument('--edge_features', type=int, default=128, help='graph edge features dimension')
parser.add_argument('--save_models', action='store_false', help='whether to save encoder, mlp and gnn models')
# dataset args
parser.add_argument('--dataset', type=str, default='MTRS', choices=['MTRS', 'office31', 'office-home', 'pacs',
                                                                        'domain-net'], help='dataset used')
parser.add_argument('--source', default='AList', help='name of source domain')
parser.add_argument('--target', default='NList_PList_UList_RList', help='names of target domains')
# parser.add_argument('--target', nargs='+', default=['dslr', 'webcam'], help='names of target domains')
parser.add_argument('--data_root', type=str, default='/apdcephfs/share_1563664/ztjiaweixu/datasets/dcgct', help='path to dataset root')
# training args
parser.add_argument('--target_inner_iters', type=int, default=1, help='number of inner steps in train_target')
parser.add_argument('--target_iters', type=int, default=100, help='number of fine-tuning iters on pseudo target')
parser.add_argument('--source_iters', type=int, default=100, help='number of source pre-train iters')
parser.add_argument('--adapt_iters', type=int, default=100, help='number of iters for a curriculum adaptation')
parser.add_argument('--finetune_iters', type=int, default=10, help='number of fine-tuning iters')
parser.add_argument('--test_interval', type=int, default=100, help='interval of two continuous test phase')
parser.add_argument('--output_dir', type=str, default='./results/ZTing', help='output directory')
parser.add_argument('--source_batch', type=int, default=16)
parser.add_argument('--target_batch', type=int, default=16)
parser.add_argument('--test_batch', type=int, default=32)
parser.add_argument('--same_id_adapt', type=int, default=1, choices=[0, 1])
parser.add_argument('--random_domain', type=int, default=0, choices=[0, 1])
parser.add_argument('--unable_gnn', type=int, default=0, choices=[0, 1])
parser.add_argument('--finetune_light', type=int, default=1, choices=[0, 1])
parser.add_argument('--distill_light', type=int, default=1, choices=[0, 1])
parser.add_argument('--mlp_pseudo', type=int, default=0, choices=[0, 1])
# optimization args
parser.add_argument('--lr_type_hyper', type=str, default='none', choices=['none', 'inv'], help='type of learning rate scheduler')
parser.add_argument('--lr_type', type=str, default='none', choices=['none', 'inv'], help='type of learning rate scheduler')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
parser.add_argument('--lambda_edge', default=0.3, type=float, help='edge loss weight')
parser.add_argument('--lambda_node', default=0.3, type=float, help='node classification loss weight')
parser.add_argument('--lambda_adv', default=1.0, type=float, help='adversarial loss weight')
parser.add_argument('--lambda_mlp', default=1.0, type=float, help='mlp loss weight')
parser.add_argument('--lambda_distill', default=1.0, type=float, help='distillation loss weight')
parser.add_argument('--threshold_progressive', type=float, default=0.7, help='threshold for progressive inference')
parser.add_argument('--threshold_target', type=float, default=0.9, help='threshold for pseudo labels in update target domain')
parser.add_argument('--threshold', type=float, default=0.7, help='threshold for pseudo labels')
parser.add_argument('--seed', type=int, default=0, help='random seed for training')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloaders')
# other args
parser.add_argument('--eval_only', type=int, default=0, help="evaluation mode")
parser.add_argument("--alg_type", type=str, default=os.path.basename(__file__)[5:-3])
parser.add_argument('--checkpoint_dir', type=str, default='/apdcephfs/share_1563664/ztjiaweixu/zting/2024051102', help='output directory')


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
    if config['unable_gnn']:
        classifier_gnn.eval()
    utils.write_logs(config, str(classifier_gnn))

    base_network.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'base_network_source.pth')))
    classifier_gnn.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'classifier_gnn_source.pth')))
    log_str = f'Loaded pre-trained models from {args.checkpoint_dir}\n'
    utils.write_logs(config, log_str)
    mlp_accuracy_dict, gnn_accuracy_dict = trainer.evaluate(0, config, base_network, classifier_gnn, dset_loaders['target_test'])
    save_json(os.path.join(config['output_path'], f'source_eval_result.json'), {"mlp": mlp_accuracy_dict, "gnn": gnn_accuracy_dict})

    ######### Step 4: fine-tuning stage on target ###########
    log_str = '==> Step 4: Fine-tuning on pseudo-target dataset ...'
    utils.write_logs(config, log_str)

    pseudo_res_all = {}
    for name in config['data']['target']['name']:
        log_str = f'==> Update target domian label on {name} ...'
        utils.write_logs(config, log_str)

        utils.write_logs(config, f"Dataset: {name}, {len(dsets['target_train'][name])}")
        pseudo_res = trainer.upgrade_target_domain(config, name, dsets, dset_loaders, base_network, classifier_gnn)
        pseudo_res_all[name] = pseudo_res
        utils.write_logs(config, f"Dataset: {name}, {len(dsets['target_train'][name])}")

        log_str = f'==> Change domian id on {name} ...'
        utils.write_logs(config, log_str)
        dsets["target_test"][name].set_domain_id(config["domain_id"][name])
        dset_loaders["target_test"][name].dataset.set_domain_id(config["domain_id"][name])
    pseudo_res_all = trainer.average_info(pseudo_res_all)
    save_json(os.path.join(config['output_path'], f'pseudo_label_result.json'), pseudo_res_all)

    result_dict_all = {}
    for name in config['data']['target']['name']:
        log_str = f'==> Starting fine-tuning on {name}'
        utils.write_logs(config, log_str)
        base_network.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'base_network_source.pth')))
        classifier_gnn.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'classifier_gnn_source.pth')))

        logger = configure(config["output_path"], ["csv"], f"_step4_{name}")
        train_target = trainer.train_target if config['target_inner_iters'] == 1 else trainer.train_target_v2
        base_network, classifier_gnn = train_target(config, base_network, classifier_gnn, dset_loaders, name, logger)
        del logger
        log_str = f'==> Finishing fine-tuning on {name}\n'
        utils.write_logs(config, log_str)

        # save models
        if args.save_models:
            torch.save(base_network.state_dict(), os.path.join(config['output_path'], f'base_network_target_{name}.pth'))
            torch.save(classifier_gnn.state_dict(), os.path.join(config['output_path'], f'classifier_gnn_target_{name}.pth'))

        log_str = f'==> Progressive Inference on {name}'
        utils.write_logs(config, log_str)

        result_dict, _ = trainer.evaluate_DisCo(config, base_network, classifier_gnn, name, dset_loaders["target_test"][name], dset_loaders["source"])
        save_json(os.path.join(config['output_path'], f'progressive_inference_{name}.json'), result_dict)
        log_str = '==> Finished progressive inference on target!\n'
        utils.write_logs(config, log_str)
        result_dict_all[name] = result_dict
    result_dict_all = trainer.average_info(result_dict_all)
    save_json(os.path.join(config['output_path'], 'progressive_inference_all.json'), result_dict_all)


if __name__ == "__main__":
    args = parser.parse_args()
    for file_name in os.listdir(args.checkpoint_dir):
        if f"{args.source}_rest_{args.seed}" in file_name:
            args.checkpoint_dir = os.path.join(args.checkpoint_dir, file_name)
            break
    with open(os.path.join(args.checkpoint_dir, "params.json"), "r") as f:
        config = json.load(f)
    for key, value in config.items():
        if key in args:
            try:
                setattr(args, key, eval(value))
                print(f"Set {key} to {value}")
            except:
                pass
    main(args)

