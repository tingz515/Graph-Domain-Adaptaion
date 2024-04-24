from collections import OrderedDict
import time
import torch
import torch.nn as nn
import numpy as np
import networks
import transfer_loss
from preprocess import ImageList, ConcatDataset
from torch.utils.data import DataLoader
import utils
from main_dcgct import DEVICE

def average_info(result_dict_all):
    mlp_t_accuracy_list, mlp_s_accuracy_list, gnn_accuracy_list = [], [], []
    progressive_mlp_accuracy_list, progressive_gnn_accuracy_list = [], []
    for val in result_dict_all.values():
        mlp_t_accuracy_list.append(val['mlp_t_accuracy'])
        mlp_s_accuracy_list.append(val['mlp_s_accuracy'])
        gnn_accuracy_list.append(val['gnn_accuracy'])
        progressive_mlp_accuracy_list.append(val['progressive_mlp_accuracy'])
        progressive_gnn_accuracy_list.append(val['progressive_gnn_accuracy'])
    result_dict_all["avg"] = {
        "mlp_t_accuracy": np.mean(mlp_t_accuracy_list),
        "mlp_s_accuracy": np.mean(mlp_s_accuracy_list),
        "gnn_accuracy": np.mean(gnn_accuracy_list),
        "progressive_mlp_accuracy": np.mean(progressive_mlp_accuracy_list),
        "progressive_gnn_accuracy": np.mean(progressive_gnn_accuracy_list),
    }
    return result_dict_all

def evaluate_progressive_v2(n_iter, config, base_network, classifier_gnn, dset_name, test_loader, source_loader):
    base_network.eval()
    classifier_gnn.eval()
    len_train_source = len(source_loader)
    logits_mlp_t_all, logits_mlp_s_all, logits_gnn_all, confidences_all, labels_all = [], [], [], [], []
    with torch.no_grad():
        iter_test = iter(test_loader)
        domain_id = test_loader.dataset.domain_id
        for i in range(len(test_loader)):
            data = iter_test.next()
            inputs = data['img'].to(DEVICE)
            # forward pass
            feature, logits_mlp_t, logits_mlp_s = base_network.progressive_forward(inputs, domain_id)

            if i % len_train_source == 0:
                iter_source = iter(source_loader)

            batch_source = iter_source.next()
            inputs_source = batch_source['img'].to(DEVICE)
            features_source = base_network.large_feature(inputs_source)
            features_all = torch.cat((features_source, feature), dim=0)
            logits_gnn, _ = classifier_gnn(features_all)
            logits_gnn = logits_gnn[-len(inputs): ]

            logits_mlp_t_all.append(logits_mlp_t.cpu())
            logits_mlp_s_all.append(logits_mlp_s.cpu())
            logits_gnn_all.append(logits_gnn.cpu())
            labels_all.append(data['target'])
            confidences_all.append(nn.Softmax(dim=1)(logits_mlp_t_all[-1]).max(1)[0])

        logits_mlp_t = torch.cat(logits_mlp_t_all, dim=0)
        logits_mlp_s = torch.cat(logits_mlp_s_all, dim=0)
        logits_gnn = torch.cat(logits_gnn_all, dim=0)
        confidences = torch.cat(confidences_all, dim=0)
        labels = torch.cat(labels_all, dim=0)

        # predict class labels
        _, predict_mlp_t = torch.max(logits_mlp_t, 1)
        _, predict_mlp_s = torch.max(logits_mlp_s, 1)
        _, predict_gnn = torch.max(logits_gnn, 1)
        mlp_t_accuracy = torch.sum(predict_mlp_t == labels).item() / len(labels)
        mlp_s_accuracy = torch.sum(predict_mlp_s == labels).item() / len(labels)
        gnn_accuracy = torch.sum(predict_gnn == labels).item() / len(labels)

        # progressive predict class labels
        progressive_index = torch.where(confidences < config['threshold_progressive'])[0]
        progressive_ratio = len(progressive_index) / len(labels)
        if len(progressive_index) > 0:
            progressive_mlp_acc = torch.sum(predict_mlp_s[progressive_index] == labels[progressive_index]).item() / len(progressive_index)
            progressive_gnn_acc = torch.sum(predict_gnn[progressive_index] == labels[progressive_index]).item() / len(progressive_index)
        else:
            progressive_mlp_acc, progressive_gnn_acc = 0, 0

        # print out test accuracy for domain
        log_str = 'Dataset:%s ID:%s\tTest Accuracy target mlp %.4f\tTest Accuracy source mlp %.4f\tTest Accuracy gnn %.4f'\
                  % (dset_name, domain_id, mlp_t_accuracy * 100, mlp_s_accuracy * 100, gnn_accuracy * 100)
        config['out_file'].write(log_str + '\n')
        config['out_file'].flush()
        print(log_str)
        log_str = 'Dataset:%s ID:%s\tProgressive Ratio %.4f\tProgressive Accuracy source mlp %.4f\tProgressive Accuracy gnn %.4f'\
                  % (dset_name, domain_id, progressive_ratio, progressive_mlp_acc * 100, progressive_gnn_acc * 100)
        config['out_file'].write(log_str + '\n')
        config['out_file'].flush()
        print(log_str)
        result_dict = {
            'mlp_t_accuracy': mlp_t_accuracy,
            'mlp_s_accuracy': mlp_s_accuracy,
            'gnn_accuracy': gnn_accuracy,
            'progressive_mlp_accuracy': progressive_mlp_acc,
            'progressive_gnn_accuracy': progressive_gnn_acc,
            'progressive_ratio': progressive_ratio,
        }

    base_network.train()
    classifier_gnn.train()
    return result_dict


def evaluate_progressive(n_iter, config, base_network, classifier_gnn, target_test_dset_dict, source_loader):
    base_network.eval()
    classifier_gnn.eval()
    len_train_source = len(source_loader)
    mlp_t_accuracy_list, mlp_s_accuracy_list, gnn_accuracy_list = [], [], []
    progressive_mlp_accuracy_list, progressive_gnn_accuracy_list = [], []
    result_dict = {}
    for dset_name, test_loader in target_test_dset_dict.items():
        logits_mlp_t_all, logits_mlp_s_all, logits_gnn_all, confidences_all, labels_all = [], [], [], [], []
        with torch.no_grad():
            iter_test = iter(test_loader)
            domain_id = test_loader.dataset.domain_id
            for i in range(len(test_loader)):
                data = iter_test.next()
                inputs = data['img'].to(DEVICE)
                # forward pass
                feature, logits_mlp_t, logits_mlp_s = base_network.progressive_forward(inputs, domain_id)

                if i % len_train_source == 0:
                    iter_source = iter(source_loader)

                batch_source = iter_source.next()
                inputs_source = batch_source['img'].to(DEVICE)
                features_source = base_network.large_feature(inputs_source)
                features_all = torch.cat((features_source, feature), dim=0)
                logits_gnn, _ = classifier_gnn(features_all)
                logits_gnn = logits_gnn[-len(inputs): ]

                logits_mlp_t_all.append(logits_mlp_t.cpu())
                logits_mlp_s_all.append(logits_mlp_s.cpu())
                logits_gnn_all.append(logits_gnn.cpu())
                labels_all.append(data['target'])
                confidences_all.append(nn.Softmax(dim=1)(logits_mlp_t_all[-1]).max(1)[0])

        logits_mlp_t = torch.cat(logits_mlp_t_all, dim=0)
        logits_mlp_s = torch.cat(logits_mlp_s_all, dim=0)
        logits_gnn = torch.cat(logits_gnn_all, dim=0)
        confidences = torch.cat(confidences_all, dim=0)
        labels = torch.cat(labels_all, dim=0)

        # predict class labels
        _, predict_mlp_t = torch.max(logits_mlp_t, 1)
        _, predict_mlp_s = torch.max(logits_mlp_s, 1)
        _, predict_gnn = torch.max(logits_gnn, 1)
        mlp_t_accuracy = torch.sum(predict_mlp_t == labels).item() / len(labels)
        mlp_s_accuracy = torch.sum(predict_mlp_s == labels).item() / len(labels)
        gnn_accuracy = torch.sum(predict_gnn == labels).item() / len(labels)

        # progressive predict class labels
        progressive_index = torch.where(confidences < config['threshold_progressive'])[0]
        progressive_ratio = len(progressive_index) / len(labels)
        if len(progressive_index) > 0:
            progressive_mlp_acc = torch.sum(predict_mlp_s[progressive_index] == labels[progressive_index]).item() / len(progressive_index)
            progressive_gnn_acc = torch.sum(predict_gnn[progressive_index] == labels[progressive_index]).item() / len(progressive_index)
        else:
            progressive_mlp_acc, progressive_gnn_acc = 0, 0

        # print out test accuracy for domain
        log_str = 'Dataset:%s ID:%s\tTest Accuracy target mlp %.4f\tTest Accuracy source mlp %.4f\tTest Accuracy gnn %.4f'\
                  % (dset_name, domain_id, mlp_t_accuracy * 100, mlp_s_accuracy * 100, gnn_accuracy * 100)
        config['out_file'].write(log_str + '\n')
        config['out_file'].flush()
        print(log_str)
        log_str = 'Dataset:%s ID:%s\tProgressive Ratio %.4f\tProgressive Accuracy source mlp %.4f\tProgressive Accuracy gnn %.4f'\
                  % (dset_name, domain_id, progressive_ratio, progressive_mlp_acc * 100, progressive_gnn_acc * 100)
        config['out_file'].write(log_str + '\n')
        config['out_file'].flush()
        print(log_str)
        result_dict[dset_name] = {
            'mlp_t_accuracy': mlp_t_accuracy,
            'mlp_s_accuracy': mlp_s_accuracy,
            'gnn_accuracy': gnn_accuracy,
            'progressive_mlp_accuracy': progressive_mlp_acc,
            'progressive_gnn_accuracy': progressive_gnn_acc,
            'progressive_ratio': progressive_ratio,
        }

        # collect info
        mlp_t_accuracy_list.append(mlp_t_accuracy)
        mlp_s_accuracy_list.append(mlp_s_accuracy)
        gnn_accuracy_list.append(gnn_accuracy)
        progressive_mlp_accuracy_list.append(progressive_mlp_acc)
        progressive_gnn_accuracy_list.append(progressive_gnn_acc)

    # print out domains averaged accuracy
    mlp_t_accuracy_avg = sum(mlp_t_accuracy_list) / len(mlp_t_accuracy_list)
    mlp_s_accuracy_avg = sum(mlp_s_accuracy_list) / len(mlp_s_accuracy_list)
    gnn_accuracy_avg = sum(gnn_accuracy_list) / len(gnn_accuracy_list)
    log_str = 'iter: %d, Avg Accuracy MLP Target Classifier: %.4f, Avg Accuracy MLP Source Classifier: %.4f, Avg Accuracy GNN classifier: %.4f'\
              % (n_iter, mlp_t_accuracy_avg * 100., mlp_s_accuracy_avg * 100., gnn_accuracy_avg * 100.)
    config['out_file'].write(log_str + '\n')
    config['out_file'].flush()
    print(log_str)

    progressive_mlp_accuracy_avg = sum(progressive_mlp_accuracy_list) / len(progressive_mlp_accuracy_list)
    progressive_gnn_accuracy_avg = sum(progressive_gnn_accuracy_list) / len(progressive_gnn_accuracy_list)
    log_str = 'iter: %d, Avg Accuracy Progressive MLP Classifier: %.4f,Avg Accuracy GNN classifier: %.4f'\
              % (n_iter, progressive_mlp_accuracy_avg * 100., progressive_gnn_accuracy_avg * 100.)
    config['out_file'].write(log_str + '\n')
    config['out_file'].flush()
    print(log_str)
    result_dict['avg'] = {
        'mlp_t_accuracy_avg': mlp_t_accuracy_avg,
        'mlp_s_accuracy_avg': mlp_s_accuracy_avg,
        'gnn_accuracy_avg': gnn_accuracy_avg,
        'progressive_mlp_accuracy_avg': progressive_mlp_accuracy_avg,
        'progressive_gnn_accuracy_avg': progressive_gnn_accuracy_avg,
    }
    base_network.train()
    classifier_gnn.train()
    return result_dict


def evaluate(i, config, base_network, classifier_gnn, target_test_dset_dict):
    base_network.eval()
    classifier_gnn.eval()
    mlp_accuracy_list, gnn_accuracy_list = [], []
    mlp_accuracy_dict, gnn_accuracy_dict = {}, {}
    for dset_name, test_loader in target_test_dset_dict.items():
        test_res = eval_domain(config, test_loader, base_network, classifier_gnn)
        mlp_accuracy, gnn_accuracy = test_res['mlp_accuracy'], test_res['gnn_accuracy']
        mlp_accuracy_list.append(mlp_accuracy)
        gnn_accuracy_list.append(gnn_accuracy)
        mlp_accuracy_dict[dset_name] = mlp_accuracy
        gnn_accuracy_dict[dset_name] = gnn_accuracy
        # print out test accuracy for domain
        log_str = 'Dataset:%s ID:%s\tTest Accuracy mlp %.4f\tTest Accuracy gnn %.4f'\
                  % (dset_name, test_loader.dataset.domain_id, mlp_accuracy * 100, gnn_accuracy * 100)
        config['out_file'].write(log_str + '\n')
        config['out_file'].flush()
        print(log_str)

    # print out domains averaged accuracy
    mlp_accuracy_avg = sum(mlp_accuracy_list) / len(mlp_accuracy_list)
    gnn_accuracy_avg = sum(gnn_accuracy_list) / len(gnn_accuracy_list)
    log_str = 'iter: %d, Avg Accuracy MLP Classifier: %.4f, Avg Accuracy GNN classifier: %.4f'\
              % (i, mlp_accuracy_avg * 100., gnn_accuracy_avg * 100.)
    config['out_file'].write(log_str + '\n')
    config['out_file'].flush()
    print(log_str)
    base_network.train()
    classifier_gnn.train()
    return mlp_accuracy_dict, gnn_accuracy_dict


def eval_domain(config, test_loader, base_network, classifier_gnn, threshold=None):
    logits_mlp_all, logits_gnn_all, confidences_gnn_all, confidences_mlp_all, labels_all = [], [], [], [], []
    with torch.no_grad():
        iter_test = iter(test_loader)
        domain_id = test_loader.dataset.domain_id
        for _ in range(len(test_loader)):
            data = iter_test.next()
            inputs = data['img'].to(DEVICE)
            # forward pass
            feature, logits_mlp = base_network(inputs, domain_id)
            # check if number of samples is greater than 1
            if len(inputs) == 1:
                # gnn cannot handle only one sample ... use MLP instead
                # this can be encountered if len_dataset % test_batch == 1
                logits_gnn = logits_mlp
            else:
                logits_gnn, _ = classifier_gnn(feature)
            logits_mlp_all.append(logits_mlp.cpu())
            logits_gnn_all.append(logits_gnn.cpu())
            confidences_gnn_all.append(nn.Softmax(dim=1)(logits_gnn_all[-1]).max(1)[0])
            confidences_mlp_all.append(nn.Softmax(dim=1)(logits_mlp_all[-1]).max(1)[0])
            labels_all.append(data['target'])

    # concatenate data
    logits_mlp = torch.cat(logits_mlp_all, dim=0)
    logits_gnn = torch.cat(logits_gnn_all, dim=0)
    confidences_gnn = torch.cat(confidences_gnn_all, dim=0)
    confidences_mlp = torch.cat(confidences_mlp_all, dim=0)
    labels = torch.cat(labels_all, dim=0)
    # predict class labels
    _, predict_mlp = torch.max(logits_mlp, 1)
    _, predict_gnn = torch.max(logits_gnn, 1)
    mlp_accuracy = torch.sum(predict_mlp == labels).item() / len(labels)
    gnn_accuracy = torch.sum(predict_gnn == labels).item() / len(labels)
    # compute mask for high confident samples
    threshold = threshold or config['threshold']
    if config["unable_gnn"]:
        sample_masks_bool = (confidences_mlp > threshold)
    else:
        sample_masks_bool = (confidences_gnn > threshold)
    sample_masks_idx = torch.nonzero(sample_masks_bool, as_tuple=True)[0].numpy()
    # compute accuracy of pseudo labels
    total_pseudo_labels = len(sample_masks_idx)
    if len(sample_masks_idx) > 0:
        correct_pseudo_labels = torch.sum(predict_gnn[sample_masks_bool] == labels[sample_masks_bool]).item()
        pseudo_label_acc = correct_pseudo_labels / total_pseudo_labels
    else:
        correct_pseudo_labels = -1.
        pseudo_label_acc = -1.
    out = {
        'mlp_accuracy': mlp_accuracy,
        'gnn_accuracy': gnn_accuracy,
        'confidences_gnn': confidences_gnn,
        'pred_cls': predict_mlp.numpy() if config['unable_gnn'] else predict_gnn.numpy(),
        'sample_masks': sample_masks_idx,
        'sample_masks_cgct': sample_masks_bool.float(),
        'pseudo_label_acc': pseudo_label_acc,
        'correct_pseudo_labels': correct_pseudo_labels,
        'total_pseudo_labels': total_pseudo_labels,
    }
    return out


def select_closest_domain(config, base_network, classifier_gnn, temp_test_loaders):
    """
    This function selects the closest domain (Stage 2 in Algorithm 2 of Supp Mat) where adaptation need to be performed.
    In the code we compute the mean of the max probability of the target samples from a domain, which can be 
    considered as inversely proportional to the mean of the entropy.

    Higher the max probability == lower is the entropy == higher the inheritability/similarity
    """
    base_network.eval()
    classifier_gnn.eval()
    max_inherit_val = 0.
    for dset_name, test_loader in temp_test_loaders.items():
        test_res = eval_domain(config, test_loader, base_network, classifier_gnn)
        domain_inheritability = test_res['confidences_gnn'].mean().item()
        
        if domain_inheritability > max_inherit_val:
            max_inherit_val = domain_inheritability
            max_inherit_domain_name = dset_name

    print('Most similar target domain: %s' % (max_inherit_domain_name))
    log_str = 'Most similar target domain: %s' % (max_inherit_domain_name)
    config['out_file'].write(log_str + '\n')
    config['out_file'].flush()
    return max_inherit_domain_name


def train_source(config, base_network, classifier_gnn, dset_loaders, logger=None):
    # define loss functions
    criterion_gedge = nn.BCELoss(reduction='mean')
    ce_criterion = nn.CrossEntropyLoss()

    # configure optimizer
    optimizer_config = config['optimizer']
    if config["unable_gnn"]:
        parameter_list = base_network.get_parameters()
    else:
        parameter_list = base_network.get_parameters() +\
                        [{'params': classifier_gnn.parameters(), 'lr_mult': 10, 'decay_mult': 2}]
    optimizer = optimizer_config['type'](parameter_list, **(optimizer_config['optim_params']))

    # configure learning rates
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group['lr'])
    schedule_param = optimizer_config['lr_param']

    # start train loop
    base_network.train()
    classifier_gnn.train()
    len_train_source = len(dset_loaders["source"])
    domain_id = 0
    for i in range(config['source_iters']):
        if optimizer_config['lr_type'] == "inv":
            optimizer = utils.inv_lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()

        # get input data
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        batch_source = iter_source.next()
        inputs_source, labels_source = batch_source['img'].to(DEVICE), batch_source['target'].to(DEVICE)

        # make forward pass for encoder and mlp head
        features_source, logits_mlp = base_network(inputs_source, domain_id)
        mlp_loss = ce_criterion(logits_mlp, labels_source)

        # make forward pass for light encoder
        light_features_source = base_network.light_feature(inputs_source)
        feature_loss = (features_source.detach() - light_features_source).pow(2).mean()

        # make forward pass for gnn head
        logits_gnn, edge_sim = classifier_gnn(features_source)
        gnn_loss = ce_criterion(logits_gnn, labels_source)
        # compute edge loss
        edge_gt, edge_mask = classifier_gnn.label2edge(labels_source.unsqueeze(dim=0))
        edge_loss = criterion_gedge(edge_sim.masked_select(edge_mask), edge_gt.masked_select(edge_mask))

        # total loss and backpropagation
        if config['unable_gnn']:
            loss = mlp_loss
        else:
            loss = feature_loss + mlp_loss + config['lambda_node'] * gnn_loss + config['lambda_edge'] * edge_loss
        loss.backward()
        optimizer.step()

        # printout train loss
        if i % 20 == 0 or i == config['source_iters'] - 1:
            log_str = 'Iters:(%4d/%d)\tMLP loss:%.4f\tGNN loss:%.4f\tEdge loss:%.4f' % (i,
                  config['source_iters'], mlp_loss.item(), gnn_loss.item(), edge_loss.item())
            utils.write_logs(config, log_str)
        # evaluate network every test_interval
        if i % config['test_interval'] == config['test_interval'] - 1 or i == config['source_iters'] - 1:
            mlp_accuracy_dict, gnn_accuracy_dict = evaluate(i, config, base_network, classifier_gnn, dset_loaders['target_test'])
            if logger is not None:
                logger.record("iter", i)
                for key, val in mlp_accuracy_dict.items():
                    logger.record(f"eval/mlp/{key}", val)
                for key, val in gnn_accuracy_dict.items():
                    logger.record(f"eval/gnn/{key}", val)
                logger.record("update/mlp_loss", mlp_loss.item())
                logger.record("update/gnn_loss", gnn_loss.item())
                logger.dump()

    return base_network, classifier_gnn


def train_target(config, base_network, classifier_gnn, dset_loaders, domain_name, logger=None):
    # define loss functions
    ce_criterion = nn.CrossEntropyLoss()

    # configure optimizer
    optimizer_config = config['optimizer']
    parameter_list = base_network.get_fc_parameters()
    if config["finetune_light"]:
        parameter_list += base_network.get_light_parameters()
    optimizer = optimizer_config['type'](parameter_list, **(optimizer_config['optim_params']))

    # configure learning rates
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group['lr'])
    schedule_param = optimizer_config['lr_param']

    # start train loop
    base_network.train()
    len_train_target = len(dset_loaders["target_train"][domain_name])
    domain_id_target = dset_loaders["target_train"][domain_name].dataset.domain_id
    for i in range(config['target_iters']):
        time_start = time.time()
        if optimizer_config['lr_type_hyper'] == "inv":
            optimizer = utils.inv_lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()

        # get input data
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target_train"][domain_name])
        batch_target = iter_target.next()
        inputs_target, labels_target = batch_target['img'].to(DEVICE), batch_target['target'].to(DEVICE)

        # make forward pass for encoder and mlp head
        _, logits_mlp = base_network.light_forward(inputs_target, domain_id_target)
        loss = ce_criterion(logits_mlp, labels_target)
        loss.backward()
        optimizer.step()
        time_end = time.time()
        time_iter = time_end - time_start

        # printout train loss
        if i % 20 == 0 or i == config['target_iters'] - 1:
            log_str = 'Iters:(%4d/%d)\tMLP loss:%.4f\t Time:%.4f' % (i, config['target_iters'], loss.item(), time_iter)
            utils.write_logs(config, log_str)
        # evaluate network every test_interval
        if i % config['test_interval'] == config['test_interval'] - 1 or i == config['target_iters'] - 1:
            # mlp_accuracy_dict, gnn_accuracy_dict = evaluate(i, config, base_network, classifier_gnn, dset_loaders['target_test'])
            test_res = eval_domain(config, dset_loaders['target_test'][domain_name], base_network, classifier_gnn)
            mlp_accuracy, gnn_accuracy = test_res['mlp_accuracy'], test_res['gnn_accuracy']
            log_str = 'Dataset:%s ID:%s\tTest Accuracy mlp %.4f\tTest Accuracy gnn %.4f'\
                  % (domain_name, domain_id_target, mlp_accuracy * 100, gnn_accuracy * 100)
            config['out_file'].write(log_str + '\n')
            config['out_file'].flush()
            print(log_str)
            if logger is not None:
                logger.record("iter", i)
                logger.record("eval/mlp", mlp_accuracy)
                logger.record("eval/gnn", gnn_accuracy)
                logger.record("update/mlp_loss", loss.item())
                logger.dump()

    return base_network, classifier_gnn


def train_target_v2(config, base_network, classifier_gnn, dset_loaders, domain_name, logger=None):
    # configure optimizer for hyper
    optimizer_config = config['optimizer']
    schedule_param = optimizer_config['lr_param']
    parameter_list = base_network.get_fc_parameters()
    optimizer = optimizer_config['type'](parameter_list, **(optimizer_config['optim_params']))

    # start train loop
    base_network.eval()
    len_train_target = len(dset_loaders["target_train"][domain_name])
    domain_id_target = dset_loaders["target_train"][domain_name].dataset.domain_id
    iter_target = iter(dset_loaders["target_train"][domain_name])
    for i in range(config['target_iters']):
        # define loss functions for inner loop
        ce_criterion = nn.CrossEntropyLoss()

        # init new MLP for inner loop and load weights from hyper
        weights = base_network.fc.get_param(domain_id_target)
        target_fc = nn.Linear(base_network.bottleneck.out_features, base_network.fc.out_features).to(DEVICE)
        target_fc.load_state_dict(weights)

        # init inner optimizer
        inner_optim = optimizer_config['type'](
            [{'params': target_fc.parameters(), 'lr_mult': 10, 'decay_mult': 2}], **(optimizer_config['optim_params'])
        )

        # storing theta_i for later calculating delta theta
        inner_state = OrderedDict({k: tensor.data for k, tensor in weights.items()})

        for j in range(config['target_inner_iters']):
            if optimizer_config['lr_type_hyper'] == "inv":
                inner_optim = utils.inv_lr_scheduler(inner_optim, j - 1, **schedule_param)
            inner_optim.zero_grad()

            if (i * config['target_inner_iters'] + j) % len_train_target == 0:
                iter_target = iter(dset_loaders["target_train"][domain_name])
            batch_target = iter_target.next()
            inputs_target, labels_target = batch_target['img'].to(DEVICE), batch_target['target'].to(DEVICE)

            feature = base_network.light_feature(inputs_target)
            logits_mlp = target_fc(feature)

            loss = ce_criterion(logits_mlp, labels_target)
            loss.backward()
            nn.utils.clip_grad_norm_(target_fc.parameters(), 50)
            inner_optim.step()

        if optimizer_config['lr_type_hyper'] == "inv":
            optimizer = utils.inv_lr_scheduler(optimizer, i - 1, **schedule_param)
        optimizer.zero_grad()

        # calculating delta theta
        final_state = target_fc.state_dict()
        delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})

        # calculating phi gradient
        hnet_grads = torch.autograd.grad(
            list(weights.values()), base_network.fc.embed, grad_outputs=list(delta_theta.values())
        )

        # update hnet weights
        base_network.fc.embed.grad = hnet_grads[0]
        nn.utils.clip_grad_norm_(base_network.fc.embed, 50)
        optimizer.step()

        # printout train loss
        if i % 20 == 0 or i == config['target_iters'] - 1:
            log_str = 'Iters:(%4d/%d)\tMLP loss:%.4f' % (i, config['target_iters'], loss.item())
            utils.write_logs(config, log_str)
        # evaluate network every test_interval
        if i % config['test_interval'] == config['test_interval'] - 1 or i == config['target_iters'] - 1:
            mlp_accuracy_dict, gnn_accuracy_dict = evaluate(i, config, base_network, classifier_gnn, dset_loaders['target_test'])
            if logger is not None:
                logger.record("iter", i)
                for key, val in mlp_accuracy_dict.items():
                    logger.record(f"eval/mlp/{key}", val)
                for key, val in gnn_accuracy_dict.items():
                    logger.record(f"eval/gnn/{key}", val)
                logger.record("update/mlp_loss", loss.item())
                logger.dump()

    return base_network, classifier_gnn


def adapt_target(config, base_network, classifier_gnn, dset_loaders, max_inherit_domain, logger=None):
    # define loss functions
    criterion_gedge = nn.BCELoss(reduction='mean')
    ce_criterion = nn.CrossEntropyLoss()
    # add random layer and adversarial network
    class_num = config['encoder']['params']['class_num']
    random_layer = networks.RandomLayer([base_network.output_num(), class_num], config['random_dim'], DEVICE)
    
    adv_net = networks.AdversarialNetwork(config['random_dim'], config['random_dim'], config['ndomains'])
    
    random_layer.to(DEVICE)
    adv_net = adv_net.to(DEVICE)

    # configure optimizer
    optimizer_config = config['optimizer']
    if config["unable_gnn"]:
        parameter_list = base_network.get_parameters() + adv_net.get_parameters()
    else:
        parameter_list = base_network.get_parameters() + adv_net.get_parameters() \
                        + [{'params': classifier_gnn.parameters(), 'lr_mult': 10, 'decay_mult': 2}]
    optimizer = optimizer_config['type'](parameter_list, **(optimizer_config['optim_params']))
    # configure learning rates
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group['lr'])
    schedule_param = optimizer_config['lr_param']

    # start train loop
    len_train_source = len(dset_loaders['source'])
    len_train_target = len(dset_loaders['target_train'][max_inherit_domain])
    domain_id_source = 0
    domain_id_target = dset_loaders['target_train'][max_inherit_domain].dataset.domain_id
    # set nets in train mode
    base_network.train()
    classifier_gnn.train()
    adv_net.train()
    random_layer.train()
    for i in range(config['adapt_iters']):
        if optimizer_config['lr_type'] == "inv":
            optimizer = utils.inv_lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()
        # get input data
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders['source'])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders['target_train'][max_inherit_domain])
        batch_source = iter_source.next()
        batch_target = iter_target.next()
        inputs_source, inputs_target = batch_source['img'].to(DEVICE), batch_target['img'].to(DEVICE)
        labels_source = batch_source['target'].to(DEVICE)
        domain_source, domain_target = batch_source['domain'].to(DEVICE), batch_target['domain'].to(DEVICE)
        domain_input = torch.cat([domain_source, domain_target], dim=0)

        # make forward pass for encoder and mlp head
        features_source, logits_mlp_source = base_network(inputs_source, domain_id_source)
        features_target, logits_mlp_target = base_network(inputs_target, domain_id_target)
        features = torch.cat((features_source, features_target), dim=0)
        logits_mlp = torch.cat((logits_mlp_source, logits_mlp_target), dim=0)
        softmax_mlp = nn.Softmax(dim=1)(logits_mlp)
        mlp_loss = ce_criterion(logits_mlp_source, labels_source)

        # make forward pass for light encoder
        light_features = base_network.light_feature(torch.cat((inputs_source, inputs_target), dim=0))
        feature_loss = (features.detach() - light_features).pow(2).mean()

        # *** GNN at work ***
        # make forward pass for gnn head
        logits_gnn, edge_sim = classifier_gnn(features)
        gnn_loss = ce_criterion(logits_gnn[:labels_source.size(0)], labels_source)
        # compute pseudo-labels for affinity matrix by mlp classifier
        out_target_class = torch.softmax(logits_mlp_target, dim=1)
        target_score, target_pseudo_labels = out_target_class.max(1, keepdim=True)
        idx_pseudo = target_score > config['threshold']
        target_pseudo_labels[~idx_pseudo] = classifier_gnn.mask_val
        # combine source labels and target pseudo labels for edge_net
        node_labels = torch.cat((labels_source, target_pseudo_labels.squeeze(dim=1)), dim=0).unsqueeze(dim=0)
        # compute source-target mask and ground truth for edge_net
        edge_gt, edge_mask = classifier_gnn.label2edge(node_labels)
        # compute edge loss
        edge_loss = criterion_gedge(edge_sim.masked_select(edge_mask), edge_gt.masked_select(edge_mask))

        # *** Adversarial net at work ***
        if config['method'] == 'CDAN+E':
            entropy = transfer_loss.Entropy(softmax_mlp)
            trans_loss = transfer_loss.CDAN(config['ndomains'], [features, softmax_mlp], adv_net,
                                            entropy, networks.calc_coeff(i), random_layer, domain_input)
        elif config['method'] == 'CDAN':
            trans_loss = transfer_loss.CDAN(config['ndomains'], [features, softmax_mlp],
                                            adv_net, None, None, random_layer, domain_input)
        else:
            raise ValueError('Method cannot be recognized.')

        # total loss and backpropagation
        if config["unable_gnn"]:
            loss = config['lambda_adv'] * trans_loss + mlp_loss
        else:
            loss = config['lambda_adv'] * trans_loss + mlp_loss + feature_loss + \
                config['lambda_node'] * gnn_loss + config['lambda_edge'] * edge_loss
        loss.backward()
        optimizer.step()
        # printout train loss
        if i % 20 == 0 or i == config['adapt_iters'] - 1:
            log_str = 'Iters:(%4d/%d)\tMLP loss: %.4f\t GNN Loss: %.4f\t Edge Loss: %.4f\t Transfer loss:%.4f' % (
                i, config["adapt_iters"], mlp_loss.item(), config['lambda_node'] * gnn_loss.item(),
                config['lambda_edge'] * edge_loss.item(), config['lambda_adv'] * trans_loss.item()
            )
            utils.write_logs(config, log_str)
        # evaluate network every test_interval
        if i % config['test_interval'] == config['test_interval'] - 1 or i == config['adapt_iters'] - 1:
            mlp_accuracy_dict, gnn_accuracy_dict = evaluate(i, config, base_network, classifier_gnn, dset_loaders['target_test'])
            if logger is not None:
                logger.record("iter", i)
                for key, val in mlp_accuracy_dict.items():
                    logger.record(f"eval/mlp/{key}", val)
                for key, val in gnn_accuracy_dict.items():
                    logger.record(f"eval/gnn/{key}", val)
                logger.record("update/mlp_loss", mlp_loss.item())
                logger.record("update/gnn_loss", gnn_loss.item())
                logger.record("update/edge_loss", edge_loss.item())
                logger.record("update/trans_loss", trans_loss.item())
                logger.dump()

    return base_network, classifier_gnn


def adapt_target_cgct(config, base_network, classifier_gnn, dset_loaders, random_layer, adv_net):
    # define loss functions
    criterion_gedge = nn.BCELoss(reduction='mean')
    ce_criterion = nn.CrossEntropyLoss()

    # configure optimizer
    optimizer_config = config['optimizer']
    parameter_list = base_network.get_parameters() + adv_net.get_parameters() \
                     + [{'params': classifier_gnn.parameters(), 'lr_mult': 10, 'decay_mult': 2}]
    optimizer = optimizer_config['type'](parameter_list, **(optimizer_config['optim_params']))
    # configure learning rates
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group['lr'])
    schedule_param = optimizer_config['lr_param']

    # start train loop
    len_train_source = len(dset_loaders['source'])
    len_train_target = len(dset_loaders['target_train'])
    domain_id_source = 0
    domain_id_target = dset_loaders['target_train'].dataset.domain_id
    # set nets in train mode
    base_network.train()
    classifier_gnn.train()
    adv_net.train()
    random_layer.train()
    for i in range(config['adapt_iters']):
        if optimizer_config['lr_type'] == "inv":
            optimizer = utils.inv_lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()
        # get input data
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders['source'])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders['target_train'])

        batch_source = iter_source.next()
        batch_target = iter_target.next()
        inputs_source, inputs_target = batch_source['img'].to(DEVICE), batch_target['img'].to(DEVICE)
        labels_source, labels_target = batch_source['target'].to(DEVICE), batch_target['target'].to(DEVICE)
        mask_target = batch_target['mask'].bool().to(DEVICE)
        domain_source, domain_target = batch_source['domain'].to(DEVICE), batch_target['domain'].to(DEVICE)
        domain_input = torch.cat([domain_source, domain_target], dim=0)

        # make forward pass for encoder and mlp head
        features_source, logits_mlp_source = base_network(inputs_source, domain_id_source)
        features_target, logits_mlp_target = base_network(inputs_target, domain_id_target)
        features = torch.cat((features_source, features_target), dim=0)
        logits_mlp = torch.cat((logits_mlp_source, logits_mlp_target), dim=0)
        softmax_mlp = nn.Softmax(dim=1)(logits_mlp)
        # ce loss for MLP head
        mlp_loss = ce_criterion(torch.cat((logits_mlp_source, logits_mlp_target[mask_target]), dim=0),
                                torch.cat((labels_source, labels_target[mask_target]), dim=0))

        # *** GNN at work ***
        # make forward pass for gnn head
        logits_gnn, edge_sim = classifier_gnn(features)
        # compute pseudo-labels for affinity matrix by mlp classifier
        out_target_class = torch.softmax(logits_mlp_target, dim=1)
        target_score, target_pseudo_labels = out_target_class.max(1, keepdim=True)
        idx_pseudo = target_score > config['threshold']
        idx_pseudo = mask_target.unsqueeze(1) | idx_pseudo
        target_pseudo_labels[~idx_pseudo] = classifier_gnn.mask_val
        # combine source labels and target pseudo labels for edge_net
        node_labels = torch.cat((labels_source, target_pseudo_labels.squeeze(dim=1)), dim=0).unsqueeze(dim=0)
        # compute source-target mask and ground truth for edge_net
        edge_gt, edge_mask = classifier_gnn.label2edge(node_labels)
        # compute edge loss
        edge_loss = criterion_gedge(edge_sim.masked_select(edge_mask), edge_gt.masked_select(edge_mask))
        # ce loss for GNN head
        gnn_loss = ce_criterion(classifier_gnn(torch.cat((features_source, features_target[mask_target]), dim=0))[0],
                                torch.cat((labels_source, labels_target[mask_target]), dim=0))

        # *** Adversarial net at work ***
        if config['method'] == 'CDAN+E':
            entropy = transfer_loss.Entropy(softmax_mlp)
            trans_loss = transfer_loss.CDAN(config['ndomains'], [features, softmax_mlp], adv_net,
                                            entropy, networks.calc_coeff(i), random_layer, domain_input)
        elif config['method'] == 'CDAN':
            trans_loss = transfer_loss.CDAN(config['ndomains'], [features, softmax_mlp],
                                            adv_net, None, None, random_layer, domain_input)
        else:
            raise ValueError('Method cannot be recognized.')

        # total loss and backpropagation
        loss = config['lambda_adv'] * trans_loss + mlp_loss + \
               config['lambda_node'] * gnn_loss + config['lambda_edge'] * edge_loss
        loss.backward()
        optimizer.step()
        # printout train loss
        if i % 20 == 0 or i == config['adapt_iters'] - 1:
            log_str = 'Iters:(%4d/%d)\tMLP loss: %.4f\t GNN Loss: %.4f\t Edge Loss: %.4f\t Transfer loss:%.4f' % (
                i, config["adapt_iters"], mlp_loss.item(), config['lambda_node'] * gnn_loss.item(),
                config['lambda_edge'] * edge_loss.item(), config['lambda_adv'] * trans_loss.item()
            )
            utils.write_logs(config, log_str)
        # evaluate network every test_interval
        if i % config['test_interval'] == config['test_interval'] - 1:
            evaluate(i, config, base_network, classifier_gnn, dset_loaders['target_test'])

    return base_network, classifier_gnn


def upgrade_source_domain(config, max_inherit_domain, dsets, dset_loaders, base_network, classifier_gnn):
    target_dataset = ImageList(image_root=config['data_root'], image_list_root=config['data']['image_list_root'],
                               dataset=max_inherit_domain, transform=config['prep']['test'], domain_label=0, domain_id=0,
                               dataset_name=config['dataset'], split='train')
    target_loader = DataLoader(target_dataset, batch_size=config['data']['test']['batch_size'],
                               num_workers=config['num_workers'], drop_last=False)
    # set networks to eval mode
    base_network.eval()
    classifier_gnn.eval()
    test_res = eval_domain(config, target_loader, base_network, classifier_gnn)

    # print out logs for domain
    log_str = 'Adding pseudo labels of dataset: %s\tPseudo-label acc: %.4f (%d/%d)\t Total samples: %d' \
              % (max_inherit_domain, test_res['pseudo_label_acc'] * 100., test_res['correct_pseudo_labels'],
                 test_res['total_pseudo_labels'], len(target_loader.dataset))
    config["out_file"].write(str(log_str) + '\n\n')
    config["out_file"].flush()
    print(log_str + '\n')

    # sub sample the dataset with the chosen confident pseudo labels
    pseudo_source_dataset = ImageList(image_root=config['data_root'],
                                      image_list_root=config['data']['image_list_root'],
                                      dataset=max_inherit_domain, transform=config['prep']['source'],
                                      domain_label=0, domain_id=0, dataset_name=config['dataset'], split='train',
                                      sample_masks=test_res['sample_masks'], pseudo_labels=test_res['pred_cls'])

    # append to the existing source list
    dsets['source'] = ConcatDataset((dsets['source'], pseudo_source_dataset), domain_id=0)
    # create new source dataloader
    dset_loaders['source'] = DataLoader(dsets['source'], batch_size=config['data']['source']['batch_size'] * 2,
                                        shuffle=True, num_workers=config['num_workers'],
                                        drop_last=True, pin_memory=False)


def upgrade_target_domain(config, max_inherit_domain, dsets, dset_loaders, base_network, classifier_gnn):
    target_dataset = ImageList(image_root=config['data_root'], image_list_root=config['data']['image_list_root'],
                               dataset=max_inherit_domain, transform=config['prep']['test'], domain_label=0, domain_id=0,
                               dataset_name=config['dataset'], split='train')
    target_loader = DataLoader(target_dataset, batch_size=config['data']['test']['batch_size'],
                               num_workers=config['num_workers'], drop_last=False)
    # set networks to eval mode
    base_network.eval()
    classifier_gnn.eval()
    test_res = eval_domain(config, target_loader, base_network, classifier_gnn, config['threshold_target'])

    # print out logs for domain
    log_str = 'Adding pseudo labels of dataset: %s\tPseudo-label acc: %.4f (%d/%d)\t Total samples: %d' \
              % (max_inherit_domain, test_res['pseudo_label_acc'] * 100., test_res['correct_pseudo_labels'],
                 test_res['total_pseudo_labels'], len(target_loader.dataset))
    config["out_file"].write(str(log_str) + '\n\n')
    config["out_file"].flush()
    print(log_str + '\n')

    # update pseudo labels
    # domain_id = dsets['target_train'][max_inherit_domain].domain_id
    domain_id = config["domain_id"][max_inherit_domain]
    target_dataset_new = ImageList(image_root=config['data_root'],
                                    image_list_root=config['data']['image_list_root'],
                                    dataset=max_inherit_domain, transform=config['prep']['target'],
                                    domain_label=1, domain_id=domain_id, dataset_name=config['dataset'], split='train',
                                    sample_masks=test_res['sample_masks'], pseudo_labels=test_res['pred_cls'])
    dsets["target_train"][max_inherit_domain] = target_dataset_new
    target_bs = dset_loaders['target_train'][max_inherit_domain].batch_size
    dset_loaders['target_train'][max_inherit_domain] = DataLoader(dataset=target_dataset_new,
                                                            batch_size=target_bs, shuffle=True,
                                                            num_workers=config['num_workers'], drop_last=True)


def upgrade_target_domains(config, dsets, dset_loaders, base_network, classifier_gnn, curri_iter):
    target_dsets_new = {}
    for i, target_domain in enumerate(dsets['target_train']):
        target_dataset = ImageList(image_root=config['data_root'], image_list_root=config['data']['image_list_root'],
                                   dataset=target_domain, transform=config['prep']['test'], domain_label=1, domain_id=i + 1,
                                   dataset_name=config['dataset'], split='train')
        target_loader = DataLoader(target_dataset, batch_size=config['data']['test']['batch_size'],
                                   num_workers=config['num_workers'], drop_last=False)
        # set networks to eval mode
        base_network.eval()
        classifier_gnn.eval()
        test_res = eval_domain(config, target_loader, base_network, classifier_gnn)

        # print out logs for domain
        log_str = 'Updating pseudo labels of dataset: %s\tPseudo-label acc: %.4f (%d/%d)\t Total samples: %d' \
                  % (target_domain, test_res['pseudo_label_acc'] * 100., test_res['correct_pseudo_labels'],
                     test_res['total_pseudo_labels'], len(target_loader.dataset))
        config["out_file"].write(str(log_str) + '\n\n')
        config["out_file"].flush()
        print(log_str + '\n')

        # update pseudo labels
        target_dataset_new = ImageList(image_root=config['data_root'],
                                       image_list_root=config['data']['image_list_root'],
                                       dataset=target_domain, transform=config['prep']['target'],
                                       domain_label=1, domain_id=i + 1, dataset_name=config['dataset'], split='train',
                                       sample_masks=test_res['sample_masks_cgct'],
                                       pseudo_labels=test_res['pred_cls'], use_cgct_mask=True)
        target_dsets_new[target_domain] = target_dataset_new

        if curri_iter == len(config['data']['target']['name']) - 1:
            # sub sample the dataset with the chosen confident pseudo labels
            target_dataset_new = ImageList(image_root=config['data_root'],
                                           image_list_root=config['data']['image_list_root'],
                                           dataset=target_domain, transform=config['prep']['source'],
                                           domain_label=0, domain_id=0, dataset_name=config['dataset'], split='train',
                                           sample_masks=test_res['sample_masks'],
                                           pseudo_labels=test_res['pred_cls'], use_cgct_mask=False)

            # append to the existing source list
            dsets['source'] = ConcatDataset((dsets['source'], target_dataset_new), domain_id=0)
    dsets['target_train'] = target_dsets_new

    if curri_iter == len(config['data']['target']['name']) - 1:
        # create new source dataloader
        dset_loaders['source'] = DataLoader(dsets['source'], batch_size=config['data']['source']['batch_size'] * 2,
                                            shuffle=True, num_workers=config['num_workers'],
                                            drop_last=True, pin_memory=False)
