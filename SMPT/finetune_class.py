#!/usr/bin/python                                                                                                                                  
#-*-coding:utf-8-*-
"""
Finetune:to do some downstream task
"""

import os
from os.path import join, exists, basename
import argparse
import numpy as np

import paddle
import paddle.nn as nn
import pgl

from model_zoo.model import SMPTGNNModel
from utils import load_json_config
from datasets.inmemory_dataset import InMemoryDataset

from src.model import DownstreamModel
from src.featurizer import DownstreamTransformFn, DownstreamCollateFn
from src.util import get_dataset, create_splitter, get_downstream_task_names, \
        calc_rocauc_score, exempt_parameters



"""
function: init_encoder_model()
return: compound_encoder_config, compound_encoder, model_config, model, task_type, task_names
"""
def init_encoder_model(args):
    """
    Build compound_encoder with SMPTGNNModel, based on geognn_l8.json
        __init__()
            construct atom_bond_block_list, bond_angle_block_list with GeoGNNBlock
        forward()
            [INPUT]  atom_bond_graph, bond_angle_graph
            [RETURN] node_repr, edge_repr, graph_repr

    Build model with DownstreamModel, based on down_mlp2.json/down_mlp3.json
        __init__()
            initialize compound encoder
            define norm/mlp/act function
        forward()
            get node_repr, edge_repr, graph_repr with compound_encoder.forward()
            predict result based on graph_repr

    Get task names
    """
    print("=========== Initialize SMPTGNNModel and DownstreamModel ===========")
    compound_encoder_config = load_json_config(args.compound_encoder_config)
    if not args.dropout_rate is None:
        compound_encoder_config['dropout_rate'] = args.dropout_rate
    if not args.subgraph_archi is None:
        compound_encoder_config['subgraph_archi'] = args.subgraph_archi
    print('compound_encoder_config:')
    print(compound_encoder_config)

    task_type = 'class'
    model_config = load_json_config(args.model_config)
    if not args.dropout_rate is None:
        model_config['dropout_rate'] = args.dropout_rate
    task_names = get_downstream_task_names(args.dataset_name, args.data_path + "/" + args.dataset_name)
    model_config['task_type'] = task_type
    model_config['num_tasks'] = len(task_names)
    print('model_config:')
    print(model_config)

    compound_encoder = SMPTGNNModel(compound_encoder_config)
    model = DownstreamModel(model_config, compound_encoder)
    print("===============================================================\n")
    return compound_encoder_config, compound_encoder, model_config, model, task_type, task_names


"""
function: init_loss_optimizer()
return: criterion, encoder_opt, head_opt
"""
def init_loss_optimizer(args, compound_encoder, model):
    """
    Define loss, Adam optimizer
    """
    print("==================== Initialize optimizer =====================")
    criterion = nn.BCELoss(reduction='none')
    encoder_params = compound_encoder.parameters()
    head_params = exempt_parameters(model.parameters(), encoder_params)
    encoder_opt = paddle.optimizer.Adam(args.encoder_lr, parameters=encoder_params)
    head_opt = paddle.optimizer.Adam(args.head_lr, parameters=head_params)
    print('Total param num: %s' % (len(model.parameters())))
    print('Encoder param num: %s' % (len(encoder_params)))
    print('Head param num: %s' % (len(head_params)))
    # for i, param in enumerate(model.named_parameters()):
    #     print(i, param[0], param[1].name)
    print("===============================================================\n")
    return criterion, encoder_opt, head_opt


"""
function: restore_model()
"""
def restore_model(args, compound_encoder):
    """
    Initialize SMPTGNNModel (compound_encoder) with init_model
    """
    if not args.init_model is None and not args.init_model == "":
        compound_encoder.set_state_dict(paddle.load(args.init_model))
        print('Load state_dict from %s' % args.init_model)


"""
function: load_data()
return: dataset
"""
def load_data(args, task_names):
    """
    featurizer:
        Gen features according to the raw data and return the graph data.
        Collate features about the graph data and return the feed dictionary.
    """
    print("========================= Load data ===========================")
    cached_data_path_datesetname = args.cached_data_path + "/" + args.dataset_name
    print("task: %s" % args.task)
    if args.task == 'data':
        print('Preprocessing data...')
        dataset = get_dataset(args.dataset_name, args.data_path + "/" + args.dataset_name, task_names)
        dataset.transform(DownstreamTransformFn(), num_workers=args.num_workers)
        print('Dataset len: %s' % len(dataset))
        print("Saving data...")
        print("Finished")
        dataset.save_data(cached_data_path_datesetname)
        exit(0)
    else:
        if cached_data_path_datesetname is None or cached_data_path_datesetname == "":
            print('Processing data...')
            dataset = get_dataset(args.dataset_name, args.data_path + "/" + args.dataset_name, task_names)
            dataset.transform(DownstreamTransformFn(), num_workers=args.num_workers)
            print('Dataset len: %s' % len(dataset))
        else:
            print('Read preprocessing data...')
            dataset = InMemoryDataset(npz_data_path=cached_data_path_datesetname)
            print('Dataset len: %s' % len(dataset))
    print("===============================================================\n")
    return dataset

"""
function: get_pos_neg_ratio()
return: pos vs neg ratio
"""
def get_pos_neg_ratio(dataset):
    """tbd"""
    labels = np.array([data['label'] for data in dataset])
    return np.mean(labels == 1), np.mean(labels == -1)


"""
function: split_dataset()
return: train_dataset, valid_dataset, test_dataset
"""
def split_dataset(args, dataset):
    """
    Split dataset into train/valid/test dataset based on split_type

    split type of the dataset: random,scaffold,random with scaffold. Here is randomsplit.
    `ScaffoldSplitter` will firstly order the compounds according to Bemis-Murcko scaffold,
    then take the first `frac_train` proportion as the train set, the next `frac_valid` proportion as the valid set
    and the rest as the test set. `ScaffoldSplitter` can better evaluate the generalization ability of the model on
    out-of-distribution samples. Note that other splitters like `RandomSplitter`, `RandomScaffoldSplitter`
    and `IndexSplitter` is also available."
    """
    print("======================= Split dataset =========================")
    splitter = create_splitter(args.split_type)
    train_dataset, valid_dataset, test_dataset = splitter.split(
        dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
    print("Train/Valid/Test num: %s/%s/%s" % (
        len(train_dataset), len(valid_dataset), len(test_dataset)))
    print('Train pos/neg ratio %s/%s' % get_pos_neg_ratio(train_dataset))
    print('Valid pos/neg ratio %s/%s' % get_pos_neg_ratio(valid_dataset))
    print('Test pos/neg ratio %s/%s' % get_pos_neg_ratio(test_dataset))
    print("===============================================================\n")
    return train_dataset, valid_dataset, test_dataset


"""
function: init_collate_fn()
return: collate_fn
"""
def init_collate_fn(compound_encoder_config, task_type):
    """

    """
    print("================ Initialize DownstreamCollateFn ===============")
    collate_fn = DownstreamCollateFn(
            atom_names=compound_encoder_config['atom_names'],
            # atomic_num, formal_charge, degree, chiral_tag, total_numHs, is_aromatic, hybridization
            bond_names=compound_encoder_config['bond_names'],  # bond_dir, bond_type, is_in_ring
            bond_float_names=compound_encoder_config['bond_float_names'],  # bond_length
            bond_angle_float_names=compound_encoder_config['bond_angle_float_names'],  # bond_angle
            plane_names=compound_encoder_config['plane_names'],  # plane_in_ring
            plane_float_names=compound_encoder_config['plane_float_names'],  # plane_mass
            dihedral_angle_float_names=compound_encoder_config['dihedral_angle_float_names'],  # DihedralAngleGraph_angles
            task_type=task_type)
    print("===============================================================\n")
    return collate_fn


"""
function: train()
return: loss
"""
def train(args, model, train_dataset, collate_fn, criterion, encoder_opt, head_opt):
    """
    Define the train function 
    Args:
        args,model,train_dataset,collate_fn,criterion,encoder_opt,head_opt;
    Returns:
        the average of the list loss
    """
    data_gen = train_dataset.get_data_loader(
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=True,
            collate_fn=collate_fn)
    list_loss = []
    model.train()
    for atom_bond_graphs, bond_angle_graphs, dihedral_angle_graphs, valids, labels in data_gen:
        if len(labels) < args.batch_size * 0.5:
            continue
        atom_bond_graphs = atom_bond_graphs.tensor()
        bond_angle_graphs = bond_angle_graphs.tensor()
        dihedral_angle_graphs = dihedral_angle_graphs.tensor()
        labels = paddle.to_tensor(labels, 'float32')
        valids = paddle.to_tensor(valids, 'float32')
        preds = model(atom_bond_graphs, bond_angle_graphs, dihedral_angle_graphs)
        loss = criterion(preds, labels)
        loss = paddle.sum(loss * valids) / paddle.sum(valids)
        loss.backward()
        encoder_opt.step()
        head_opt.step()
        encoder_opt.clear_grad()
        head_opt.clear_grad()
        list_loss.append(loss.numpy())
    return np.mean(list_loss)


"""
function: evaluate()
return: auc
"""
def evaluate(args, model, test_dataset, collate_fn):
    """
    Define the evaluate function
    In the dataset, a proportion of labels are blank. So we use a `valid` tensor 
    to help eliminate these blank labels in both training and evaluation phase.
    """
    data_gen = test_dataset.get_data_loader(
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=False,
            collate_fn=collate_fn)
    total_pred = []
    total_label = []
    total_valid = []
    model.eval()
    for atom_bond_graphs, bond_angle_graphs, dihedral_angle_graphs, valids, labels in data_gen:
        atom_bond_graphs = atom_bond_graphs.tensor()
        bond_angle_graphs = bond_angle_graphs.tensor()
        dihedral_angle_graphs = dihedral_angle_graphs.tensor()
        labels = paddle.to_tensor(labels, 'float32')
        valids = paddle.to_tensor(valids, 'float32')
        preds = model(atom_bond_graphs, bond_angle_graphs, dihedral_angle_graphs)
        total_pred.append(preds.numpy())
        total_valid.append(valids.numpy())
        total_label.append(labels.numpy())
    total_pred = np.concatenate(total_pred, 0)
    total_label = np.concatenate(total_label, 0)
    total_valid = np.concatenate(total_valid, 0)
    return calc_rocauc_score(total_label, total_pred, total_valid)



def main(args):
    """
    Call the configuration function of the model, build the model and load data, then start training.
    model_config:
        a json file  with the hyperparameters, such as dropout rate, learning rate, num tasks and so on;
    num_tasks:
        it means the number of task that each dataset contains, it's related to the dataset;
    """


    # initialize compound encoder and downstream model, get task names
    compound_encoder_config, compound_encoder, model_config, model, task_type, task_names = init_encoder_model(args)


    # initialize loss and optimizer
    criterion, encoder_opt, head_opt = init_loss_optimizer(args, compound_encoder, model)


    # restore existed pretrained model
    restore_model(args, compound_encoder)


    # load raw data and tranform them based on DownstreamTransformFn
    dataset = load_data(args, task_names)


    # split dataset into train/valid/test data
    train_dataset, valid_dataset, test_dataset = split_dataset(args, dataset)
    list(dataset).clear()


    # initialize collate_fn
    collate_fn = init_collate_fn(compound_encoder_config, task_type)


    # start train
    # Load the train function and calculate the train loss in each epoch.
    # Here we set the epoch is in range of max epoch,you can change it if you want.
    # Then we will calculate the train loss ,valid auc, test auc and print them.
    # Finally we save it to the model according to the dataset.
    print("\n======================= Train/Valid/Test ======================\n")
    list_val_auc, list_test_auc = [], []
    for epoch_id in range(args.max_epoch):
        print("\nepoch:%s" % epoch_id)
        train_loss = train(args, model, train_dataset, collate_fn, criterion, encoder_opt, head_opt)
        list(train_dataset).clear()
        val_auc = evaluate(args, model, valid_dataset, collate_fn)
        list(valid_dataset).clear()
        test_auc = evaluate(args, model, test_dataset, collate_fn)
        list(test_dataset).clear()

        list_val_auc.append(val_auc)
        list_test_auc.append(test_auc)
        test_auc_by_eval = list_test_auc[np.argmax(list_val_auc)]
        print("epoch:%s train loss:%s" % (epoch_id, train_loss))
        print("epoch:%s val auc:%s" % (epoch_id, val_auc))
        print("epoch:%s test auc:%s" % (epoch_id, test_auc))
        print("epoch:%s test auc_by_eval:%s" % (epoch_id, test_auc_by_eval))
        paddle.save(compound_encoder.state_dict(), '%s/%s/%s_[%s]_bs%s_elr%s_hlr%s_dr%s/epoch%d/compound_encoder.pdparams'
                    % (args.model_dir, task_type, args.dataset_name, compound_encoder_config['subgraph_archi'],
                           args.batch_size, args.encoder_lr, args.head_lr, args.dropout_rate, epoch_id))
        paddle.save(model.state_dict(), '%s/%s/%s_[%s]_bs%s_elr%s_hlr%s_dr%s/epoch%d/model.pdparams'
                    % (args.model_dir, task_type, args.dataset_name, compound_encoder_config['subgraph_archi'],
                       args.batch_size, args.encoder_lr, args.head_lr, args.dropout_rate, epoch_id))

    print("\n===============================================================\n")

    outs = {
        'model_config': basename(args.model_config).replace('.json', ''),
        'metric': '',
        'dataset': args.dataset_name, 
        'split_type': args.split_type, 
        'batch_size': args.batch_size,
        'dropout_rate': args.dropout_rate,
        'encoder_lr': args.encoder_lr,
        'head_lr': args.head_lr,
        'exp_id': args.exp_id,
    }
    offset = 0  # 20
    best_epoch_id = np.argmax(list_val_auc[offset:]) + offset
    for metric, value in [
            ('test_auc', list_test_auc[best_epoch_id]),
            ('max_valid_auc', np.max(list_val_auc)),
            ('max_test_auc', np.max(list_test_auc))]:
        outs['metric'] = metric
        print('\t'.join(['FINAL'] + ["%s:%s" % (k, outs[k]) for k in outs] + [str(value)]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=['train', 'data'], default='train')

    parser.add_argument("--batch_size", type=int, default=32)  # 32
    parser.add_argument("--num_workers", type=int, default=8)  # 2
    parser.add_argument("--max_epoch", type=int, default=2)  # 100

    parser.add_argument("--data_path", type=str, default='downstream_datasets')
    parser.add_argument("--cached_data_path", type=str, default='cached_data')
    parser.add_argument("--dataset_name", default='bace', choices=['bace', 'bbbp', 'hiv', 'clintox', 'muv', 'sider', 'tox21', 'toxcast', 'pcba'])
    parser.add_argument("--split_type", default='scaffold', choices=['random', 'scaffold',  'index'])

    parser.add_argument("--subgraph_archi", type=str, default='ab')  # ab, ab_ba, ab_ba_da
    parser.add_argument("--compound_encoder_config", type=str, default='model_configs/gnn_l8.json')
    parser.add_argument("--model_config", type=str, default='model_configs/down_mlp3.json')  # down_mlp2.json
    parser.add_argument("--init_model", type=str, default='pretrained_models/zinc_[ab_ba_da]_bs512_lr0.001_dr0.2/epoch99.pdparams')  ##pretrained model path
    parser.add_argument("--model_dir", type=str, default='downstream_models')
    parser.add_argument("--encoder_lr", type=float, default=0.001)  # 0.001
    parser.add_argument("--head_lr", type=float, default=0.001)  # 0.001
    parser.add_argument("--dropout_rate", type=float, default=0.2)  # 0.2
    parser.add_argument("--exp_id", type=int, help='used for identification only')
    args = parser.parse_args()
    print("==========================  args  =============================")
    for arg in vars(args):
        print(arg, ':', getattr(args, arg))
    print("===============================================================\n")
    
    main(args)
