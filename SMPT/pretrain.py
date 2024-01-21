
import os
from os.path import join, exists, basename
import sys
import argparse
import time
import numpy as np
from glob import glob
import logging

import paddle
import paddle.distributed as dist

from datasets.inmemory_dataset import InMemoryDataset
from utils import load_json_config
from featurizers.gen_featurizer import PredTransformFn, PredCollateFn
from model_zoo.model import SMPTGNNModel, SMPTPredModel
from src.util import exempt_parameters

import pickle
from src.util import *
from multiprocessing import Pool

"""
train()
"""
def train(epoch_id, fb_id, args, model, optimizer, data_gen):
    """tbd"""
    # print("\nrank:%s, epoch:%s/%s, fblock:%s/%s, call train()" % (dist.get_rank(), epoch_id, args.max_epoch, fb_id, args.fblock))
    print("call train()")
    # model.train()
    
    steps = get_steps_per_epoch(args)
    step = 0
    list_loss = []
    for graph_dict, feed_dict in data_gen:
        # if dist.get_rank() == 1:
        #     time.sleep(100000)
        for k in graph_dict:
            graph_dict[k] = graph_dict[k].tensor()
        for k in feed_dict:
            feed_dict[k] = paddle.to_tensor(feed_dict[k])
        train_loss = model(graph_dict, feed_dict)
        train_loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        list_loss.append(train_loss.numpy().mean())
        # print("step %s, train loss: %s, " % (step, train_loss.numpy().mean()))
        if step % 200 == 0:
            print('rank:%s, epoch:%s/%s, fblock:%s/%s, step:%s, train loss: %s' %
                  (dist.get_rank(), epoch_id+1, args.max_epoch, fb_id+1, args.fblock, step, train_loss.numpy().mean()))
        step += 1

        if step > steps:
            print("step(", step, ")>steps(", steps, "), jumpping out")
            break
    # return np.mean(list_loss)
    return list_loss


"""
evaluate()
"""
@paddle.no_grad()
def evaluate(epoch_id, fb_id, args, model, test_dataset, collate_fn):
    """tbd"""
    #print("rank:%s, epoch:%s, fblock:%s, call evaluate()" % (dist.get_rank(), epoch_id, fb_id))
    print("call evaluate()")
    model.eval()
    data_gen = test_dataset.get_data_loader(
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=True, 
            collate_fn=collate_fn)

    dict_loss = {'loss': []}
    for graph_dict, feed_dict in data_gen:
        for k in graph_dict:
            graph_dict[k] = graph_dict[k].tensor()
        for k in feed_dict:
            feed_dict[k] = paddle.to_tensor(feed_dict[k])
        loss, sub_losses = model(graph_dict, feed_dict, return_subloss=True)

        for name in sub_losses:
            if not name in dict_loss:
                dict_loss[name] = []
            v_np = sub_losses[name].numpy()
            dict_loss[name].append(v_np)
        dict_loss['loss'] = loss.numpy()
    dict_loss = {name: np.mean(dict_loss[name]) for name in dict_loss}
    print('rank:%s, epoch:%s/%s, fblock:%s/%s, test loss: %s' %
          (dist.get_rank(), epoch_id+1, args.max_epoch, fb_id+1, args.fblock, dict_loss['loss']))
    return dict_loss


"""
function: init_encoder_model()
return: compound_encoder_config, compound_encoder, model_config, model
"""
def init_encoder_model(args):
   
    print("=========== Initialize SMPTGNNModel and SMPTPredModel ===========")
    compound_encoder_config = load_json_config(args.compound_encoder_config)
    model_config = load_json_config(args.model_config)
    if not args.dropout_rate is None:
        compound_encoder_config['dropout_rate'] = args.dropout_rate
        model_config['dropout_rate'] = args.dropout_rate
    compound_encoder = SMPTGNNModel(compound_encoder_config)
    model = SMPTPredModel(model_config, compound_encoder_config, compound_encoder)
    print("===============================================================\n")

    print("====================== Parallel setting =======================")
    print('args.distributed: %s' % (args.distributed))
    print('args.num_workers: %s' % (args.num_workers))
    print('dist.get_rank: %s' % (dist.get_rank()))
    print('dist.get_world_size: %s' % (dist.get_world_size()))
    print('args.lr: %s' % (args.lr))
    if args.distributed:
        model = paddle.DataParallel(model)
    print("===============================================================\n")

    return compound_encoder_config, compound_encoder, model_config, model


"""
funtion: init_optimizer()
return: opt
"""
def init_optimizer(args, model):
    """
    Define Adam optimizer
    """
    print("==================== Initialize optimizer =====================")
    opt = paddle.optimizer.Adam(learning_rate=args.lr, parameters=model.parameters())
    print('Total param num: %s' % (len(model.parameters())))  # 180
    # for i, param in enumerate(model.named_parameters()):
    #     print(i, param[0], param[1].name)
    print("===============================================================\n")
    return opt


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
function: pickle_dump_data()
"""
def pickle_dump_data(args, model_config):

    if not args.pickel_dump:
        exit(0)

    print("====================== Pickle data ============================")

    """
    Load data (InMemoryDataset)
    Each line is the smile of a molecular
    """
    print("Load smile data")
    dataset = load_smiles_to_dataset(args.data_path)
    dataset_len = len(dataset)
    print('  dataset all len: %s' % (dataset_len))  # 12525012

    print('  args.DEBUG: %s' % (args.DEBUG))
    if args.DEBUG:
        print('  from: %s, to: %s' % (args.dataset_from, args.dataset_to))
        dataset = dataset[args.dataset_from:args.dataset_to]  # [100:180]
        dataset_dir = args.pickle_path + "/" + str(args.dataset_from) + "_" + str(args.dataset_to)
    else:
        dataset_dir = args.pickle_path + "/all"
    mkdir(dataset_dir)
    print("  store in: ", dataset_dir)

    dataset = dataset[dist.get_rank()::dist.get_world_size()]
    smiles_lens = [len(smiles) for smiles in dataset]
    dataset_len = len(dataset)
    print('  dataset current len: %s' % (dataset_len))
    print('  dataset smiles min/max/avg length: %s/%s/%s' % (
    np.min(smiles_lens), np.max(smiles_lens), np.mean(smiles_lens)))  # 30/36/32.25

    print("Pickle dump data")
    dataset_index_list = range(dataset_len)
    dataset_index_block_list = [dataset_index_list[i:i+args.pickle_step] for i in range(0, dataset_len, args.pickle_step)]
    atomic_lens = []
    for pickle_epoch, index_block in enumerate(dataset_index_block_list):
        left_index = index_block[0]
        right_index = index_block[-1]
        dataset_temp = dataset[left_index:right_index+1]
        print("  epoch ", pickle_epoch, ", ", left_index, "-", right_index, ", data len: ", len(dataset_temp))
        if (left_index < args.restart_index): continue
        """
        Convert smile to mol by rdkit for each molecular, then convert mol to graph data
        data contanins 
            smiles
            atomic_num, atom_pos, chiral_tag, degree, explicit_valence, formal_charge, hybridization, is_aromatic, total_numHs,
            mass, bond_dir, bond_type, is_in_ring, edges, morgan_fp, maccs_fp, daylight_fg_counts, atom_pos, bond_length,
            BondAngleGraph_edges, bond_angle, Ba_node_i, Ba_node_j, Ba_node_k, Ba_bond_angle, Ad_node_i, Ad_node_j, Ad_atom_dist
            Ba_bond_angle, Bl_bond_length, Ad_atom_dist, Da_node_i, Da_node_m, Da_node_n, Da_node_j, Da_dihedral_angle
        model_config['mask_ratio'] is not used
        """
        # print("Initialize and call PredTransformFn")
        transform_fn = PredTransformFn(model_config['pretrain_tasks'], model_config['mask_ratio'])
        # this step will be time consuming due to rdkit 3d calculation
        # Inplace apply `transform_fn` on the `data_list` with multiprocess.
        dataset_temp.transform(transform_fn, num_workers=args.num_workers)
        """
        dataset[0]:
        {'atomic_num': array([8, 6, 7, 6, 8, 7, 6, 6, 6, 6, 8, 6, 6, 9]), 
         ...,
         'edges': array([[ 0,  1], [ 1,  0], [ 1,  2], [ 2,  1], [ 2,  3], [ 3,  2], [ 3,  4], [ 4,  3], [ 3,  5],
                [ 5,  3], [ 5,  6], [ 6,  5], [ 6,  7], [ 7,  6], [ 7,  8], [ 8,  7], [ 8,  9], [ 9,  8], [ 9, 10],
                [10,  9], [ 5, 11], [11,  5], [11, 12], [12, 11], [12, 13], [13, 12], [12,  1], [ 1, 12], [10,  6],
                [ 6, 10], [ 0,  0], [ 1,  1], [ 2,  2], [ 3,  3], [ 4,  4], [ 5,  5], [ 6,  6], [ 7,  7], [ 8,  8],
                [ 9,  9], [10, 10], [11, 11], [12, 12], [13, 13]]),
         ...}
        """

        pkldir_num = pickle_epoch // args.pickle_per_dir
        pkl_dir = dataset_dir + "/" + str(pkldir_num)
        mkdir(pkl_dir)
        pd = open(pkl_dir + "/"
                  + str(pickle_epoch) + "_"
                  + str(left_index) + "_" + str(right_index)
                  + ".pkl", "wb")
        pickle.dump(dataset_temp, pd)
        atomic_lens_temp = [len(data['atomic_num']) for data in dataset_temp]
        atomic_lens += atomic_lens_temp
    print('  dataset atomic min/max/avg length: %s/%s/%s' % (
        np.min(atomic_lens), np.max(atomic_lens), np.mean(atomic_lens)))
    print("Finish pickle dump data, exit")
    print("===============================================================\n")


"""
function: pickle_load_data()
return: dataset
"""
def pickle_load_data(args, files_list):
    print("\n------------------ Pickle data ------------------------")
    print("Pickle load dataset")
    print("  files list len:", len(files_list))

    dataset = []
    smile_lens = []
    atomic_lens = []

    # multiprocess load pkls
    print("  begin multiprocessing load pkl ...")
    start_time = time.time()

    files_list = [(i, file) for i, file in enumerate(files_list)]
    pool = Pool(args.num_workers)
    map_results = pool.map_async(load_pkls_to_list, files_list)
    pool.close()
    pool.join()
    for result in map_results.get():
        dataset += result
        smile_lens_temp = [len(r['smiles']) for r in result]
        smile_lens += smile_lens_temp
        atomic_lens_temp = [len(r['atomic_num']) for r in result]
        atomic_lens += atomic_lens_temp
    print("\n  finish multiprocessing load pkl ...")
    # print("  used time: ", time.time()-start_time)
    print("  dataset len:", len(dataset))
    print('  dataset smile min/max/avg length: %s/%s/%s' % (
        np.min(smile_lens), np.max(smile_lens), np.mean(smile_lens)))
    print('  dataset atomic min/max/avg length: %s/%s/%s' % (
        np.min(atomic_lens), np.max(atomic_lens), np.mean(atomic_lens)))
    dataset = InMemoryDataset(data_list=dataset)
    dataset = dataset[dist.get_rank()::dist.get_world_size()]
    print("Finish pickle load data")
    return dataset


"""
function: init_collate_fn()
return: collate_fn
"""
def init_collate_fn(compound_encoder_config, model_config):

    print("================= Initialize PredCollateFn =================")
    collate_fn = PredCollateFn(
        atom_names=compound_encoder_config['atom_names'],
        # atomic_num, formal_charge, degree, chiral_tag, total_numHs, is_aromatic, hybridization
        bond_names=compound_encoder_config['bond_names'],  # bond_dir, bond_type, is_in_ring
        bond_float_names=compound_encoder_config['bond_float_names'],  # bond_length
        bond_angle_float_names=compound_encoder_config['bond_angle_float_names'],  # bond_angle
        plane_names=compound_encoder_config['plane_names'],  # plane_in_ring
        plane_float_names=compound_encoder_config['plane_float_names'],  # plane_mass
        dihedral_angle_float_names=compound_encoder_config['dihedral_angle_float_names'],  # DihedralAngleGraph_angles
        pretrain_tasks=model_config['pretrain_tasks'],  # "Cm", "Fg", "Bar", "Blr", "Adc"
        mask_ratio=model_config['mask_ratio'],  # 0.15
        Cm_vocab=model_config['Cm_vocab'])  # 2400
    print("===============================================================\n")
    return collate_fn


"""
function: generate_tr_te_set()
return: collate_fn, train_data_gen, test_dataset
"""
def generate_tr_te_set(args, dataset, collate_fn):
    """
        Split train and test data based on test_ratio
    """
    print("-------------- Split train and test data --------------")
    test_index = int(len(dataset) * (1 - args.test_ratio))
    train_dataset = dataset[:test_index]
    test_dataset = dataset[test_index:]
    train_data_len = len(train_dataset)
    test_data_len = len(test_dataset)
    print("Train/Test num: %s/%s" % (train_data_len, test_data_len))  # 2/2

    """
    Yield train data by batch size
    """
    # print("====================== Yield train data =======================")
    train_data_gen = train_dataset.get_data_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.train_shuffle,
        collate_fn=collate_fn)
    # print('batch_size:%s' % args.batch_size)
    # print('num_workers:%s' % args.num_workers)
    # print('train_shuffle:%s' % args.train_shuffle)
    # print("===============================================================\n")
    return train_data_gen, test_dataset


"""
main()
"""
def main(args):
    """tbd"""

    # initialize compound encoder and predict model
    compound_encoder_config, compound_encoder, model_config, model = init_encoder_model(args)

    # initialize optimizer
    opt = init_optimizer(args, model)

    # restore existed pretrained model
    restore_model(args, compound_encoder)

    # dump data, if finish dumping then exit
    if args.pickle_dump:
        pickle_dump_data(args, model_config)
        exit(0)


    # get all pickle files list
    all_files_list = get_pickle_files_list(args.pickle_path)
    # all_files_list = all_files_list[args.pickle_load_pkl_beg:args.pickle_load_pkl_end]
    all_files_list = all_files_list[args.pickle_load_pkl_beg:len(all_files_list)]
    avg_files_list_list = avg_split_list(all_files_list, args.fblock)  # 3921/10


    # initialize collate_fn
    collate_fn = init_collate_fn(compound_encoder_config, model_config)


    """
    Train model
    """
    print("\n=========================== Training ==========================\n")
    list_test_loss = []

    ep_st = time.time()
    for epoch_id in range(args.max_epoch):
        st = time.time()

        ep_train_loss_list = []
        ep_test_loss_list = []
        for fb_id, files_list in enumerate(avg_files_list_list):
            dataset = pickle_load_data(args, files_list)
            train_data_gen, test_dataset = generate_tr_te_set(args, dataset, collate_fn)
            list(dataset).clear()

            model.train()
            fl_train_loss_list = train(epoch_id, fb_id, args, model, opt, train_data_gen)
            ep_train_loss_list += fl_train_loss_list
            list(train_data_gen).clear()

            test_loss = evaluate(epoch_id, fb_id, args, model, test_dataset, collate_fn)
            ep_test_loss_list.append(test_loss['loss'])
            list(test_dataset).clear()

            # print("finish epoch:%s/%s, fblock:%s/%s train & test\n\n" % (epoch_id, args.max_epoch, fb_id, args.fblock))
            print("finish train & test")

        if not args.distributed or dist.get_rank() == 0:
            paddle.save(compound_encoder.state_dict(), '%s_[%s]_bs%s_lr%s_dr%s/epoch%d.pdparams'
                        % (args.model_dir, compound_encoder_config['subgraph_archi'],
                           args.batch_size, args.lr, args.dropout_rate, epoch_id))
            list_test_loss.append(np.mean(ep_test_loss_list))
            print("\n++++++++++ epoch:%d train/loss:%s" % (epoch_id, np.mean(ep_train_loss_list)))
            print("++++++++++ epoch:%d test/loss:%s" % (epoch_id, np.mean(ep_test_loss_list)))
            print("++++++++++ epoch time used:%ss\n\n" % (time.time() - st))
    print("\n===============================================================\n")
    
    if not args.distributed or dist.get_rank() == 0:
        print('Best epoch id:%s' % np.argmin(list_test_loss))

    print("++++++++++ Time used:%ss\n\n" % (time.time() - ep_st))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default='zinc')
    parser.add_argument("--data_path", type=str, default='ZINC-data/')

    parser.add_argument("--pickle_path", type=str, default='pickle_dataset')
    # pickle dump
    parser.add_argument("--pickle_dump", action='store_true', default=False)
    parser.add_argument("--DEBUG", action='store_true', default=True)
    parser.add_argument("--dataset_from", type=int, default=0)
    parser.add_argument("--dataset_to", type=int, default=500000)
    parser.add_argument("--restart_index", type=int, default=995200)

    parser.add_argument("--pickle_step", type=int, default=3200)
    parser.add_argument("--pickle_per_dir", type=int, default=500)
    # pickle load, all=3921
    parser.add_argument("--pickle_load", action='store_true', default=True)
    parser.add_argument("--pickle_load_pkl_beg", type=int, default=0)
    parser.add_argument("--pickle_load_pkl_end", type=int, default=3921)
    parser.add_argument("--fblock", type=int, default=50)  # 20

    parser.add_argument("--distributed", action='store_true', default=False)  # Test for False
    parser.add_argument("--num_workers", type=int, default=16)  # 4

    parser.add_argument("--train_shuffle", action='store_true', default=True)  # False
    parser.add_argument("--batch_size", type=int, default=512)  # 512
    parser.add_argument("--max_epoch", type=int, default=100)  # default 100
    parser.add_argument("--test_ratio", type=float, default=0.1)  # default 0.1
    parser.add_argument("--compound_encoder_config", type=str, default='model_configs/gnn_l8.json')
    parser.add_argument("--model_config", type=str, default='model_configs/pretrain.json')
    parser.add_argument("--init_model", type=str)
    parser.add_argument("--model_dir", type=str, default='pretrained_models/zinc')
    parser.add_argument("--lr", type=float, default=0.001)  # 0.001
    parser.add_argument("--dropout_rate", type=float, default=0.2)  # 0.2
    args = parser.parse_args()

    print("\n\n[BEGIN PRETRAINING]\n")
    print("==========================  args  =============================")
    for arg in vars(args):
        print(arg, ':', getattr(args, arg))
    print("===============================================================\n")

    if args.distributed:
        dist.init_parallel_env()

    main(args)
    print("\n[END PRETRAINING]\n\n")