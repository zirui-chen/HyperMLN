# -- coding: utf-8 --
import sys
import os
import datetime

import torch.cuda

from utils import evaluate, augment_tuple

dataset = './data/JF17K/'
path = './record'

iterations = 2

khge_model = 'HSimplE'
khge_batch = 128
khge_neg_rate = 10
khge_dim = 200
khge_lr = 0.01
khge_iters = 10
khge_hbatch = 1
khge_in_channels = 1
khge_out_channels = 6
khge_filw = 1
khge_filh = 1
khge_hidden_drop = 0.2
khge_input_drop = 0.2
khge_stride = 2
khge_max_arity = 6
khge_cuda = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")



if khge_model == "MTransH":
    if dataset.split("/")[-2] == "JF17K":
        khge_batch, khge_neg_rate, khge_dim, khge_lr, khge_iters, khge_hbatch, khge_hidden_drop = 256, 10, 200, 0.01, 100, 1, 0.2
    if dataset.split("/")[-2] == "M-FB15K":
        khge_batch, khge_neg_rate, khge_dim, khge_lr, khge_iters, khge_hbatch, khge_hidden_drop = 32, 10, 200, 0.01, 10000, 1, 0.2
    if dataset.split("/")[-2] == "FB-AUTO":
        khge_batch, khge_neg_rate, khge_dim, khge_lr, khge_iters, khge_hbatch, khge_hidden_drop = 32, 10, 200, 0.01, 10000, 1, 0.2
    if dataset.split("/")[-2] == "FB15K-237":
        khge_batch, khge_neg_rate, khge_dim, khge_lr, khge_iters, khge_hbatch, khge_hidden_drop = 32, 10, 200, 0.01, 10000, 1, 0.2
    if dataset.split("/")[-2] == "wn18rr":
        khge_batch, khge_neg_rate, khge_dim, khge_lr, khge_iters, khge_hbatch, khge_hidden_drop = 32, 10, 200, 0.01, 10000, 1, 0.2

if khge_model == "MDistMult":
    if dataset.split("/")[-2] == "JF17K":
        khge_batch, khge_neg_rate, khge_dim, khge_lr, khge_iters, khge_hbatch, khge_hidden_drop = 256, 10, 200, 0.01, 100, 1, 0.2
    if dataset.split("/")[-2] == "M-FB15K":
        khge_batch, khge_neg_rate, khge_dim, khge_lr, khge_iters, khge_hbatch, khge_hidden_drop = 32, 10, 200, 0.01, 10000, 1, 0.2
    if dataset.split("/")[-2] == "FB-AUTO":
        khge_batch, khge_neg_rate, khge_dim, khge_lr, khge_iters, khge_hbatch, khge_hidden_drop = 32, 10, 200, 0.01, 10000, 1, 0.2
    if dataset.split("/")[-2] == "FB15K-237":
        khge_batch, khge_neg_rate, khge_dim, khge_lr, khge_iters, khge_hbatch, khge_hidden_drop = 32, 10, 200, 0.01, 10000, 1, 0.2
    if dataset.split("/")[-2] == "wnrr":
        khge_batch, khge_neg_rate, khge_dim, khge_lr, khge_iters, khge_hbatch, khge_hidden_drop = 32, 10, 200, 0.01, 10000, 1, 0.2

if khge_model == "MCP":
    if dataset.split("/")[-2] == "JF17K":
        khge_batch, khge_neg_rate, khge_dim, khge_lr, khge_iters, khge_hbatch, khge_hidden_drop = 256, 10, 200, 0.01, 100, 1, 0.2
    if dataset.split("/")[-2] == "M-FB15K":
        khge_batch, khge_neg_rate, khge_dim, khge_lr, khge_iters, khge_hbatch, khge_hidden_drop = 32, 10, 200, 0.01, 10000, 1, 0.2
    if dataset.split("/")[-2] == "FB-AUTO":
        khge_batch, khge_neg_rate, khge_dim, khge_lr, khge_iters, khge_hbatch, khge_hidden_drop = 32, 10, 200, 0.01, 10000, 1, 0.2
    if dataset.split("/")[-2] == "FB15K-237":
        khge_batch, khge_neg_rate, khge_dim, khge_lr, khge_iters, khge_hbatch, khge_hidden_drop = 32, 10, 200, 0.01, 10000, 1, 0.2
    if dataset.split("/")[-2] == "wnrr":
        khge_batch, khge_neg_rate, khge_dim, khge_lr, khge_iters, khge_hbatch, khge_hidden_drop = 32, 10, 200, 0.01, 10000, 1, 0.2

if khge_model == "HypE":
    if dataset.split("/")[-2] == "JF17K":
        khge_batch, khge_neg_rate, khge_dim, khge_lr, khge_iters, khge_hbatch, khge_hidden_drop, khge_input_drop, khge_in_channels, khge_out_channels, khge_stride, khge_filh, khge_filw = 256, 10, 200, 0.01, 100, 1, 0.2, 0.2, 1, 6, 2, 1, 1
    if dataset.split("/")[-2] == "M-FB15K":
        khge_batch, khge_neg_rate, khge_dim, khge_lr, khge_iters, khge_hbatch, khge_hidden_drop, khge_input_drop, khge_in_channels, khge_out_channels, khge_stride, khge_filh, khge_filw = 32, 10, 200, 0.01, 500, 1, 0.2, 0.2, 1, 6, 2, 1, 1
    if dataset.split("/")[-2] == "FB-AUTO":
        khge_batch, khge_neg_rate, khge_dim, khge_lr, khge_iters, khge_hbatch, khge_hidden_drop, khge_input_drop, khge_in_channels, khge_out_channels, khge_stride, khge_filh, khge_filw = 32, 10, 200, 0.01, 500, 1, 0.2, 0.2, 1, 6, 2, 1, 1
    if dataset.split("/")[-2] == "FB15K-237":
        khge_batch, khge_neg_rate, khge_dim, khge_lr, khge_iters, khge_hbatch, khge_hidden_drop, khge_input_drop, khge_in_channels, khge_out_channels, khge_stride, khge_filh, khge_filw = 32, 10, 200, 0.01, 500, 1, 0.2, 0.2, 1, 6, 2, 1, 1
    if dataset.split("/")[-2] == "wnrr":
        khge_batch, khge_neg_rate, khge_dim, khge_lr, khge_iters, khge_hbatch, khge_hidden_drop, khge_input_drop, khge_in_channels, khge_out_channels, khge_stride, khge_filh, khge_filw = 32, 10, 200, 0.01, 500, 1, 0.2, 0.2, 1, 6, 2, 1, 1

if khge_model == "HSimplE":
    if dataset.split("/")[-2] == "JF17K":
        khge_batch, khge_neg_rate, khge_dim, khge_lr, khge_iters, khge_hbatch, khge_hidden_drop = 128, 10, 200, 0.01, 1000, 100, 0.2
    if dataset.split("/")[-2] == "M-FB15K":
        khge_batch, khge_neg_rate, khge_dim, khge_lr, khge_iters, khge_hbatch, khge_hidden_drop = 32, 10, 200, 0.01, 10, 10000, 0.2
    if dataset.split("/")[-2] == "FB-AUTO":
        khge_batch, khge_neg_rate, khge_dim, khge_lr, khge_iters, khge_hbatch, khge_hidden_drop = 32, 10, 200, 0.01, 10, 10000, 0.2
    if dataset.split("/")[-2] == "FB15K-237":
        khge_batch, khge_neg_rate, khge_dim, khge_lr, khge_iters, khge_hbatch, khge_hidden_drop = 32, 10, 200, 0.01, 10, 10000, 0.2
    if dataset.split("/")[-2] == "wnrr":
        khge_batch, khge_neg_rate, khge_dim, khge_lr, khge_iters, khge_hbatch, khge_hidden_drop = 32, 10, 200, 0.01, 10, 10000, 0.2

if dataset.split('/')[-2] == 'JF17K':
    mln_threshold_of_rule = 0.0
    mln_threshold_of_tuple = 0.1
    weight = 0.5
if dataset.split('/')[-2] == 'FB15k-237':
    mln_threshold_of_rule = 0.6
    mln_threshold_of_tuple = 0.7
    weight = 0.5
if dataset.split('/')[-2] == 'FB-AUTO':
    mln_threshold_of_rule = 0.0
    mln_threshold_of_tuple = 0.5
    weight = 0.5
if dataset.split('/')[-2] == 'M-FB15K':
    mln_threshold_of_rule = 0.0
    mln_threshold_of_tuple = 0.5
    weight = 0.5
if dataset.split('/')[-2] == 'wn18rr':
    mln_threshold_of_rule = 0.1
    mln_threshold_of_tuple = 0.5
    weight = 100

mln_iters = 1000
mln_lr = 0.0001
mln_threads = 8

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def cmd_khge(workspace_path, model):
    if model == 'MTransH':
        return 'python ./khge/run.py --do_train --do_valid --do_test --data_path {} --work_path {} --model {} --num_iterations {} --ary_ml {} --hidden_dim {} --batch_size {} --nr {} --hidden_batch_size {} --learning_rate {} --save_path {} --hidden_drop {} --cuda {}'.format(dataset, workspace_path, model, khge_iters, khge_max_arity, khge_dim, khge_batch, khge_neg_rate, khge_hbatch, khge_lr, path, khge_hidden_drop, khge_cuda)
    if model == 'MDistMult':
        return 'python ./khge/run.py --do_train --do_valid --do_test --data_path {} --work_path {} --model {} --num_iterations {} --ary_ml {} --hidden_dim {} --batch_size {} --nr {} --hidden_batch_size {} --learning_rate {} --save_path {} --hidden_drop {} --cuda {}'.format(dataset, workspace_path, model, khge_iters, khge_max_arity, khge_dim, khge_batch, khge_neg_rate, khge_hbatch, khge_lr, path, khge_hidden_drop, khge_cuda)
    if model == 'MCP':
        return 'python ./khge/run.py --do_train --do_valid --do_test --data_path {} --work_path {} --model {} --num_iterations {} --ary_ml {} --hidden_dim {} --batch_size {} --nr {} --hidden_batch_size {} --learning_rate {} --save_path {} --hidden_drop {} --cuda {}'.format(dataset, workspace_path, model, khge_iters, khge_max_arity, khge_dim, khge_batch, khge_neg_rate, khge_hbatch, khge_lr, path, khge_hidden_drop, khge_cuda)
    if model == 'HSimplE':
        return 'python ./khge/run.py --do_train --do_valid --do_test --data_path {} --work_path {} --model {} --num_iterations {} --ary_ml {} --hidden_dim {} --batch_size {} --nr {} --hidden_batch_size {} --learning_rate {} --save_path {} --hidden_drop {} --cuda {}'.format(dataset, workspace_path, model, khge_iters, khge_max_arity, khge_dim, khge_batch, khge_neg_rate, khge_hbatch, khge_lr, path, khge_hidden_drop, khge_cuda)
    if model == 'HypE':
        return 'python ./khge/run.py --do_train --do_valid --do_test --data_path {} --work_path {} --model {} --num_iterations {} --ary_ml {} --hidden_dim {} --batch_size {} --nr {} --hidden_batch_size {} --learning_rate {} --save_path {} --hidden_drop {} --input_drop {} --in_channels {} --out_channels {} --filt_h {} --filt_w {} --stride {} --cuda {}'.format(dataset, workspace_path, model, khge_iters, khge_max_arity, khge_dim, khge_batch, khge_neg_rate, khge_hbatch, khge_lr, path, khge_hidden_drop, khge_input_drop, khge_in_channels, khge_out_channels, khge_filw, khge_filh, khge_stride, khge_cuda)

def cmd_mln(main_path, workspace_path=None, preprocessing=False):
    if preprocessing == True:
        return './mln/hypergraph_mln -observed {}/train.txt -out-hidden {}/hidden.txt -save {}/mln_saved.txt -thresh-rule {} -iterations 0 -threads {}'.format(main_path, main_path, main_path, mln_threshold_of_rule, mln_threads)
    else:
        return './mln/hypergraph_mln -observed {}/train.txt -probability {}/annotation.txt -out-prediction {}/pred_mln.txt -out-rule {}/rule.txt -thresh-tuple 0 -iterations {} -lr {} -threads {}'.format(main_path, workspace_path, workspace_path, workspace_path, mln_iters, mln_lr, mln_threads)

def save_cmd(save_path):
    with open(save_path, 'w') as fo:
        fo.write('dataset: {}\n'.format(dataset))
        fo.write('iterations: {}\n'.format(iterations))
        fo.write('khge_model: {}\n'.format(khge_model))
        fo.write('khge_batch: {}\n'.format(khge_batch))
        fo.write('khge_dim: {}\n'.format(khge_dim))
        fo.write('khge_lr: {}\n'.format(khge_lr))
        fo.write('khge_iters: {}\n'.format(khge_iters))
        fo.write('khge_hbatch: {}\n'.format(khge_hbatch))
        fo.write('mln_threshold_of_rule: {}\n'.format(mln_threshold_of_rule))
        fo.write('mln_threshold_of_tuple: {}\n'.format(mln_threshold_of_tuple))
        fo.write('mln_iters: {}\n'.format(mln_iters))
        fo.write('mln_lr: {}\n'.format(mln_lr))
        fo.write('mln_threads: {}\n'.format(mln_threads))
        fo.write('weight: {}\n'.format(weight))

time = str(datetime.datetime.now()).replace(' ', '_')
path = path + '/' + time
ensure_dir(path)
save_cmd('{}/cmd.txt'.format(path))

# ------------------------------------------

os.system('cp {}/train.txt {}/train.txt'.format(dataset, path))
os.system('cp {}/train.txt {}/train_augmented.txt'.format(dataset, path))
os.system(cmd_mln(path, preprocessing=True))

for k in range(iterations):

    workspace_path = path + '/' + str(k)
    ensure_dir(workspace_path)

    os.system('cp {}/train_augmented.txt {}/train_khge.txt'.format(path, workspace_path))
    os.system('cp {}/hidden.txt {}/hidden.txt'.format(path, workspace_path))
    print(cmd_khge(workspace_path, khge_model))
    os.system(cmd_khge(workspace_path, khge_model))

    os.system(cmd_mln(path, workspace_path, preprocessing=False))
    augment_tuple('{}/pred_mln.txt'.format(workspace_path), '{}/train.txt'.format(path), '{}/train_augmented.txt'.format(workspace_path), mln_threshold_of_tuple)
    os.system('cp {}/train_augmented.txt {}/train_augmented.txt'.format(workspace_path, path))

    evaluate('{}/pred_mln.txt'.format(workspace_path), '{}/pred_khge_test_raw.txt'.format(workspace_path), '{}/pred_khge_test_fil.txt'.format(workspace_path), '{}/result_khge_mln_raw.txt'.format(workspace_path), '{}/result_khge_mln_fil.txt'.format(workspace_path), weight)
