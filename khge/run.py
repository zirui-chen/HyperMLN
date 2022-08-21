# -- coding: utf-8 --
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import matplotlib.pyplot as plt
import argparse
import json
import logging
logging.getLogger().setLevel(logging.INFO)
from dataloader import Dataset
from tester import Tester

import os
import random

import numpy as np
import torch
from model import  *


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Hypergraph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', type=str, default="cpu", help='use GPU')

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--work_path', type=str, default=None)
    parser.add_argument('--model', default='HypE', type=str)

    parser.add_argument('--num_iterations', default=5, type=int)
    parser.add_argument('--ary_ml', default=6, type=int)

    parser.add_argument('-d', '--hidden_dim', default=200, type=int)
    parser.add_argument('-b', '--batch_size', default=32, type=int)
    parser.add_argument('--nr', default=10, type=int)
    parser.add_argument('-hb', '--hidden_batch_size', default=1, type=int)

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    parser.add_argument('--in_channels', default=1, type=int)
    parser.add_argument('--out_channels', default=6, type=int)
    parser.add_argument('--filt_h', default=1, type=int)
    parser.add_argument('--filt_w', default=1, type=int)
    parser.add_argument('--hidden_drop', default=0.2, type=float)
    parser.add_argument('--input_drop', default=0.2, type=float)
    parser.add_argument('--stride', default=2, type=int)

    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('--record', default=True, action='store_true')

    return parser.parse_args(args)



def save_model(model, optimizer, measure, args, itr=0, test_or_valid='test', is_best_model=False):
    """
    Save the model state to the output folder.
    If is_best_model is True, then save the model also as best_model.chkpnt
    """
    if is_best_model:
        torch.save(model.state_dict(), os.path.join(args.save_path, 'best_model.chkpnt'))
        print("######## Saving the BEST MODEL")

    model_name = 'model_{}itr.chkpnt'.format(itr)
    opt_name = 'opt_{}itr.chkpnt'.format(itr) if itr else '{}.chkpnt'.format(args.model)
    measure_name = '{}_measure_{}itr.json'.format(test_or_valid, itr) if itr else '{}.json'.format(args.model)
    print("######## Saving the model {}".format(os.path.join(args.save_path, model_name)))

    torch.save(model.state_dict(), os.path.join(args.save_path, model_name))
    torch.save(optimizer.state_dict(), os.path.join(args.save_path, opt_name))
    if measure is not None:
        measure_dict = vars(measure)
        # If a best model exists
        if is_best_model:
            measure_dict["best_iteration"] = model.best_itr.cpu().item()
            measure_dict["best_mrr"] = model.best_mrr.cpu().item()
        with open(os.path.join(args.save_path, measure_name), 'w') as f:
            json.dump(measure_dict, f, indent=4, sort_keys=True)
    # Note that measure_by_arity is only computed at test time (not validation)
    # if (self.test_by_arity) and (self.measure_by_arity is not None):
    #     H = {}
    #     measure_by_arity_name = '{}_measure_{}itr_by_arity.json'.format(test_or_valid,
    #                                                                     itr) if itr else '{}.json'.format(
    #         self.model_name)
    #     for key in self.measure_by_arity:
    #         H[key] = vars(self.measure_by_arity[key])
    #     with open(os.path.join(self.output_dir, measure_by_arity_name), 'w') as f:
    #         json.dump(H, f, indent=4, sort_keys=True)

def read_tuple(file_path, ent2id, rel2id, max_arity=6):
    if not os.path.exists(file_path):
        print("*** {} not found. Skipping. ***".format(file_path))
        return ()
    with open(file_path, "r") as f:
        lines = f.readlines()
    tuples = np.zeros((len(lines), max_arity + 1))
    for i, line in enumerate(lines):
        tuples[i] = tuple2id(line.strip().split("\t"), ent2id, rel2id)
    return tuples

def tuple2id(tuple_, ent2id, rel2id, max_arity=6):
    output = np.zeros(max_arity + 1)
    for ind, t in enumerate(tuple_):
        if ind == 0:
            output[ind] = get_rel_id(t, rel2id)
        else:
            output[ind] = get_ent_id(t, ent2id)
    return output

def get_ent_id(ent, ent2id):
    if not ent in ent2id:
        ent2id[ent] = len(ent2id)
    return ent2id[ent]

def get_rel_id(rel, rel2id):
    if not rel in rel2id:
        rel2id[rel] = len(rel2id)
    return rel2id[rel]

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def decompose_predictions(targets, predictions, max_length):
    positive_indices = np.where(targets > 0)[0]
    seq = []
    for ind, val in enumerate(positive_indices):
        if(ind == len(positive_indices)-1):
            seq.append(padd(predictions[val:], max_length))
        else:
            seq.append(padd(predictions[val:positive_indices[ind + 1]], max_length))
    return seq

def padd(a, max_length):
    b = F.pad(a, (0,max_length - len(a)), 'constant', -math.inf)
    return b

def padd_and_decompose(targets, predictions, max_length):
    seq = decompose_predictions(targets, predictions, max_length)
    return torch.stack(seq)

def main(args):
    # args.do_train = True
    # args.do_valid = True
    # args.do_test = True
    # args.record = True
    # args.learning_rate = 0.01
    # args.nr = 10
    # args.cuda = "cuda:0"
    # args.filt_h = 1
    # args.filt_w = 1
    # args.hidden_drop = 0.2
    # args.input_drop = 0.2
    # args.stride = 2
    # args.hidden_batch_size = 1
    # args.num_iterations = 10
    # args.data_path = "../data/FB-AUTO/"
    # args.work_path = "../record/2022-01-27_11:12:26.873930/0"
    # args.save_path = "../record/2022-01-27_11:12:26.873930"

    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')

    if args.init_checkpoint:
        pass
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        id2entity = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)
            id2entity[int(eid)] = entity

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        id2relation = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
            id2relation[int(rid)] = relation

    nentity = len(entity2id)
    nrelation = len(relation2id)

    args.nentity = nentity
    args.nrelation = nrelation

    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)

    # train_tuples = read_tuple(os.path.join(args.workspace_path, "train_kge.txt"), entity2id, relation2id)
    train_tuples = read_tuple(os.path.join(args.data_path, "train.txt"), entity2id, relation2id)
    logging.info('#train: %d' % len(train_tuples))
    train_original_tuples = read_tuple(os.path.join(args.data_path, "train.txt"), entity2id, relation2id)
    logging.info('#train original: %d' % len(train_original_tuples))
    valid_tuples = read_tuple(os.path.join(args.data_path, "valid.txt"), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_tuples))
    test_tuples = read_tuple(os.path.join(args.data_path, "test.txt"), entity2id, relation2id)
    logging.info('#test: %d' % len(test_tuples))
    hidden_tuples = read_tuple(os.path.join(args.work_path, "hidden.txt"), entity2id, relation2id)
    # # hidden_tuples = read_tuple(os.path.join(args.data_path, "hidden.txt"), entity2id, relation2id)
    logging.info('#hidden: %d' % len(hidden_tuples))

    train_facts = [list(int(j) for j in i) for i in train_original_tuples]
    valid_facts = [list(int(j) for j in i) for i in valid_tuples]
    test_facts = [list(int(j) for j in i) for i in test_tuples]
    all_true_tuples = train_facts + valid_facts + test_facts
    device = torch.device("{}".format(args.cuda))
    dataset = Dataset(args.data_path, args.work_path)
    kwargs = {"in_channels": args.in_channels, "out_channels": args.out_channels, "filt_h": args.filt_h,
              "filt_w": args.filt_w,
              "hidden_drop": args.hidden_drop, "stride": args.stride, "input_drop": args.input_drop}

    if(args.model == "MDistMult"):
        model = MDistMult(dataset.num_ent(), dataset.num_rel(), args.hidden_dim, **kwargs).to(device)
    elif(args.model == "MCP"):
        model = MCP(dataset.num_ent(), dataset.num_rel(), args.hidden_dim, **kwargs).to(device)
    elif(args.model == "HSimplE"):
        model = HSimplE(dataset.num_ent(), dataset.num_rel(), args.hidden_dim, **kwargs).to(device)
    elif(args.model == "HypE"):
        model = HypE(dataset.num_ent(), dataset.num_rel(), args.hidden_dim, **kwargs).to(device)
    elif(args.model == "MTransH"):
        model = MTransH(dataset.num_ent(), dataset.num_rel(), args.hidden_dim, **kwargs).to(device)
    else:
        raise Exception("!!!! No mode called {} found !!!!".format(args.model))

    logging.info('Model Parameter Configuration:')
    for name, param in model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.do_train:
        if (model.cur_itr.data >= args.num_iterations):
            logging.info("*************")
            logging.info("Number of iterations is the same as that in the pretrained model.")
            logging.info("Nothing left to train. Exiting.")
            logging.info("*************")
            return


        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.learning_rate)

        if args.init_checkpoint:
            # Restore model from checkpoint directory
            logging.info('Loading checkpoint %s...' % args.init_checkpoint)
            checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
            init_step = checkpoint['step']
            model.load_state_dict(checkpoint['model_state_dict'])
            if args.do_train:
                current_learning_rate = checkpoint['current_learning_rate']
                warm_up_steps = checkpoint['warm_up_steps']
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            logging.info('Ramdomly Initializing %s Model...' % args.model)
            init_step = 0

    step = init_step
    logging.info("Training the {} model...".format(args.model))
    logging.info("Number of training data points: {}".format(len(train_tuples)))
    logging.info("Starting training at iteration ... {}".format(model.cur_itr.data))
    logging.info('learning_rate = %.4f' % args.learning_rate)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('hidden_dim = %d' % args.hidden_dim)

    if args.record:
        local_path = args.work_path
        ensure_dir(local_path)

        opt = vars(args)
        with open(local_path + '/opt.txt', 'w') as fo:
            for key, val in opt.items():
                fo.write('{} {}\n'.format(key, val))


    if args.do_train:
        best_model = None
        training_logs = []
        each_loss = []
        train_iter = []
        loss_layer = torch.nn.CrossEntropyLoss()
        model.init()
        for it in range(model.cur_itr.data, args.num_iterations + 1):
            losses = 0
            last_batch = False
            model.train()
            model.cur_itr.data += 1
            while not last_batch:
                r, e1, e2, e3, e4, e5, e6, targets, ms, bs = dataset.next_batch(args.batch_size, neg_ratio=args.nr, device=device, mode="train")

                last_batch = dataset.was_last_batch()
                optimizer.zero_grad()
                number_of_positive = len(np.where(targets > 0)[0])
                if (args.model == "HypE"):
                    predictions = model.forward(r, e1, e2, e3, e4, e5, e6, ms, bs)
                elif (args.model == "MTransH"):
                    predictions = model.forward(r, e1, e2, e3, e4, e5, e6, ms)
                else:
                    predictions = model.forward(r, e1, e2, e3, e4, e5, e6)
                predictions = padd_and_decompose(targets, predictions, args.nr * args.ary_ml)
                targets = torch.zeros(number_of_positive).long().to(device)
                loss = loss_layer(predictions, targets)
                loss.backward()
                optimizer.step()
                losses += loss.item()
                each_loss.append(loss.item())
                log = {
                    'loss': loss.item()
                }
                training_logs.append(log)
            logging.info("Iteration#: {}, loss: {}".format(it, losses))
            train_iter.append(it)
            if (it % 100 == 0 and it != 0) or (it == args.num_iterations) and args.do_valid:
                logging.info('Evaluating on Valid Dataset...')
                model.eval()
                with torch.no_grad():
                    tester_valid = Tester(dataset, model, "valid", args.model, args.work_path)
                    measure_valid, _ = tester_valid.test()
                    mrr = measure_valid.mrr["fil"]
                    is_best_model = (best_model is None) or (mrr > best_model.best_mrr)
                    if is_best_model:
                        best_model = model
                        # Update the best_mrr value
                        best_model.best_mrr.data = torch.from_numpy(np.array([mrr]))
                        best_model.best_itr.data = torch.from_numpy(np.array([it]))
                    save_model(model, optimizer, measure_valid, args, it, "valid", is_best_model=is_best_model)

        # plt.plot(train_iter, each_loss)
        # plt.show()
        # plt.savefig("Loss.png")


    if args.do_test:
        if (model.cur_itr.data >= args.num_iterations):
            best_model.eval()
            tester_test = Tester(dataset, best_model, "test", args.model, args.work_path)
            measure_test, _ = tester_test.test()
            save_model(best_model, optimizer, measure_test, args, model.cur_itr, "test")

    if args.record:
        # Annotate hidden tuples
        scores = []
        best_model.eval()
        last_batch = False
        while not last_batch:
            r, e1, e2, e3, e4, e5, e6, ms, bs = dataset.next_batch(args.hidden_batch_size, neg_ratio=args.nr,
                                                                            device=device, mode="hidden")
            last_batch = dataset.was_last_batch()
            if (args.model == "HypE"):
                score = best_model.forward(r, e1, e2, e3, e4, e5, e6, ms, bs)
            elif (args.model == "MTransH"):
                score = best_model.forward(r, e1, e2, e3, e4, e5, e6, ms)
            else:
                score = best_model.forward(r, e1, e2, e3, e4, e5, e6)

            scores.append(torch.sigmoid(score/100).data.cpu().numpy().tolist())

        with open(local_path + "/annotation.txt", "w") as f:
            index = 0
            for hidden_tuple in dataset.data["hidden"]:
                r, e1, e2, e3, e4, e5, e6 = hidden_tuple
                r = dataset.id2rel[int(r)]
                e1 = dataset.id2ent[int(e1)]
                e2 = dataset.id2ent[int(e2)]
                e3 = dataset.id2ent[int(e3)]
                e4 = dataset.id2ent[int(e4)]
                e5 = dataset.id2ent[int(e5)]
                e6 = dataset.id2ent[int(e6)]

                if r != "":
                    f.write(r+"\t")
                if e1 != "":
                    f.write(e1+"\t")
                if e2 != "":
                    f.write(e2+"\t")
                if e3 != "":
                    f.write(e3+"\t")
                if e4 != "":
                    f.write(e4+"\t")
                if e5 != "":
                    f.write(e5+"\t")
                if e6 != "":
                    f.write(e6+"\t")

                f.write("{}\n".format(scores[index][0]))
                index+=1

            f.close()




    # if args.do_valid:
    #     logging.info('Evaluating on Valid Dataset...')
    #     model.eval()
    #     with torch.no_grad():
    #         tester = Tester(dataset, model, "valid", args.model)
    #         measure_valid, _ = tester.test()
    #         mrr = measure_valid.mrr["fil"]
    #         save_model(model, optimizer, measure_valid, args, it, "valid")


if __name__ == '__main__':
    main(parse_args())
