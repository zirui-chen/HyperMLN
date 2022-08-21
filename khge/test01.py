# -- coding: utf-8 --
from dataloader import Dataset
import torch
import torch.nn as nn
import os

data_path = "../data/FB-AUTO/"
work_path = "../a/"
dataset = Dataset(data_path.split("/")[-2])
# r, e1, e2, e3, e4, e5, e6, ms, bs = dataset.next_batch(1, 10, "cpu", mode="hidden")
# print(r)

print(dataset.num_ent())
print(dataset.num_rel())
# for hidden_tuple in dataset.data["hidden"]:
#     print(hidden_tuple)
    # r, e1, e2, e3, e4, e5, e6 = hidden_tuple
    # r = dataset.id2rel[int(r)]
    # e1 = dataset.id2ent[int(e1)]
    # e2 = dataset.id2ent[int(e2)]
    # e3 = dataset.id2ent[int(e3)]
    # e4 = dataset.id2ent[int(e4)]
    # e5 = dataset.id2ent[int(e5)]
    # e6 = dataset.id2ent[int(e6)]