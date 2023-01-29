# HyperMLN

This is an implementation of HyperMLN from the paper "Explainable Link Prediction in Knowledge Hypergraphs".

## Requirements

Python is running at version <u>3.8.10</u>. Other Python package versions can be found in **requirements.txt**

It is recommended to create a virtual environment with the above version of Python using **conda**, and install the python packages in requirements.txt using **pip** in the virtual environment.

## Run the Codes

The **data** folder contains seven datasets, including JF17K, M-FB15K, FB-AUTO, FB15k, WN18, FB15k-237, and WN18RR. The **khge** folder offers the Python implementation of the knowledge hypergraph embedding methods, including m-TransH, m-DistMult, m-CP, HSimplE, and HypE. The **mln** folder gives the C++ implementation of Markov logic networks. The **rules** folder provides the three types of logic rules (symmetric, inverse, and subrelation) we mined for the seven datasets.

Since the MLNs module is written in C++, we need to compile the MLNs codes before running the program. To compile the codes, we can enter the **mln** folder and execute the following command:

```bash
g++ -O3 hypergraph_mln.cpp -o hypergraph_mln -lpthread
```

Afterward, we can modify the hyperparameters of knowledge hypergraph embedding models and MLNs model and run HyperMLN by the Python script **run.py** in the main folder. To reproduce the paper results, you can use the default parameter settings in this script. The default training setting in the script executes the variational E-Step and M-Step simultaneously. If you want to execute only the variational E-Step during training, you can comment out the code block at the end of the script.

```bash
python run.py
```

During training, the program will create a saving folder in **record** to save the intermediate outputs and the results, and the folder is named as the time when the job is submitted. For each iteration, the program will create a subfolder inside the saving folder. The final result is stored in XXX_test_measure_101itr.json in the time directory.

## Citation

* Zirui Chen, Xin Wang, Chenxu Wang, and Jianxin Li. 2022. Explainable Link Prediction in Knowledge Hypergraphs. In Proceedings of the 31st ACM International Conference on Information &amp; Knowledge Management (CIKM '22). Association for Computing Machinery, New York, NY, USA, 262–271. https://doi.org/10.1145/3511808.3557316.