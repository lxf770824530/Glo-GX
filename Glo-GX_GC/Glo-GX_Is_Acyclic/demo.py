import argparse
from GNNs import Train_gcn_model
import time
from GNN_Explainer import Explain_model
from dateset import IsAcyclicDataset



if __name__ == '__main__':

    data = IsAcyclicDataset('data', name='Is_Acyclic')
    n=100
    for i in range(len(data)):
        if len(data[i].x)<=n:
            n = len(data[i].x)
    print(n)


