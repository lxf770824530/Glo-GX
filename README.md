#Glo-GX

We present a novel global GNN model explainer (Glo-GX) from the input-independent and model-level perspective. Glo-GX understands and explains the GNNs by discovering a general graph pattern for a trained GNN model from a complete graph rather than specific input graph examples. We introduce an edge mask learning policy to discover a general graph pattern by discarding the edges from a complete graph. Then, we incorporate a Simulated Annealing algorithm-based method to uncover a general graph pattern to alleviate prediction errors efficiently and avoid suboptimal results. Moreover, our Glo-GX is well suited for investigating explanations for some common GNNs-based tasks, including node classification and graph classification.

##Requirement

Python 3.8

Pytorch 1.9.0

Pytorch-Geometric 2.0.0

network 2.5.1

##Data

Our method is evaluated on three datasets, As shown in following table. These datasets can be found in an open-source library [DIG](https://github.com/divelab/DIG/tree/main/dig/xgraph/datasets), which can be directly used to reproduce results of existing GNN explanation methods, develop new algorithms, and conduct evaluations for explanation results.


| Dataset    | Task                  | Data class     |
|------------|-----------------------|----------------|
| Tree_cycle | Node classification   | Synthetic data |
| Is_Acyclic | Graph classification  | Synthetic data |
| MUTAG      | Graph classification  | Real-word data |


##How to use

Our method can be used to explain both node classification model and graph classification. For each task, you just need to enter the corresponding folder to find the main.py.

For example, when we run the method for explaining the graph classification model which is trained on the Is_Acyclic dataset, we run the following command in the console.

First, we need train a simple GNN model which will be explained:
```
python Glo-GX_GC/Glo-GX_Is_Acyclic/main.py --mode train
```
If needed, you can change other configuration parameters in the source file 'main.py'.

Then, running the explanation process by this command:

```
python Glo-GX_GC/Glo-GX_Is_Acyclic/main.py --mode explain --initNodeNum 10 --explain_class 1 --final_node_number 6
```

The explanation results will be saved in the folder '/img'.  


