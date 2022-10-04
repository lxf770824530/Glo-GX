import torch
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt
import numpy as np
import random



def Get_tensor_classes_num(y_tensor):
    #获取类别数目

    return len(set(y_tensor.numpy().tolist()))



def load_checkpoint(model, checkpoint_PATH):
    #加载模型

    model_CKPT = torch.load(checkpoint_PATH)
    model.load_state_dict(model_CKPT['state_dict'])
    # print('loading checkpoint......')
    # optimizer.load_state_dict(model_CKPT['optimizer'])
    return model



def Initial_graph_generate(args):
    #初始化一个无向完全图
    edge_d1 = []
    edge_d2 = []
    for i in range(args.initNodeNum):
        for j in range(args.initNodeNum):
            if i!=j:
                edge_d1.append(i)
                edge_d2.append(j)

    x = torch.Tensor(args.initNodeNum, 10).uniform_(-1,1)
    edge_index = torch.tensor([edge_d1, edge_d2]).long()

    index_mask1 = torch.randint(low=0, high=2, size=(1, args.initNodeNum * (args.initNodeNum - 1))).bool()
    index_mask2 = torch.randint(low=0, high=2, size=(1, args.initNodeNum * (args.initNodeNum - 1))).bool()
    index_mask = index_mask1 | index_mask2
    index_mask = torch.stack((index_mask[0],index_mask[0]))

    masked_edge_index = edge_index[index_mask].view(2,-1)
    edge_attr = torch.ones(len(edge_d1),dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    print(data)
    return data

def Fix_nodes_index(edge_index):
    b = edge_index.view(-1)
    c = []
    d = {}
    e = []
    for i in b:
        if i not in c:
            c.append(i.item())
    c.sort()

    for v,k in enumerate(c):
        d[k] = v

    for i in edge_index:
        for j in i :
            e.append(d[j.item()])
    t = torch.tensor(e).view(2,-1)
    return t


def Get_dataset_class_num(dataset_name):
    if dataset_name == 'BA_shapes':
        return 2
    elif dataset_name == 'Tree_Cycle':
        return 2


def Draw_graph(Data,j):
    edge_attr = []
    edge_max = 0.0
    for i in Data.edge_attr:
        edge_attr.append(i.item())
        if i.item() > edge_max:
            edge_max = i.item()
        G = to_networkx(Data, to_undirected=True, remove_self_loops=True)
    # print(edge_attr)
    pos = nx.spring_layout(G)
    for n in G.nodes:
        if n == 0:
            color = 'red'
        else:
            color = 'blue'
        nx.draw_networkx_nodes(G, pos, nodelist=[n], node_color=color)
    i=0
    for (u, v, d) in G.edges(data=True):

        # print(u, v, edge_attr[i])
        # G.add_edge(u, v, weight=edge_attr[i])
        nx.draw_networkx_edges(G, pos, edgelist=[(u,v)])
        i += 1
    # nx.draw(G)
    image_save_path = 'img/graph'+str(j+1)+'.png'
    plt.savefig(image_save_path)
    plt.close('all')
    # plt.show()

def make_one_hot(data1,args):
    l = Get_dataset_class_num(args.dataset)

    return (np.arange(l)==data1[:,None]).astype(np.integer)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def Random_select_0(gate):
    index_0 = []
    for i in range(len(gate)):
        if gate[i]<1:
            index_0.append(i)
    ran_max = random.randint(0,len(index_0)-1)
    ga=(gate >= 0).float()
    ga[index_0[ran_max]]=0
    return ga




def Fix_gate(gate, graph_index):
    if len(gate)-gate.sum().item()>1:
        gate = Random_select_0(gate)
        # for j in range(len(gate)-gate.sum().long().item()-1):
        #     gate[gate.argmin()]=1.0

    if len(graph_index) == 2:
        graph_index = torch.t(graph_index)
    min_index = gate.argmin()
    u = graph_index[min_index.item()][0].item()
    v = graph_index[min_index.item()][1].item()
    # inverse_uv = torch.Tensor([v,u]).to(device)
    inverse_uv = torch.Tensor([v, u])

    for i, edge in enumerate(graph_index.float()):
        if torch.equal(edge, inverse_uv):
            gate[i]=0
    return gate










