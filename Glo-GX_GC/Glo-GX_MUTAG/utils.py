import torch
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.nn.functional as F


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




# def Initial_graph_generate(args):
#     #初始化一个ER随机图
#     ER = nx.random_graphs.erdos_renyi_graph(args.initNodeNum, 0.7)
#
#     x = torch.Tensor(ER.number_of_nodes(), 10).uniform_(-1,1)
#     adj = nx.to_scipy_sparse_matrix(ER).tocoo()
#
#     # row是adj中非零元素所在的行索引
#     row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
#     # col是adj中非零元素所在的列索引。
#     col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
#
#     # 将行和列进行拼接，shape变为[2, num_edges], 包含两个列表，第一个是row, 第二个是col
#     edge_index = torch.stack([row, col], dim=0)
#
#     edge_attr = torch.ones(len(edge_index[0]),dtype=torch.float)
#
#     data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
#     print(data)
#     return data





def Initial_graph_generate(args):
    #初始化一个无向完全图
    edge_d1 = []
    edge_d2 = []
    for i in range(args.initNodeNum):
        for j in range(args.initNodeNum):
            if i != j:
                edge_d1.append(i)
                edge_d2.append(j)
    x_rand = torch.randint(0, 7, (1, args.initNodeNum))
    for i in range(6):
        x_rand[0][i] = 0
    x = F.one_hot(x_rand, num_classes=7).squeeze(0).float()
    edge_index = torch.tensor([edge_d1, edge_d2]).long()
    edge_type = torch.randint(0, 4, (1, len(edge_index[0])))
    edge_attr = F.one_hot(edge_type, num_classes=4).squeeze(0).float()
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
    elif dataset_name == 'MUTAG':
        return 2

def Draw_graph(Data,index):
    G = to_networkx(Data, to_undirected=True, remove_self_loops=True)
    # print(edge_attr)
    pos = nx.spring_layout(G)
    for n in G.nodes:
        if Data.x[n].argmax().item() == 0:
            color = '#130c0e'   #黑色 C
        elif Data.x[n].argmax().item() == 1:
            color = '#102b6a'   #青蓝 N
        elif Data.x[n].argmax().item() == 2:
            color = '#b2d235'   #黄绿 O
        elif Data.x[n].argmax().item() == 3:
            color = '#f26522'   #朱色 F
        elif Data.x[n].argmax().item() == 4:
            color = '#72baa7'   #青竹色 I
        elif Data.x[n].argmax().item() == 5:
            color = '#f47920'   #橙色 Cl
        else:
            color = '#ffe600'   #黄色 Br
        nx.draw_networkx_nodes(G, pos, nodelist=[n], node_color=color)
    for (u, v, d) in G.edges(data=True):

        # print(u, v, edge_attr[i])
        # G.add_edge(u, v, weight=edge_attr[i])
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)])
    # nx.draw(G)
    image_save_path = 'img/graph'+str(index)+'.png'
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
    if len(gate) - gate.sum().item() > 1:
        gate = Random_select_0(gate)

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










