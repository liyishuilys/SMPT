
"""
This is an implementation of GeoGNN:
"""
import numpy as np

import paddle
import paddle.nn as nn
import pgl
from pgl.nn import GraphPool

from networks.gnn_block import GIN
from networks.compound_encoder import AtomEmbedding, BondEmbedding, \
        BondFloatRBF, BondAngleFloatRBF, PlaneEmbedding, PlaneFloatRBF, DihedralAngleFloatRBF
from utils.compound_tools import CompoundKit
from pahelix.networks.gnn_block import MeanPool, GraphNorm
from pahelix.networks.basic_block import MLP


class GNNBlock(nn.Layer):
    """
    GeoGNN Block
    """
    def __init__(self, embed_dim, dropout_rate, last_act):
        super(GNNBlock, self).__init__()

        self.embed_dim = embed_dim
        self.last_act = last_act

        self.gnn = GIN(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.graph_norm = GraphNorm()
        if last_act:
            self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, graph, node_hidden, edge_hidden):
        """tbd"""
        out = self.gnn(graph, node_hidden, edge_hidden)
        out = self.norm(out)
        out = self.graph_norm(graph, out)
        if self.last_act:
            out = self.act(out)
        out = self.dropout(out)
        out = out + node_hidden
        return out


class SMPTGNNModel(nn.Layer):
    """
    The GeoGNN Model used in GEM.

    Args:
        model_config(dict): a dict of model configurations.
    """
    def __init__(self, model_config={}):
        super(SMPTGNNModel, self).__init__()

        self.subgraph_archi = model_config['subgraph_archi']
        self.embed_dim = model_config.get('embed_dim', 32)
        self.dropout_rate = model_config.get('dropout_rate', 0.2)
        self.layer_num = model_config.get('layer_num', 8)
        self.readout = model_config.get('readout', 'mean')


        self.atom_names = model_config['atom_names']
        self.bond_names = model_config['bond_names']
        self.bond_float_names = model_config['bond_float_names']
        # ab_ba: atom_bond_graph => bond_angle_graph
        if self.subgraph_archi == "ab_ba" or self.subgraph_archi == "ab_ba_da":
            self.bond_angle_float_names = model_config['bond_angle_float_names']
        # ab_ba_da: atom_bond_graph => bond_angle_graph => dihedral_angle_graph
        if self.subgraph_archi == "ab_ba_da":
            self.plane_names = model_config['plane_names']
            self.plane_float_names = model_config['plane_float_names']
            self.dihedral_angle_float_names = model_config['dihedral_angle_float_names']


        self.init_atom_embedding = AtomEmbedding(self.atom_names, self.embed_dim)
        self.init_bond_embedding = BondEmbedding(self.bond_names, self.embed_dim)
        self.init_bond_float_rbf = BondFloatRBF(self.bond_float_names, self.embed_dim)
        # ab_ba_da: atom_bond_graph => bond_angle_graph => dihedral_angle_graph
        if self.subgraph_archi == "ab_ba_da":
            self.init_plane_embedding = PlaneEmbedding(self.plane_names, self.embed_dim)
            self.init_plane_float_rbf = PlaneFloatRBF(self.plane_float_names, self.embed_dim)
        """
        see PaddleHelix-dev/pahelix/utils/compound_tools.py
        eg. "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'], 12+5=17
        
        AtomEmbedding(
         (embed_list): LayerList(
           (0): Embedding(124, 32, sparse=False)  # atomic_num
           (1): Embedding(22, 32, sparse=False)   # formal_charge
           (2): Embedding(17, 32, sparse=False)   # degree
           (3): Embedding(9, 32, sparse=False)    # chiral_tag
           (4): Embedding(15, 32, sparse=False)   # total_numHs
           (5): Embedding(7, 32, sparse=False)    # is_aromatic
           (6): Embedding(13, 32, sparse=False)   # hybridization
         )
        )
        
        BondEmbedding(
         (embed_list): LayerList(
           (0): Embedding(12, 32, sparse=False)
           (1): Embedding(27, 32, sparse=False)
           (2): Embedding(7, 32, sparse=False)
         )
        )
        
        BondFloatRBF(
         (linear_list): LayerList(
           (0): Linear(in_features=20, out_features=32, dtype=float32)
         )
         (rbf_list): LayerList(
           (0): RBF()
         )
        )
        """

        self.bond_embedding_list = nn.LayerList()
        self.bond_float_rbf_list = nn.LayerList()
        self.atom_bond_block_list = nn.LayerList()
        # ab_ba: atom_bond_graph => bond_angle_graph
        if self.subgraph_archi == "ab_ba" or self.subgraph_archi == "ab_ba_da":
            self.bond_angle_float_rbf_list = nn.LayerList()
            self.bond_angle_block_list = nn.LayerList()
        # ab_ba_da: atom_bond_graph => bond_angle_graph => dihedral_angle_graph
        if self.subgraph_archi == "ab_ba_da":
            self.plane_embedding_list = nn.LayerList()
            self.plane_float_rbf_list = nn.LayerList()
            self.dihedral_angle_float_rbf_list = nn.LayerList()
            self.dihedral_angle_block_list = nn.LayerList()

        for layer_id in range(self.layer_num):
            self.bond_embedding_list.append(
                    BondEmbedding(self.bond_names, self.embed_dim))
            self.bond_float_rbf_list.append(
                    BondFloatRBF(self.bond_float_names, self.embed_dim))
            self.atom_bond_block_list.append(
                GNNBlock(self.embed_dim, self.dropout_rate, last_act=(layer_id != self.layer_num - 1)))
            # ab_ba: atom_bond_graph => bond_angle_graph
            if self.subgraph_archi == "ab_ba" or self.subgraph_archi == "ab_ba_da":
                self.bond_angle_float_rbf_list.append(
                    BondAngleFloatRBF(self.bond_angle_float_names, self.embed_dim))
                self.bond_angle_block_list.append(
                    GNNBlock(self.embed_dim, self.dropout_rate, last_act=(layer_id != self.layer_num - 1)))
            # ab_ba_da: atom_bond_graph => bond_angle_graph => dihedral_angle_graph
            if self.subgraph_archi == "ab_ba_da":
                self.plane_embedding_list.append(
                    PlaneEmbedding(self.plane_names, self.embed_dim))
                self.plane_float_rbf_list.append(
                    PlaneFloatRBF(self.plane_float_names, self.embed_dim))
                self.dihedral_angle_float_rbf_list.append(
                        DihedralAngleFloatRBF(self.dihedral_angle_float_names, self.embed_dim))
                self.dihedral_angle_block_list.append(
                        GNNBlock(self.embed_dim, self.dropout_rate, last_act=(layer_id != self.layer_num - 1)))
        
        # TODO: use self-implemented MeanPool due to pgl bug.
        if self.readout == 'mean':
            self.graph_pool = MeanPool()
        else:
            self.graph_pool = pgl.nn.GraphPool(pool_type=self.readout)

        print('[SMPTGNNModel] subgraph_archi:%s' % self.subgraph_archi)
        print('[SMPTGNNModel] embed_dim:%s' % self.embed_dim)
        print('[SMPTGNNModel] dropout_rate:%s' % self.dropout_rate)
        print('[SMPTGNNModel] layer_num:%s' % self.layer_num)
        print('[SMPTGNNModel] readout:%s' % self.readout)
        print('[SMPTGNNModel] atom_names:%s' % str(self.atom_names))
        print('[SMPTGNNModel] bond_names:%s' % str(self.bond_names))
        print('[SMPTGNNModel] bond_float_names:%s' % str(self.bond_float_names))
        # ab_ba: atom_bond_graph => bond_angle_graph
        if self.subgraph_archi == "ab_ba" or self.subgraph_archi == "ab_ba_da":
            print('[SMPTGNNModel] bond_angle_float_names:%s' % str(self.bond_angle_float_names))
        # ab_ba_da: atom_bond_graph => bond_angle_graph => dihedral_angle_graph
        if self.subgraph_archi == "ab_ba_da":
            print('[SMPTGNNModel] plane_names:%s' % str(self.plane_names))
            print('[SMPTGNNModel] plane_float_names:%s' % str(self.plane_float_names))
            print('[SMPTGNNModel] dihedral_angle_float_names:%s' % str(self.dihedral_angle_float_names))

    @property
    def node_dim(self):
        """the out dim of graph_repr"""
        return self.embed_dim

    @property
    def graph_dim(self):
        """the out dim of graph_repr"""
        return self.embed_dim

    def forward(self, atom_bond_graph, bond_angle_graph, dihedral_angle_graph):
        """
        Build the network.
        """
        # print('    call SMPTGNNModel')
        node_hidden = self.init_atom_embedding(atom_bond_graph.node_feat)

        bond_embed = self.init_bond_embedding(atom_bond_graph.edge_feat)
        edge_hidden = bond_embed + self.init_bond_float_rbf(atom_bond_graph.edge_feat)

        node_hidden_list = [node_hidden]
        edge_hidden_list = [edge_hidden]
        plane_hidden_list = []

        # ag: atom_bond_graph
        if self.subgraph_archi == "ab":
            for layer_id in range(self.layer_num):
                node_hidden = self.atom_bond_block_list[layer_id](
                    atom_bond_graph,
                    node_hidden_list[layer_id],
                    edge_hidden_list[layer_id])

                cur_edge_hidden = self.bond_embedding_list[layer_id](atom_bond_graph.edge_feat)
                edge_hidden = cur_edge_hidden + self.bond_float_rbf_list[layer_id](atom_bond_graph.edge_feat)

                node_hidden_list.append(node_hidden)
                edge_hidden_list.append(edge_hidden)

        # ab_ba: atom_bond_graph => bond_angle_graph
        elif self.subgraph_archi == "ab_ba":
            for layer_id in range(self.layer_num):
                node_hidden = self.atom_bond_block_list[layer_id](
                    atom_bond_graph,
                    node_hidden_list[layer_id],
                    edge_hidden_list[layer_id])

                cur_edge_hidden = self.bond_embedding_list[layer_id](atom_bond_graph.edge_feat)
                cur_edge_hidden = cur_edge_hidden + self.bond_float_rbf_list[layer_id](atom_bond_graph.edge_feat)
                cur_angle_hidden = self.bond_angle_float_rbf_list[layer_id](bond_angle_graph.edge_feat)
                edge_hidden = self.bond_angle_block_list[layer_id](
                    bond_angle_graph,
                    cur_edge_hidden,
                    cur_angle_hidden)

                node_hidden_list.append(node_hidden)
                edge_hidden_list.append(edge_hidden)

        # ab_ba_da: atom_bond_graph => bond_angle_graph => dihedral_angle_graph
        elif self.subgraph_archi == "ab_ba_da":
            plane_embed = self.init_plane_embedding(dihedral_angle_graph.node_feat)
            plane_hidden = plane_embed + self.init_plane_float_rbf(dihedral_angle_graph.node_feat)
            plane_hidden_list.append(plane_hidden)

            for layer_id in range(self.layer_num):
                node_hidden = self.atom_bond_block_list[layer_id](
                    atom_bond_graph,
                    node_hidden_list[layer_id],
                    edge_hidden_list[layer_id])

                cur_edge_hidden = self.bond_embedding_list[layer_id](atom_bond_graph.edge_feat)
                cur_edge_hidden = cur_edge_hidden + self.bond_float_rbf_list[layer_id](atom_bond_graph.edge_feat)
                cur_angle_hidden = self.bond_angle_float_rbf_list[layer_id](bond_angle_graph.edge_feat)
                cur_angle_hidden = cur_angle_hidden + plane_hidden_list[layer_id]
                edge_hidden = self.bond_angle_block_list[layer_id](
                    bond_angle_graph,
                    cur_edge_hidden,
                    cur_angle_hidden)

                cur_plane_hidden = self.plane_embedding_list[layer_id](dihedral_angle_graph.node_feat)
                cur_plane_hidden = cur_plane_hidden + self.plane_float_rbf_list[layer_id](
                    dihedral_angle_graph.node_feat)
                cur_dihedral_angle_hidden = self.dihedral_angle_float_rbf_list[layer_id](dihedral_angle_graph.edge_feat)
                plane_hidden = self.dihedral_angle_block_list[layer_id](
                    dihedral_angle_graph,
                    cur_plane_hidden,
                    cur_dihedral_angle_hidden)

                node_hidden_list.append(node_hidden)
                edge_hidden_list.append(edge_hidden)
                plane_hidden_list.append(plane_hidden)
        else:
            print("Error! subgraph_archi must be set as 'ab', 'ab_ba' or 'ab_ba_da'.")
            exit(0)


        # print("[SMPTGNNModel] forward node_hidden_list: ", node_hidden_list)  # shape=[31, 32],
        # print("[SMPTGNNModel] forward edge_hidden_list: ", edge_hidden_list)  # shape=[97, 32]
        node_repr = node_hidden_list[-1]
        edge_repr = edge_hidden_list[-1]
        plane_repr = []
        if len(plane_hidden_list) != 0: plane_repr = plane_hidden_list[-1]
        graph_repr = self.graph_pool(atom_bond_graph, node_repr)
        # print("[SMPTGNNModel] forward graph_repr: ", graph_repr)  # shape=[2, 32]
        return node_repr, edge_repr, plane_repr, graph_repr


class SMPTPredModel(nn.Layer):
    """tbd"""
    def __init__(self, model_config, compound_encoder_config, compound_encoder):
        super(SMPTPredModel, self).__init__()
        self.subgraph_archi = compound_encoder_config['subgraph_archi']
        self.compound_encoder = compound_encoder
        
        self.hidden_size = model_config['hidden_size']
        self.dropout_rate = model_config['dropout_rate']
        self.act = model_config['act']
        self.pretrain_tasks = model_config['pretrain_tasks']
        # context mask
        if 'Cm' in self.pretrain_tasks:
            self.Cm_vocab = model_config['Cm_vocab']
            self.Cm_linear = nn.Linear(compound_encoder.embed_dim, self.Cm_vocab + 3)
            self.Cm_loss = nn.CrossEntropyLoss()
        # functional group
        self.Fg_linear = nn.Linear(compound_encoder.embed_dim, model_config['Fg_size'])
        self.Fg_loss = nn.BCEWithLogitsLoss()

        # bond length with regression
        if 'Blr' in self.pretrain_tasks:
            self.Blr_mlp = MLP(2,
                    hidden_size=self.hidden_size,
                    act=self.act,
                    in_size=compound_encoder.embed_dim * 2,
                    out_size=1,
                    dropout_rate=self.dropout_rate)
            self.Blr_loss = nn.SmoothL1Loss()
        # atom distance with classification
        if 'Adc' in self.pretrain_tasks:
            self.Adc_vocab = model_config['Adc_vocab']
            self.Adc_mlp = MLP(2,
                    hidden_size=self.hidden_size,
                    in_size=self.compound_encoder.embed_dim * 2,
                    act=self.act,
                    out_size=self.Adc_vocab + 3,
                    dropout_rate=self.dropout_rate)
            self.Adc_loss = nn.CrossEntropyLoss()
        if self.subgraph_archi == "ab_ba" or self.subgraph_archi == "ab_ba_da":
            # bond angle with regression
            if 'Bar' in self.pretrain_tasks:
                self.Bar_mlp = MLP(2,
                                   hidden_size=self.hidden_size,
                                   act=self.act,
                                   in_size=compound_encoder.embed_dim * 3,
                                   out_size=1,
                                   dropout_rate=self.dropout_rate)
                self.Bar_loss = nn.SmoothL1Loss()
        if self.subgraph_archi == "ab_ba_da":
            # dihedral angle with regression
            if 'Dar' in self.pretrain_tasks:
                self.Dar_mlp = MLP(2,
                        hidden_size=self.hidden_size,
                        act=self.act,
                        in_size=compound_encoder.embed_dim * 4,
                        out_size=1,
                        dropout_rate=self.dropout_rate)
                self.Dar_loss = nn.SmoothL1Loss()

    def _get_Cm_loss(self, feed_dict, node_repr):
        # print('    call _get_Cm_loss')
        masked_node_repr = paddle.gather(node_repr, feed_dict['Cm_node_i'])
        # print("[SMPTPredModel] _get_Cm_loss masked_node_repr:", masked_node_repr)  # shape=[4, 32]
        logits = self.Cm_linear(masked_node_repr)
        # print("[SMPTPredModel] _get_Cm_loss logits:", logits)  # shape=[4, 2403]
        loss = self.Cm_loss(logits, feed_dict['Cm_context_id'])
        # print("[SMPTPredModel] _get_Cm_loss loss:", loss)  # shape=[1]
        return loss

    def _get_Fg_loss(self, feed_dict, graph_repr):
        fg_label = paddle.concat(
                [feed_dict['Fg_morgan'], 
                feed_dict['Fg_daylight'], 
                feed_dict['Fg_maccs']], 1)
        logits = self.Fg_linear(graph_repr)
        loss = self.Fg_loss(logits, fg_label)
        return loss

    def _get_Bar_loss(self, feed_dict, node_repr):
        node_i_repr = paddle.gather(node_repr, feed_dict['Ba_node_i'])
        node_j_repr = paddle.gather(node_repr, feed_dict['Ba_node_j'])
        node_k_repr = paddle.gather(node_repr, feed_dict['Ba_node_k'])
        node_ijk_repr = paddle.concat([node_i_repr, node_j_repr, node_k_repr], 1)
        pred = self.Bar_mlp(node_ijk_repr)
        loss = self.Bar_loss(pred, feed_dict['Ba_bond_angle'] / np.pi)
        return loss

    def _get_Blr_loss(self, feed_dict, node_repr):
        node_i_repr = paddle.gather(node_repr, feed_dict['Bl_node_i'])
        node_j_repr = paddle.gather(node_repr, feed_dict['Bl_node_j'])
        node_ij_repr = paddle.concat([node_i_repr, node_j_repr], 1)
        pred = self.Blr_mlp(node_ij_repr)
        loss = self.Blr_loss(pred, feed_dict['Bl_bond_length'])
        return loss

    def _get_Adc_loss(self, feed_dict, node_repr):
        node_i_repr = paddle.gather(node_repr, feed_dict['Ad_node_i'])
        node_j_repr = paddle.gather(node_repr, feed_dict['Ad_node_j'])
        node_ij_repr = paddle.concat([node_i_repr, node_j_repr], 1)
        logits = self.Adc_mlp.forward(node_ij_repr)
        atom_dist = paddle.clip(feed_dict['Ad_atom_dist'], 0.0, 20.0)
        atom_dist_id = paddle.cast(atom_dist / 20.0 * self.Adc_vocab, 'int64')
        loss = self.Adc_loss(logits, atom_dist_id)
        return loss

    def _get_Dar_loss(self, feed_dict, node_repr):
        node_i_repr = paddle.gather(node_repr, feed_dict['Da_node_i'])
        node_m_repr = paddle.gather(node_repr, feed_dict['Da_node_m'])
        node_n_repr = paddle.gather(node_repr, feed_dict['Da_node_n'])
        node_j_repr = paddle.gather(node_repr, feed_dict['Da_node_j'])
        node_imnj_repr = paddle.concat([node_i_repr, node_m_repr, node_n_repr, node_j_repr], 1)
        pred = self.Dar_mlp(node_imnj_repr)
        loss = self.Dar_loss(pred, feed_dict['Da_dihedral_angle'] / np.pi)
        return loss

    def forward(self, graph_dict, feed_dict, return_subloss=False):
        """
        Build the network.
        """
        # print('  call SMPTPredModel')

        node_repr, edge_repr, plane_repr, graph_repr = self.compound_encoder.forward(
                graph_dict['atom_bond_graph'], graph_dict['bond_angle_graph'], graph_dict['dihedral_angle_graph'])
        masked_node_repr, masked_edge_repr, masked_plane_repr, masked_graph_repr = self.compound_encoder.forward(
                graph_dict['masked_atom_bond_graph'], graph_dict['masked_bond_angle_graph'], graph_dict['masked_dihedral_angle_graph'])

        sub_losses = {}
        if 'Cm' in self.pretrain_tasks:
            sub_losses['Cm_loss'] = self._get_Cm_loss(feed_dict, node_repr)
            sub_losses['Cm_loss'] += self._get_Cm_loss(feed_dict, masked_node_repr)
        if 'Fg' in self.pretrain_tasks:
            sub_losses['Fg_loss'] = self._get_Fg_loss(feed_dict, graph_repr)
            sub_losses['Fg_loss'] += self._get_Fg_loss(feed_dict, masked_graph_repr)
        if 'Blr' in self.pretrain_tasks:
            sub_losses['Blr_loss'] = self._get_Blr_loss(feed_dict, node_repr)
            sub_losses['Blr_loss'] += self._get_Blr_loss(feed_dict, masked_node_repr)
        if 'Adc' in self.pretrain_tasks:
            sub_losses['Adc_loss'] = self._get_Adc_loss(feed_dict, node_repr)
            sub_losses['Adc_loss'] += self._get_Adc_loss(feed_dict, masked_node_repr)
        if self.subgraph_archi == "ab_ba" or self.subgraph_archi == "ab_ba_da":
            if 'Bar' in self.pretrain_tasks:
                sub_losses['Bar_loss'] = self._get_Bar_loss(feed_dict, node_repr)
                sub_losses['Bar_loss'] += self._get_Bar_loss(feed_dict, masked_node_repr)
        if self.subgraph_archi == "ab_ba_da":
            if 'Dar' in self.pretrain_tasks:
                sub_losses['Dar_loss'] = self._get_Dar_loss(feed_dict, node_repr)
                sub_losses['Dar_loss'] += self._get_Dar_loss(feed_dict, masked_node_repr)

        loss = 0
        for name in sub_losses:
            loss += sub_losses[name]
        if return_subloss:
            return loss, sub_losses
        else:
            return loss