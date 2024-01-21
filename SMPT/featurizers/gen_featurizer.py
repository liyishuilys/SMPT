

"""
| Featurizers for pretrain-gnn.

| Adapted from https://github.com/snap-stanford/pretrain-gnns/tree/master/chem/utils.py
"""

import numpy as np
import networkx as nx
from copy import deepcopy
import pgl
from rdkit.Chem import AllChem

from sklearn.metrics import pairwise_distances
import hashlib
from utils.compound_tools import *



def md5_hash(string):
    """tbd"""
    md5 = hashlib.md5(string.encode('utf-8')).hexdigest()
    return int(md5, 16)


def mask_context_of_geognn_graph(
        g, 
        superedge_g,
        superplane_g,
        edges,
        plane_atoms,
        target_atom_indices=None, 
        mask_ratio=None, 
        mask_value=0, 
        subgraph_num=None,
        version='gem'):

    """tbd"""
    # print('      call mask_context_of_geognn_graph')

    def get_subgraph_str(g, superplane_g, atom_index, nei_atom_indices, nei_bond_indices, nei_plane_indices):
        """tbd"""
        atomic_num = g.node_feat['atomic_num'].flatten()  # flatten() convert into one-dimension array
        bond_type = g.edge_feat['bond_type'].flatten()
        plane_in_ring = superplane_g.node_feat['plane_in_ring'].flatten()
        subgraph_str = 'A' + str(atomic_num[atom_index])
        subgraph_str += 'N' + ':'.join([str(x) for x in np.sort(atomic_num[nei_atom_indices])])
        subgraph_str += 'E' + ':'.join([str(x) for x in np.sort(bond_type[nei_bond_indices])])
        subgraph_str += 'P' + ':'.join([str(x) for x in np.sort(plane_in_ring[nei_plane_indices])])
        return subgraph_str

    g = deepcopy(g)
    N = g.num_nodes
    # E = g.num_edges
    E = superedge_g.num_nodes
    P = superplane_g.num_nodes
    full_atom_indices = np.arange(N)
    full_bond_indices = np.arange(E)
    full_plane_indices = np.arange(P)

    """
    Randomly generate target atom indices based on mask_size for masking
    """
    if target_atom_indices is None:
        masked_size = max(1, int(N * mask_ratio))   # at least 1 atom will be selected.
        target_atom_indices = np.random.choice(full_atom_indices, size=masked_size, replace=False)
        # target_atom_indices = [3, 6]
        # print('        target_atom_indices: ', target_atom_indices)  # [3 6]
    """
    Loop over target masked atom indices
    Mask target atoms and their neighbour atoms, edges and planes
    Generate label for each target atom
    """
    target_labels = []
    Cm_node_i = []
    masked_bond_indices = []
    masked_plane_indices = []

    """
    Find Cm_node_i, masked_bond_indices, masked_plane_indices, target_labels
    Cm_node_i: target masked atom indices, neighbour atom indices of target masked atom
    masked_bond_indices: bond indices associated with target masked atom
    masked_plane_indices: plane indices associated with target masked atom
    """
    for atom_index in target_atom_indices:
        # print('          target_atom: ', atom_index)  # 3 || 6


        """
        Find neighbour bond indices based on target atom index
        """
        # left_nei_bond_indices = full_bond_indices[g.edges[:, 0] == atom_index]
        left_nei_bond_indices = full_bond_indices[edges[:, 0] == atom_index]  # find bound that tatget atom being left neighbour
        # print('            left_nei_bond_indices: ', left_nei_bond_indices)  # [ 5  6  8 33]  ||  [11 12 29 36]
        # right_nei_bond_indices = full_bond_indices[g.edges[:, 1] == atom_index]
        right_nei_bond_indices = full_bond_indices[edges[:, 1] == atom_index]  # find bound that tatget atom being right neighbour
        # print('            right_nei_bond_indices: ', right_nei_bond_indices)  # [ 4  7  9 33]  || [10 13 28 36]
        left_nei_bond_indices = np.random.choice(left_nei_bond_indices, size=int(np.ceil(len(left_nei_bond_indices) * 0.5)), replace=False)
        right_nei_bond_indices = np.random.choice(right_nei_bond_indices, size=int(np.ceil(len(right_nei_bond_indices) * 0.5)), replace=False)
        nei_bond_indices = np.append(left_nei_bond_indices, right_nei_bond_indices)  # find all bound that contain target atom


        """
        Find all neighbour atom indices of target atom based on nei_bond_indices
        """
        left_nei_atom_indices = g.edges[left_nei_bond_indices, 1]  # find right neighbour based on left_nei_bond_indices
        # print('            left_nei_atom_indices: ', left_nei_atom_indices)  # [2 4 5 3]  ||  [ 5  7 10  6]
        right_nei_atom_indices = g.edges[right_nei_bond_indices, 0]  # find left neighbour based on right_nei_bond_indices
        # print('            right_nei_atom_indices: ', right_nei_atom_indices)  # [2 4 5 3]  || [ 5  7 10  6]
        nei_atom_indices = np.append(left_nei_atom_indices, right_nei_atom_indices)  # find all neighbour of target atom


        """
        Find plane indices contain target atom index
        """
        a0_nei_plane_indices = full_plane_indices[plane_atoms[:, 0] == atom_index]  # find plane that tatget atom being a0 neighbour
        a1_nei_plane_indices = full_plane_indices[plane_atoms[:, 1] == atom_index]  # find plane that tatget atom being a1 neighbour
        b0_nei_plane_indices = full_plane_indices[plane_atoms[:, 2] == atom_index]  # find plane that tatget atom being b0 neighbour
        b1_nei_plane_indices = full_plane_indices[plane_atoms[:, 3] == atom_index]  # find plane that tatget atom being b1 neighbour
        nei_plane_indices = np.append(a0_nei_plane_indices, a1_nei_plane_indices)  # find all plane that contain target atom
        nei_plane_indices = np.append(nei_plane_indices, b0_nei_plane_indices)
        nei_plane_indices = np.append(nei_plane_indices, b1_nei_plane_indices)
        nei_plane_indices = np.random.choice(nei_plane_indices, size=int(np.ceil(len(nei_plane_indices) * 0.5)), replace=False)

        """
        Generate subgraph str for target atom based on atom_index, nei_atom_indices, nei_bond_indices
        """
        if version == 'gem':
            subgraph_str = get_subgraph_str(g, superplane_g, atom_index, nei_atom_indices, nei_bond_indices, nei_plane_indices)
            # print('            subgraph_str: ', subgraph_str)  # A6N6:6:7:7:7:7:8:8E3:3:13:13:13:13:24:24  ||  A6N6:6:6:6:7:7:8:8E2:2:2:2:2:2:24:24
            subgraph_id = md5_hash(subgraph_str) % subgraph_num
            # print('            subgraph_id: ', subgraph_id)  # 419  || 1219
            target_label = subgraph_id
        else:
            raise ValueError(version)


        """
        Respectively construct lists of target masked atoms, neighbour atoms, neighbour bonds, labels
        """
        Cm_node_i.append([atom_index])  # list of target masked atom indices
        Cm_node_i.append(nei_atom_indices)  # add neighbour atom indices for each target masked atom to the above list
        masked_bond_indices.append(nei_bond_indices)  # lists of neighbour bond indices for each target masked atom
        masked_plane_indices.append(nei_plane_indices)   # lists of neighbour plane indices for each target masked atom
        target_labels.append(target_label)  # lists of label for each target masked atom
        # print('            Cm_node_i (target & neighbour atom) : ', Cm_node_i)  # [[3], array([2, 4, 5, 3, 2, 4, 5, 3]), [6], array([ 5,  7, 10,  6,  5,  7, 10,  6])]
        # print('            masked_bond_indices: ', masked_bond_indices)  # [array([ 5,  6,  8, 33,  4,  7,  9, 33]), array([11, 12, 29, 36, 10, 13, 28, 36])]
        # print('            target_labels: ', target_labels)  # [419, 1219]

    target_atom_indices = np.array(target_atom_indices)  # convert to array
    Cm_node_i = np.concatenate(Cm_node_i, 0)  # convert to N*1 array
    masked_bond_indices = np.concatenate(masked_bond_indices, 0)  # convert to N*1 array
    masked_plane_indices = np.concatenate(masked_plane_indices, 0)  # convert to N*1 array
    target_labels = np.array(target_labels)  # convert to array


    """
    =====================
    ab_g mask
    =====================
    """
    """
    Atom-bond graph: node is atom, edge is bond
    Mask node feats and edge feats of the atom-bond graph based on Cm_node_i and masked_bond_indices
    Mask target atom and all neighbour, set corresponding node and edge feat value to 0
    """
    # print("          before mask, g.node_feat.keys(): ", g.node_feat.keys())
    # print("          before mask, g.node_feat['atomic_num']: ", g.node_feat['atomic_num'])
    # print("          before mask, g.edge_feat.keys(): ", g.edge_feat.keys())
    # print("          before mask, g.edge_feat['bond_length']: ", g.edge_feat['bond_length'])
    for name in g.node_feat:
        g.node_feat[name][Cm_node_i] = mask_value
    for name in g.edge_feat:
        g.edge_feat[name][masked_bond_indices] = mask_value
    # print("          after mask, g.node_feat['atomic_num']: ", g.node_feat['atomic_num'])
    # print("          after mask, g.edge_feat['bond_length']: ", g.edge_feat['bond_length'])
    """
    g.node_feat.keys():
        dict_keys(['atomic_num', 'formal_charge', 'degree', 'chiral_tag', 'total_numHs', 'is_aromatic', 'hybridization'])

    g.edge_feat.keys():
        dict_keys(['bond_dir', 'bond_type', 'is_in_ring', 'bond_length'])

    g.node_feat['atomic_num']
    mol 1 before mask:  
        [[8] [6] [7] [6] [8] [7] [6] [6] [6] [6] [8] [6] [6] [9]]
    mol 1 after mask:  Cm_node_i=>[[3], array([2, 4, 5, 3, 2, 4, 5, 3]), [6], array([ 5,  7, 10,  6,  5,  7, 10,  6])]
        [[8] [6] [0] [0] [0] [0] [0] [0] [6] [6] [0] [6] [6] [9]]
    mol 2 before mask:
        [[8] [6] [7] [6] [6] [6] [6] [6] [8] [6] [6] [6] [8] [7] [6] [8] [7]]
    mol 2 after mask:  Cm_node_i=>[[3], array([2, 4, 3, 2, 4, 3]), [6], array([5, 7, 6, 5, 7, 6])]
        [[8] [6] [0] [0] [0] [0] [0] [0] [8] [6] [6] [6] [8] [7] [6] [8] [7]]

    g.edge_feat['bond_length']
    before mask:
        [[1.2325547] [1.2325547] [1.3345274] [1.3345274] [1.3373295] [1.3373295] [1.2345552] [1.2345552] [1.4093939] 
        [1.4093939] [1.4610093] [1.4610093] [1.5282334] [1.5282334] [1.5070841] [1.5070841] [1.509946 ] [1.509946 ] 
        [1.4312733] [1.4312733] [1.3717738] [1.3717738] [1.3252139] [1.3252139] [1.3478078] [1.3478078] [1.4771085] 
        [1.4771085] [1.449182 ] [1.449182 ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] 
        [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ]]
    after mask:  masked_bond_indices=>[array([ 5,  6,  8, 33,  4,  7,  9, 33]), array([11, 12, 29, 36, 10, 13, 28, 36])]
        [[1.2325547] [1.2325547] [1.3345274] [1.3345274] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] 
        [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [1.5070841] [1.5070841] [1.509946 ] [1.509946 ] 
        [1.4312733] [1.4312733] [1.3717738] [1.3717738] [1.3252139] [1.3252139] [1.3478078] [1.3478078] [1.4771085] 
        [1.4771085] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] 
        [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ]]
    """

    """
    =====================
    ba_g mask
    =====================
    """
    """
    Bond-angle graph: node is bond, edge is angle
    Mask edge feats of the bond-angle graph based on masked bond indices
    Mask angle associated with mashed bond indices
    """
    full_superedge_indices = np.arange(superedge_g.num_edges)
    # print("          superedge_g.num_edges: ", superedge_g.num_edges)  # 132
    masked_superedge_indices = []
    for bond_index in masked_bond_indices:
        left_indices = full_superedge_indices[superedge_g.edges[:, 0] == bond_index]
        right_indices = full_superedge_indices[superedge_g.edges[:, 1] == bond_index]
        masked_superedge_indices.append(np.append(left_indices, right_indices))
    masked_superedge_indices = np.concatenate(masked_superedge_indices, 0)
    # print("          before mask, superedge_g.edge_feat.keys(): ", superedge_g.edge_feat.keys())
    # print("          before mask, superedge_g.edge_feat['bond_angle']: ", superedge_g.edge_feat['bond_angle'])
    for name in superedge_g.edge_feat:
        superedge_g.edge_feat[name][masked_superedge_indices] = mask_value
    # print("          after mask, superedge_g.edge_feat['bond_angle']: ", superedge_g.edge_feat['bond_angle'])
    """
    superedge_g.edge_feat.keys():
        dict_keys(['bond_angle'])
    
    superedge_g.edge_feat['bond_angle']
    before mask:
        [[3.1359193] [0.       ] [3.1359193] [0.9761861] [1.0863941] [0.       ] [0.9761861] [3.13609  ] [1.0790395] 
        [0.       ] [3.13609  ] [1.0067631] [0.       ] [1.0067631] [3.1361227] [0.       ] [3.1361227] [1.0120649] 
        [1.0631546] [0.       ] [1.0120649] [3.1359088] [1.0664213] [0.       ] [3.1359088] [0.       ] [1.0631546] 
        [1.0664213] [3.1362662] [0.       ] [3.1362662] [1.1005127] [1.0291058] [0.       ] [1.1005127] [3.1363451] 
        [1.0123228] [0.       ] [3.1363451] [1.1732631] [1.197066 ] [0.       ] [1.1732631] [3.1364832] [1.2893195] 
        [0.       ] [3.1364832] [1.3498893] [0.       ] [1.3498893] [3.1364367] [0.       ] [3.1364367] [1.3681307] 
        [0.       ] [1.3681307] [3.1364253] [0.       ] [3.1364253] [1.2728226] [0.       ] [1.2728226] [3.136311 ] 
        [0.       ] [3.136311 ] [1.2475502] [0.       ] [1.0291058] [1.0123228] [3.1361883] [0.       ] [3.1361883] 
        [1.0552311] [0.       ] [1.0552311] [3.1361117] [0.       ] [3.1361117] [1.0090687] [1.0500273] [0.       ] 
        [1.0090687] [3.1361444] [1.0825477] [0.       ] [3.1361444] [0.       ] [1.0500273] [1.0825477] [3.1363792] 
        [0.       ] [1.0863941] [1.0790395] [3.1363792] [0.       ] [1.2475502] [3.1363451] [0.       ] [1.197066 ] 
        [1.2893195] [3.1363451] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] 
        [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] 
        [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] 
        [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ]]
    after mask:
        [[3.1359193] [0.       ] [3.1359193] [0.9761861] [1.0863941] [0.       ] [0.9761861] [3.13609  ] [1.0790395] 
        [0.       ] [3.13609  ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] 
        [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] 
        [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] 
        [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] 
        [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [3.1364367] [0.       ] [3.1364367] [1.3681307] 
        [0.       ] [1.3681307] [3.1364253] [0.       ] [3.1364253] [1.2728226] [0.       ] [1.2728226] [3.136311 ] 
        [0.       ] [3.136311 ] [0.       ] [0.       ] [0.       ] [0.       ] [3.1361883] [0.       ] [3.1361883] 
        [1.0552311] [0.       ] [1.0552311] [3.1361117] [0.       ] [3.1361117] [1.0090687] [1.0500273] [0.       ] 
        [1.0090687] [3.1361444] [1.0825477] [0.       ] [3.1361444] [0.       ] [1.0500273] [1.0825477] [3.1363792] 
        [0.       ] [1.0863941] [1.0790395] [3.1363792] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] 
        [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] 
        [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] 
        [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] 
        [0.       ] [0.       ] [0.       ] [0.       ] [0.       ] [0.       ]]
    
    """


    """
    =====================
    da_g mask
    =====================
    """
    """
    Dihedral-angle graph: node is plane, edge is angle
    Mask edge feats of the dihedral-angle graph based on masked plane indices
    Mask angle associated with mashed plane indices
    """
    full_superplane_indices = np.arange(superplane_g.num_edges)
    masked_superplane_indices = []
    for plane_index in masked_plane_indices:
        left_indices = full_superplane_indices[superplane_g.edges[:, 0] == plane_index]
        right_indices = full_superplane_indices[superplane_g.edges[:, 1] == plane_index]
        masked_superplane_indices.append(np.append(left_indices, right_indices))
    masked_superplane_indices = np.concatenate(masked_superplane_indices, 0)
    for name in superplane_g.edge_feat:
        superplane_g.edge_feat[name][masked_superplane_indices] = mask_value



    return [g, superedge_g, superplane_g, target_atom_indices, target_labels]
    

def get_pretrain_bond_angle(edges, atom_poses):
    """tbd"""
    def _get_angle(vec1, vec2):
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0
        vec1 = vec1 / (norm1 + 1e-5)    # 1e-5: prevent numerical errors
        vec2 = vec2 / (norm2 + 1e-5)
        angle = np.arccos(np.dot(vec1, vec2))
        return angle
    def _add_item(
            node_i_indices, node_j_indices, node_k_indices, bond_angles, 
            node_i_index, node_j_index, node_k_index):
        node_i_indices += [node_i_index, node_k_index]
        node_j_indices += [node_j_index, node_j_index]
        node_k_indices += [node_k_index, node_i_index]
        pos_i = atom_poses[node_i_index]
        pos_j = atom_poses[node_j_index]
        pos_k = atom_poses[node_k_index]
        angle = _get_angle(pos_i - pos_j, pos_k - pos_j)
        bond_angles += [angle, angle]

    E = len(edges)
    node_i_indices = []
    node_j_indices = []
    node_k_indices = []
    bond_angles = []
    for edge_i in range(E - 1):
        for edge_j in range(edge_i + 1, E):
            a0, a1 = edges[edge_i]
            b0, b1 = edges[edge_j]
            if a0 == b0 and a1 == b1:  # [0, 1], [0, 1]
                continue
            if a0 == b1 and a1 == b0:  # [0, 1], [1, 0]
                continue
            if a0 == b0:  # [1, 0], [1, 2] => 0-1-2
                _add_item(
                        node_i_indices, node_j_indices, node_k_indices, bond_angles,
                        a1, a0, b1)
            if a0 == b1:  # [1, 0], [2, 1] => 0-1-2
                _add_item(
                        node_i_indices, node_j_indices, node_k_indices, bond_angles,
                        a1, a0, b0)
            if a1 == b0:  # [1, 2], [2, 3] => 1-2-3
                _add_item(
                        node_i_indices, node_j_indices, node_k_indices, bond_angles,
                        a0, a1, b1)
            if a1 == b1:  # [1, 2], [3, 2] => 1-2-3
                _add_item(
                        node_i_indices, node_j_indices, node_k_indices, bond_angles,
                        a0, a1, b0)
    node_ijk = np.array([node_i_indices, node_j_indices, node_k_indices])
    uniq_node_ijk, uniq_index = np.unique(node_ijk, return_index=True, axis=1)  # order node_i_indices
    node_i_indices, node_j_indices, node_k_indices = uniq_node_ijk
    bond_angles = np.array(bond_angles)[uniq_index]
    return [node_i_indices, node_j_indices, node_k_indices, bond_angles]


class PredTransformFn(object):
    """Gen features for downstream model"""
    def __init__(self, pretrain_tasks, mask_ratio):
        self.pretrain_tasks = pretrain_tasks
        self.mask_ratio = mask_ratio

    def prepare_pretrain_task(self, data):
        """
        prepare data for pretrain task
        """

        # calculate angle between two edges based on atom_pos
        node_i, node_j, node_k, bond_angles = get_pretrain_bond_angle(data['edges'], data['atom_pos'])  # atom_pos is the coordinate

        # angle among Ba_node_i, Ba_node_j, Ba_node_k
        data['Ba_node_i'] = node_i
        data['Ba_node_j'] = node_j
        data['Ba_node_k'] = node_k
        data['Ba_bond_angle'] = bond_angles

        # get bond length between Bl_node_i and Bl_node_j
        data['Bl_node_i'] = np.array(data['edges'][:, 0])
        data['Bl_node_j'] = np.array(data['edges'][:, 1])
        data['Bl_bond_length'] = np.array(data['bond_length'])

        # calculate distance between any two atoms
        n = len(data['atom_pos'])
        dist_matrix = pairwise_distances(data['atom_pos'])
        indice = np.repeat(np.arange(n).reshape([-1, 1]), n, axis=1)
        data['Ad_node_i'] = indice.reshape([-1, 1])
        data['Ad_node_j'] = indice.T.reshape([-1, 1])
        data['Ad_atom_dist'] = dist_matrix.reshape([-1, 1])

        return data

    def __call__(self, raw_data):
        """
        Gen features according to raw data and return a single graph data.
        Args:
            raw_data: It contains smiles and label,we convert smiles 
            to mol by rdkit,then convert mol to graph data.
        Returns:
            data: It contains reshape label and smiles.
        """
        smiles = raw_data
        mol = AllChem.MolFromSmiles(smiles)

        if mol is None:
            return None
        data = mol_to_geognn_graph_data_MMFF3d(mol)
        data['smiles'] = smiles
        data = self.prepare_pretrain_task(data)
        return data


class PredCollateFn(object):
    """tbd"""
    def __init__(self,
             atom_names,
             bond_names,
             bond_float_names,
             bond_angle_float_names,
             plane_names,
             plane_float_names,
             dihedral_angle_float_names,
             pretrain_tasks, 
             mask_ratio,
             Cm_vocab):
        self.atom_names = atom_names  # atomic_num, formal_charge, degree, chiral_tag, etc.
        self.bond_names = bond_names  # bond_dir, bond_type, is_in_ring
        self.bond_float_names = bond_float_names  # bond_length
        self.bond_angle_float_names = bond_angle_float_names  # bond_angle
        self.plane_names = plane_names  # plane_in_ring
        self.plane_float_names = plane_float_names  # plane_mass
        self.dihedral_angle_float_names = dihedral_angle_float_names  # DihedralAngleGraph_angles
        self.pretrain_tasks = pretrain_tasks  # "Cm", "Fg", "Bar", "Blr", "Adc"
        self.mask_ratio = mask_ratio  # 0.15
        self.Cm_vocab = Cm_vocab  # 2400


        print('[PredCollateFn] atom_names:%s' % self.atom_names)
        print('[PredCollateFn] bond_names:%s' % self.bond_names)
        print('[PredCollateFn] bond_float_names:%s' % self.bond_float_names)
        print('[PredCollateFn] bond_angle_float_names:%s' % self.bond_angle_float_names)
        print('[PredCollateFn] plane_names:%s' % self.plane_names)
        print('[PredCollateFn] plane_float_names:%s' % self.plane_float_names)
        print('[PredCollateFn] dihedral_angle_float_names:%s' % self.dihedral_angle_float_names)
        print('[PredCollateFn] pretrain_tasks:%s' % self.pretrain_tasks)
        print('[PredCollateFn] mask_ratio:%s' % self.mask_ratio)
        print('[PredCollateFn] Cm_vocab:%s' % self.Cm_vocab)


        
    def _flat_shapes(self, d):
        """TODO: reshape due to pgl limitations on the shape"""
        for name in d:
            d[name] = d[name].reshape([-1])

    def __call__(self, batch_data_list):
        """tbd"""
        # print('  call PredCollateFn')

        # graph list
        atom_bond_graph_list = []
        bond_angle_graph_list = []
        dihedral_angle_graph_list = []
        masked_atom_bond_graph_list = []
        masked_bond_angle_graph_list = []
        masked_dihedral_angle_graph_list = []

        # current mask node
        Cm_node_i = []  # current mask node id
        Cm_context_id = []  # current mask node context

        # fingerprint
        Fg_morgan = []  # morgan fingerprint
        Fg_daylight = []  # daylight fingerprint
        Fg_maccs = []  # maccs fingerprint

        # three nodes, two bond, one angle
        Ba_node_i = []  # bond-atom node i
        Ba_node_j = []  # bond-atom node j
        Ba_node_k = []  # bond-atom node k
        Ba_bond_angle = []  # bond-atom node i, j, k angle

        # two nodes, one bond, one length
        Bl_node_i = []  # bond-length node i
        Bl_node_j = []  # bond-length node j
        Bl_bond_length = []  # bond-length node i, j length

        # two nodes, one distance
        Ad_node_i = []  # atom-distance node i
        Ad_node_j = []  # atom-distance node j
        Ad_atom_dist = []  # atom-distance node i, j distance

        # four nodes, two planes share on edge, on angle
        Da_node_i = []  # dihedral-angle node i
        Da_node_m = []  # dihedral-angle node m
        Da_node_n = []  # dihedral-angle node n
        Da_node_j = []  # dihedral-angle node j
        Da_dihedral_angle = []  # dihedral-angle, plane 0&1, angle


        node_count = 0
        ii = 0
        for data in batch_data_list:  # one data is one molecular
            ii += 1
            # print('    process molecular: ', ii,)
            # print('      atomic_num: ', data['atomic_num'])
            N = len(data[self.atom_names[0]])  # atom count
            E = len(data['edges'])  # edge count
            P = len(data['BondAngleGraph_edges'])  # plane count
            # print('      atom count: ', N, ', edge count: ', E, ', plane count: ', P)  # 14, 44  ||  17, 53

            """
            Construct ab_g (atom-bond graph), ba_g (bond-angle graph), da_g (dihedral-angle graph)
            ab_g -> node_feat: {'atomic_num': array([[],[],[],...]), 'formal_charge': array([[],[],[],...]), ... }
            ab_g -> edge_feat: {'bond_dir': array([[],[],[],...]), 'bond_type': array([[],[],[],...]), ..., 'bond_length': array([[],[],[],...])}
            
            ba_g -> edge_feat: {'bond_angle': array([[],[],[],...])}
            
            da_g -> edge_feat: {'DihedralAngleGraph_angles': array([[],[],[],...])}
            
            ab_g: {"class": "Graph", "num_nodes": 14, "edges_shape": [44, 2], "node_feat": [{"name": "atomic_num", "shape": [14, 1], "dtype": "int64"}, {"name": "formal_charge", "shape": [14, 1], "dtype": "int64"}, ...]}
            ba_g: {"class": "Graph", "num_nodes": 44, "edges_shape": [132, 2], "node_feat": [], "edge_feat": [{"name": "bond_angle", "shape": [132, 1], "dtype": "float32"}]}
            da_g: {"class": "Graph", "num_nodes": **, "edges_shape": [**, **], "node_feat": [], "edge_feat": [{"name": "DihedralAngleGraph_angles", "shape": [**, 1], "dtype": "float32"}]}
            """
            ab_g = pgl.graph.Graph(
                    num_nodes=N,
                    edges=data['edges'],
                    node_feat={name: data[name].reshape([-1, 1]) for name in self.atom_names},
                    edge_feat={name: data[name].reshape([-1, 1]) for name in self.bond_names + self.bond_float_names})
            ba_g = pgl.graph.Graph(
                    num_nodes=E,
                    edges=data['BondAngleGraph_edges'],
                    node_feat={},
                    edge_feat={name: data[name].reshape([-1, 1]) for name in self.bond_angle_float_names})
            da_g = pgl.graph.Graph(
                    num_nodes=P,
                    edges=data['DihedralAngleGraph_edges'],
                    node_feat={name: data[name].reshape([-1, 1]) for name in self.plane_names + self.plane_float_names},
                    edge_feat={name: data[name].reshape([-1, 1]) for name in self.dihedral_angle_float_names})
            atom_bond_graph_list.append(ab_g)
            bond_angle_graph_list.append(ba_g)
            dihedral_angle_graph_list.append(da_g)

            """
            Contruct mased_ab_g, masked_ba_g, masked_da_g, mask_node_i, context_id
            """
            edges = data['edges']
            plane_atoms = data['plane_atoms']
            # mask_context_of_geognn_graph()
            # return [g, superedge_g, superplane_g, target_atom_indices, target_labels]
            masked_ab_g, masked_ba_g, masked_da_g, mask_node_i, context_id = mask_context_of_geognn_graph(
                    ab_g,
                    ba_g,
                    da_g,
                    edges,
                    plane_atoms,
                    mask_ratio=self.mask_ratio,
                    subgraph_num=self.Cm_vocab)
            masked_atom_bond_graph_list.append(masked_ab_g)
            masked_bond_angle_graph_list.append(masked_ba_g)
            masked_dihedral_angle_graph_list.append(masked_da_g)

            # print('      prepare mask id & label for pretrain tasks')
            if 'Cm' in self.pretrain_tasks:
                # print('        mask_node_i: ', mask_node_i)  # [3 6]
                # print('        node_count: ', node_count)  # 0
                Cm_node_i.append(mask_node_i + node_count)  # [array([3, 6])]
                Cm_context_id.append(context_id)  #
                # print('        Cm_node_i: ', Cm_node_i)  # [array([3, 6])]
                # print('        Cm_context_id: ', Cm_context_id)  # [array([ 419, 1219])]
            if 'Fg' in self.pretrain_tasks:
                Fg_morgan.append(data['morgan_fp'])
                Fg_daylight.append(data['daylight_fg_counts'])
                Fg_maccs.append(data['maccs_fp'])
            if 'Bar' in self.pretrain_tasks:
                Ba_node_i.append(data['Ba_node_i'] + node_count)
                Ba_node_j.append(data['Ba_node_j'] + node_count)
                Ba_node_k.append(data['Ba_node_k'] + node_count)
                Ba_bond_angle.append(data['Ba_bond_angle'])
            if 'Blr' in self.pretrain_tasks:
                Bl_node_i.append(data['Bl_node_i'] + node_count)
                Bl_node_j.append(data['Bl_node_j'] + node_count)
                Bl_bond_length.append(data['Bl_bond_length'])
            if 'Adc' in self.pretrain_tasks:
                Ad_node_i.append(data['Ad_node_i'] + node_count)
                Ad_node_j.append(data['Ad_node_j'] + node_count)
                Ad_atom_dist.append(data['Ad_atom_dist'])
            if 'Dar' in self.pretrain_tasks:
                Da_node_i.append(data['Da_node_i'] + node_count)
                Da_node_m.append(data['Da_node_m'] + node_count)
                Da_node_n.append(data['Da_node_n'] + node_count)
                Da_node_j.append(data['Da_node_j'] + node_count)
                Da_dihedral_angle.append(data['Da_dihedral_angle'])

            node_count += N  # 0+17=17


        graph_dict = {}    
        feed_dict = {}

        """
        atom-bond graph graph_dict, feed_dict
        """
        # print('  construct graph dict')
        atom_bond_graph = pgl.Graph.batch(atom_bond_graph_list)
        self._flat_shapes(atom_bond_graph.node_feat)
        self._flat_shapes(atom_bond_graph.edge_feat)
        # print('    atom_bond_graph.node_feat: ', str(atom_bond_graph.node_feat))
        # print("    atom_bond_graph.node_feat['atomic_num']: ", str(atom_bond_graph.node_feat['atomic_num']))
        # print('    atom_bond_graph.edge_feat: ', str(atom_bond_graph.edge_feat))
        # print("    atom_bond_graph.edge_feat['bond_dir']: ", str(atom_bond_graph.edge_feat['bond_dir']))
        graph_dict['atom_bond_graph'] = atom_bond_graph
        """
        atom_bond_graph.node_feat:  {'atomic_num': array([8, 6, 7, 6, 8, 7, 6, 6, 6, 6, 8, 6, 6, 9, 8, 6, 7, 6, 6, 6, 6, 6,
       8, 6, 6, 6, 8, 7, 6, 8, 7]), 'formal_charge': array([6, 6, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
       6, 6, 6, 6, 6, 5, 6, 6, 6]), 'degree': array([2, 4, 3, 4, 2, 4, 4, 3, 3, 3, 3, 3, 4, 2, 2, 4, 3, 3, 4, 3, 3, 3,
       3, 4, 3, 4, 2, 3, 4, 2, 3]), 'chiral_tag': array([1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1]), 'total_numHs': array([1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 1, 2, 1, 1, 1, 1, 2, 3, 1, 2, 2, 2,
       1, 1, 2, 1, 1, 1, 1, 1, 2]), 'is_aromatic': array([1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2,
       2, 2, 2, 2, 1, 2, 2, 1, 2]), 'hybridization': array([4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 4, 4, 5, 4, 4, 4, 5, 4, 4, 4, 4,
       4, 4, 4, 4, 4, 4, 4, 4, 4])}
       
        atom_bond_graph.edge_feat:  {'bond_dir': array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9, 9, 9,
       9, 9, 9, 9, 9, 9, 9, 9, 9]), ...}
        """

        """
        bond-angle graph graph_dict, feed_dict
        """
        bond_angle_graph = pgl.Graph.batch(bond_angle_graph_list)
        self._flat_shapes(bond_angle_graph.node_feat)
        self._flat_shapes(bond_angle_graph.edge_feat)
        # print('    bond_angle_graph.node_feat: ', str(bond_angle_graph.node_feat))
        # print("    bond_angle_graph.edge_feat['bond_angle'][:5]: ", str(bond_angle_graph.edge_feat['bond_angle'][:5]))
        graph_dict['bond_angle_graph'] = bond_angle_graph
        """
        bond_angle_graph.node_feat:  {}
        
        bond_angle_graph.edge_feat:  
        {'bond_angle': array([3.1359193 , 0.        , 3.1359193 , 0.9761861 , 1.0863941 ,
            0.        , 0.9761861 , 3.13609   , 1.0790395 , 0.        , ... ], dtype=float32)}
        """

        """
        dihedral-angle graph graph_dict, feed_dict
        """
        dihedral_angle_graph = pgl.Graph.batch(dihedral_angle_graph_list)
        self._flat_shapes(dihedral_angle_graph.node_feat)
        self._flat_shapes(dihedral_angle_graph.edge_feat)
        graph_dict['dihedral_angle_graph'] = dihedral_angle_graph


        """
        masked atom-bond graph graph_dict, feed_dict
        """
        masked_atom_bond_graph = pgl.Graph.batch(masked_atom_bond_graph_list)
        self._flat_shapes(masked_atom_bond_graph.node_feat)
        self._flat_shapes(masked_atom_bond_graph.edge_feat)
        # print("    masked_atom_bond_graph.node_feat['atomic_num']: ", str(masked_atom_bond_graph.node_feat['atomic_num']))
        # print("    masked_atom_bond_graph.edge_feat['bond_dir']: ", str(masked_atom_bond_graph.edge_feat['bond_dir']))
        graph_dict['masked_atom_bond_graph'] = masked_atom_bond_graph
        """
        masked_atom_bond_graph.node_feat:  {'atomic_num': array([8, 6, 0, 0, 0, 0, 0, 0, 6, 6, 0, 6, 6, 9, 8, 6, 0, 0, 0, 0, 0, 0,
       8, 6, 6, 6, 8, 7, 6, 8, 7]), 'formal_charge': array([6, 6, 0, 0, 0, 0, 0, 0, 6, 6, 0, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 0,
       6, 6, 6, 6, 6, 5, 6, 6, 6]), 'degree': array([2, 4, 0, 0, 0, 0, 0, 0, 3, 3, 0, 3, 4, 2, 2, 4, 0, 0, 0, 0, 0, 0,
       3, 4, 3, 4, 2, 3, 4, 2, 3]), 'chiral_tag': array([1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
       1, 1, 1, 1, 1, 1, 1, 1, 1]), 'total_numHs': array([1, 1, 0, 0, 0, 0, 0, 0, 3, 3, 0, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
       1, 1, 2, 1, 1, 1, 1, 1, 2]), 'is_aromatic': array([1, 2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0,
       2, 2, 2, 2, 1, 2, 2, 1, 2]), 'hybridization': array([4, 4, 0, 0, 0, 0, 0, 0, 5, 5, 0, 4, 4, 5, 4, 4, 0, 0, 0, 0, 0, 0,
       4, 4, 4, 4, 4, 4, 4, 4, 4])}
       
       masked_atom_bond_graph.edge_feat:  {'bond_dir': array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 0, 0, 9, 9, 9, 0, 9, 9, 0, 9, 9, 9, 9, 9, 9, 9,
       1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 0, 9, 9, 0, 9,
       9, 9, 9, 9, 9, 9, 9, 9, 9]), ... }
        """

        """
        masked bond-angle graph graph_dict, feed_dict
        """
        masked_bond_angle_graph = pgl.Graph.batch(masked_bond_angle_graph_list)
        self._flat_shapes(masked_bond_angle_graph.node_feat)
        self._flat_shapes(masked_bond_angle_graph.edge_feat)
        # print('    masked_bond_angle_graph.node_feat: ', str(masked_bond_angle_graph.node_feat))
        # print("    masked_bond_angle_graph.edge_feat['bond_angle'][:5]: ", str(masked_bond_angle_graph.edge_feat['bond_angle'][:5]))
        graph_dict['masked_bond_angle_graph'] = masked_bond_angle_graph
        """
        masked_bond_angle_graph.node_feat:  {}
        
        masked_bond_angle_graph.edge_feat:
        {'bond_angle': array([3.1359193 , 0.        , 3.1359193 , 0.9761861 , 1.0863941 ,
            0.        , 0.9761861 , 3.13609   , 1.0790395 , 0.        , ...], dtype=float32)}
        """


        """
        masked dihedral-angle graph graph_dict, feed_dict
        """
        masked_dihedral_angle_graph = pgl.Graph.batch(masked_dihedral_angle_graph_list)
        self._flat_shapes(masked_dihedral_angle_graph.node_feat)
        self._flat_shapes(masked_dihedral_angle_graph.edge_feat)
        graph_dict['masked_dihedral_angle_graph'] = masked_dihedral_angle_graph



        # print('  construct feed dict')
        if 'Cm' in self.pretrain_tasks:
            feed_dict['Cm_node_i'] = np.concatenate(Cm_node_i, 0).reshape(-1).astype('int64')
            # print("    feed_dict['Cm_node_i']: ", list(feed_dict['Cm_node_i']))  # [ 3  6 17 20]
            feed_dict['Cm_context_id'] = np.concatenate(Cm_context_id, 0).reshape(-1, 1).astype('int64')
            # print("    feed_dict['Cm_context_id']: ", list(feed_dict['Cm_context_id']))  # [[ 419] [1219] [1942] [ 837]]
        if 'Fg' in self.pretrain_tasks:
            feed_dict['Fg_morgan'] = np.array(Fg_morgan, 'float32')
            feed_dict['Fg_daylight'] = (np.array(Fg_daylight) > 0).astype('float32')  # >1: 1x
            feed_dict['Fg_maccs'] = np.array(Fg_maccs, 'float32')
        if 'Bar' in self.pretrain_tasks:
            feed_dict['Ba_node_i'] = np.concatenate(Ba_node_i, 0).reshape(-1).astype('int64')
            feed_dict['Ba_node_j'] = np.concatenate(Ba_node_j, 0).reshape(-1).astype('int64')
            feed_dict['Ba_node_k'] = np.concatenate(Ba_node_k, 0).reshape(-1).astype('int64')
            feed_dict['Ba_bond_angle'] = np.concatenate(Ba_bond_angle, 0).reshape(-1, 1).astype('float32')
        if 'Blr' in self.pretrain_tasks:
            feed_dict['Bl_node_i'] = np.concatenate(Bl_node_i, 0).reshape(-1).astype('int64')
            feed_dict['Bl_node_j'] = np.concatenate(Bl_node_j, 0).reshape(-1).astype('int64')
            feed_dict['Bl_bond_length'] = np.concatenate(Bl_bond_length, 0).reshape(-1, 1).astype('float32')
        if 'Adc' in self.pretrain_tasks:
            feed_dict['Ad_node_i'] = np.concatenate(Ad_node_i, 0).reshape(-1).astype('int64')
            feed_dict['Ad_node_j'] = np.concatenate(Ad_node_j, 0).reshape(-1).astype('int64')
            feed_dict['Ad_atom_dist'] = np.concatenate(Ad_atom_dist, 0).reshape(-1, 1).astype('float32')
        if 'Dar' in self.pretrain_tasks:
            feed_dict['Da_node_i'] = np.concatenate(Da_node_i, 0).reshape(-1).astype('int64')
            feed_dict['Da_node_m'] = np.concatenate(Da_node_m, 0).reshape(-1).astype('int64')
            feed_dict['Da_node_n'] = np.concatenate(Da_node_n, 0).reshape(-1).astype('int64')
            feed_dict['Da_node_j'] = np.concatenate(Da_node_j, 0).reshape(-1).astype('int64')
            feed_dict['Da_dihedral_angle'] = np.concatenate(Da_dihedral_angle, 0).reshape(-1, 1).astype('float32')

        return graph_dict, feed_dict

