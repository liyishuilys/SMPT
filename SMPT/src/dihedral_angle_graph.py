
import time
import argparse
from src.util import *
from collections import Counter
from multiprocessing import Pool


def pickle_load_data(path):
    files_list = []
    for root, dirs, files in os.walk(args.pickle_path):
        for name in files:
            if name.endswith(".pkl"):
                files_list.append([root, name])  
    return files_list


def _get_plane_atoms(data, plane):
    edges = data['edges']
    e0, e1 = plane  # edge0, edge1
    a0, a1 = edges[e0]  # edge0, atom0, atom1
    b0, b1 = edges[e1]  # edge1, atom0, atom1
    return [a0, a1, b0, b1]


def get_planes(data):

    def _add_atoms_item(plane_atoms, atoms):
        plane_atoms.append(atoms)

    def _add_ring_item(plane_in_ring, is_in_ring, edge0, edge1):
        is_in_ring = list(is_in_ring)
        if is_in_ring[edge0] == is_in_ring[edge1]:
            plane_in_ring.append(is_in_ring[edge0])
        else:
            plane_in_ring.append(0)

    def _add_mass_item(plane_mass, mass, node_0, node_1, node_2, node_3):
        mass_plus = mass[node_0] + mass[node_1] + mass[node_2] + mass[node_3]
        plane_mass.append(mass_plus)

    is_in_ring = data['is_in_ring']
    mass = data['mass']
    BondAngleGraph_edges = data['BondAngleGraph_edges']  # one edge of BAG can be as a plane

    plane_atoms = []
    plane_in_ring = []
    plane_mass = []
    for pid, plane in enumerate(BondAngleGraph_edges):
        e0, e1 = plane  # edge0, edge1
        a0, a1, b0, b1 = _get_plane_atoms(data, plane)  # edge0, atom0, atom1; edge1, atom0, atom1

        # add plane atoms
        _add_atoms_item(plane_atoms, [a0, a1, b0, b1])
        # add plane is in ring
        _add_ring_item(plane_in_ring, is_in_ring, e0, e1)
        # add plane mass
        _add_mass_item(plane_mass, mass, a0, a1, b0, b1)

    if len(plane_atoms) == 0 or len(plane_in_ring) == 0 or len(plane_mass) == 0:
        plane_atoms = np.zeros([0, 4], 'int64')
        plane_in_ring = np.zeros([0, ], 'int64')
        plane_mass = np.zeros([0, ], 'float32')
    else:
        plane_atoms = np.array(plane_atoms, 'int64')
        plane_in_ring = np.array(plane_in_ring, 'int64')
        plane_mass = np.array(plane_mass, 'float32')

    return plane_atoms, plane_in_ring, plane_mass



def _get_norm_vector(plane_atoms, atom_poses):
    a0, a1, b0, b1 = plane_atoms  # presuppose that only three different atoms
    atoms = []
    if a0 == b0:
        atoms = [a1, a0, b1]
    if a0 == b1:
        atoms = [a1, a0, b0]
    if a1 == b0:
        atoms = [a0, a1, b1]
    if a1 == b1:
        atoms = [a0, a1, b0]

    atom0_posx, atom0_posy, atom0_posz = atom_poses[atoms[0]]
    atom1_posx, atom1_posy, atom1_posz = atom_poses[atoms[1]]
    atom2_posx, atom2_posy, atom2_posz = atom_poses[atoms[2]]

    norm_vector_x = (atom0_posy - atom1_posy) * (atom2_posz - atom1_posz) \
                    - (atom2_posy - atom1_posy) * (atom0_posz - atom1_posz)
    norm_vector_y = (atom0_posz - atom1_posz) * (atom0_posx - atom1_posx) \
                    - (atom2_posz - atom1_posz) * (atom0_posx - atom1_posx)
    norm_vector_z = (atom0_posx - atom1_posx) * (atom2_posy - atom1_posy) \
                    - (atom2_posx - atom1_posx) * (atom0_posy - atom1_posy)
    norm_vector = [norm_vector_x, norm_vector_y, norm_vector_z]
    return norm_vector



def _get_dihedral_angle(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    vec1 = vec1 / (norm1 + 1e-5)    # 1e-5: prevent numerical errors
    vec2 = vec2 / (norm2 + 1e-5)
    angle = np.arccos(np.dot(vec1, vec2))
    return angle


def get_dihedral_angle_graph(data):

    atom_poses = data['atom_pos']
    plane_atoms = data['plane_atoms']

    dihedral_angle_graph_edges = []
    dihedral_angle_graph_angles = []

    """
                  i
                 /\
                /  \
              m/____\n
               \    /
                \  /
                 \/
                  j
        """
    da_node_i_indices = []
    da_node_m_indices = []
    da_node_n_indices = []
    da_node_j_indices = []
    da_dihedral_angles = []

    # no planes at all
    if len(plane_atoms) == 0:
        da_node_i_indices = np.zeros([0, ], 'int64')
        da_node_m_indices = np.zeros([0, ], 'int64')
        da_node_n_indices = np.zeros([0, ], 'int64')
        da_node_j_indices = np.zeros([0, ], 'int64')
        da_dihedral_angles = np.zeros([0, ], 'float32')
        dihedral_angle_graph_edges = np.zeros([0, 2], 'int64')
        dihedral_angle_graph_angles = np.zeros([0, ], 'float32')
    else:
        BondAngleGraph_edges = data['BondAngleGraph_edges']
        bond_angle = data['bond_angle']
        dihedral_angle_graph_nodes = BondAngleGraph_edges
        for p0 in range(len(dihedral_angle_graph_nodes) - 1):
            for p1 in range(p0 + 1, len(dihedral_angle_graph_nodes)):

                # get plane 0&1
                plane0 = dihedral_angle_graph_nodes[p0]
                plane1 = dihedral_angle_graph_nodes[p1]
                plane0_atoms = plane_atoms[p0]
                plane1_atoms = plane_atoms[p1]

                # plane0 = plane1
                # if dict(Counter(plane0)) == dict(Counter(plane1)):
                if (set(plane0) == set(plane1)):
                    continue

                # plane 0&1, bond angle = 0, only three different atoms
                if bond_angle[p0] == 0 or len(set(plane0_atoms)) != 3:
                    continue
                if bond_angle[p1] == 0 or len(set(plane1_atoms)) != 3:
                    continue

                # plane0 & plane1 if not share one edge or two nodes, continue
                shared_edge = set(plane0).intersection(plane1)  # union set
                shared_atoms = set(plane0_atoms).intersection(plane1_atoms)  # union set
                if len(shared_edge) != 1 or len(shared_atoms) != 2:
                    continue

                # plane 0&1 normal vector
                plane0_norm_vector = _get_norm_vector(plane0_atoms, atom_poses)
                plane1_norm_vector = _get_norm_vector(plane1_atoms, atom_poses)
                # calculate dihedral angle between plane 0&1
                dihedral_angle = _get_dihedral_angle(plane0_norm_vector, plane1_norm_vector)

                # DihedralAngleGraph
                dihedral_angle_graph_edges.append([p0, p1])
                dihedral_angle_graph_angles.append(dihedral_angle)

                # Da_node_i/m/n/j, Da_dihedral_angle
                da_node_i = list(set(plane0_atoms).difference(shared_atoms))[0]  # differece set
                da_node_m = list(shared_atoms)[0]
                da_node_n = list(shared_atoms)[1]
                da_node_j = list(set(plane1_atoms).difference(shared_atoms))[0]  # differece set

                da_node_i_indices += [da_node_i, da_node_i, da_node_j, da_node_j]
                da_node_m_indices += [da_node_m, da_node_n, da_node_m, da_node_n]
                da_node_n_indices += [da_node_n, da_node_m, da_node_n, da_node_m]
                da_node_j_indices += [da_node_j, da_node_j, da_node_i, da_node_i]
                da_dihedral_angles += [dihedral_angle, dihedral_angle, dihedral_angle, dihedral_angle]

        da_node_imnj = np.array([da_node_i_indices, da_node_m_indices, da_node_n_indices, da_node_j_indices])
        uniq_da_node_imnj, uniq_index = np.unique(da_node_imnj, return_index=True, axis=1)  # order da_node_i_indices
        da_node_i_indices, da_node_m_indices, da_node_n_indices, da_node_j_indices = uniq_da_node_imnj
        da_dihedral_angles = np.array(da_dihedral_angles)[uniq_index]

        if len(dihedral_angle_graph_edges) == 0 or len(dihedral_angle_graph_angles) == 0:
            da_node_i_indices = np.zeros([0, ], 'int64')
            da_node_m_indices = np.zeros([0, ], 'int64')
            da_node_n_indices = np.zeros([0, ], 'int64')
            da_node_j_indices = np.zeros([0, ], 'int64')
            da_dihedral_angles = np.zeros([0, ], 'float32')
            dihedral_angle_graph_edges = np.zeros([0, 2], 'int64')
            dihedral_angle_graph_angles = np.zeros([0, ], 'float32')
        else:
            da_node_i_indices = np.array(da_node_i_indices, 'int64')
            da_node_m_indices = np.array(da_node_m_indices, 'int64')
            da_node_n_indices = np.array(da_node_n_indices, 'int64')
            da_node_j_indices = np.array(da_node_j_indices, 'int64')
            da_dihedral_angles = np.array(da_dihedral_angles, 'float32')
            dihedral_angle_graph_edges = np.array(dihedral_angle_graph_edges, 'int64')
            dihedral_angle_graph_angles = np.array(dihedral_angle_graph_angles, 'float32')

    return dihedral_angle_graph_edges, dihedral_angle_graph_angles, \
           da_node_i_indices, da_node_m_indices, da_node_n_indices, da_node_j_indices, da_dihedral_angles




def gen_new_data_with_dihedral_angle(arg_v):
    """
    DAG: Dihedral Angle Graph

    node: BondAngleGraph_edges
    node_value: plane_in_ring, plane_mass

    edge: DihedralAngleGraph_edges
    edge_value: DihedralAngleGraph_angles
    """
    idx, data = arg_v
    if idx % 400 == 0:
        print(" ", idx, end=", ")
    plane_atoms, plane_in_ring, plane_mass = get_planes(data)
    if 'planes' in data:  # clear old invalid data
        del data['planes']
    data['plane_atoms'] = plane_atoms
    data['plane_in_ring'] = plane_in_ring
    data['plane_mass'] = plane_mass
    dihedral_angle_graph_edges, dihedral_angle_graph_angles, \
    da_node_i, da_node_m, da_node_n, da_node_j, da_dihedral_angles \
        = get_dihedral_angle_graph(data)
    data['DihedralAngleGraph_edges'] = dihedral_angle_graph_edges
    data['DihedralAngleGraph_angles'] = dihedral_angle_graph_angles
    data['Da_node_i'] = da_node_i
    data['Da_node_m'] = da_node_m
    data['Da_node_n'] = da_node_n
    data['Da_node_j'] = da_node_j
    data['Da_dihedral_angle'] = da_dihedral_angles

    if len(data['BondAngleGraph_edges']) == 0 or len(data['bond_angle']) == 0:
        data['BondAngleGraph_edges'] = [[0, 0]]
        data['bond_angle'] = [0.]
        data['bond_angle'] = np.array(data['bond_angle'], 'float32')
    if len(data['plane_atoms']) == 0 or len(data['plane_atoms']) == 0 or len(data['plane_atoms']) == 0:
        data['plane_atoms'] = [[0, 0, 0, 0]]
        data['plane_in_ring'] = [0]
        data['plane_mass'] = [0.]
        data['plane_mass'] = np.array(data['plane_mass'], 'float32')
    if len(data['DihedralAngleGraph_edges']) == 0 or len(data['DihedralAngleGraph_angles']) == 0:
        data['DihedralAngleGraph_edges'] = [[0, 0]]
        data['DihedralAngleGraph_angles'] = [0.]
        data['DihedralAngleGraph_angles'] = np.array(data['DihedralAngleGraph_angles'], 'float32')

    return data



def gen_new_data_with_dihedral_angle_simple(data):
    """
    DAG: Dihedral Angle Graph

    node: BondAngleGraph_edges
    node_value: plane_in_ring, plane_mass

    edge: DihedralAngleGraph_edges
    edge_value: DihedralAngleGraph_angles
    """
    plane_atoms, plane_in_ring, plane_mass = get_planes(data)
    if 'planes' in data:  # clear old invalid data
        del data['planes']
    data['plane_atoms'] = plane_atoms
    data['plane_in_ring'] = plane_in_ring
    data['plane_mass'] = plane_mass
    dihedral_angle_graph_edges, dihedral_angle_graph_angles, \
    da_node_i, da_node_m, da_node_n, da_node_j, da_dihedral_angles \
        = get_dihedral_angle_graph(data)
    data['DihedralAngleGraph_edges'] = dihedral_angle_graph_edges
    data['DihedralAngleGraph_angles'] = dihedral_angle_graph_angles

    if len(data['BondAngleGraph_edges']) == 0 or len(data['bond_angle']) == 0:
        data['BondAngleGraph_edges'] = [[0, 0]]
        data['bond_angle'] = [0.]
        data['bond_angle'] = np.array(data['bond_angle'], 'float32')
    if len(data['plane_atoms']) == 0 or len(data['plane_atoms']) == 0 or len(data['plane_atoms']) == 0:
        data['plane_atoms'] = [[0, 0, 0, 0]]
        data['plane_in_ring'] = [0]
        data['plane_mass'] = [0.]
        data['plane_mass'] = np.array(data['plane_mass'], 'float32')
    if len(data['DihedralAngleGraph_edges']) == 0 or len(data['DihedralAngleGraph_angles']) == 0:
        data['DihedralAngleGraph_edges'] = [[0, 0]]
        data['DihedralAngleGraph_angles'] = [0.]
        data['DihedralAngleGraph_angles'] = np.array(data['DihedralAngleGraph_angles'], 'float32')

    return data




def trasform_data_dtype(idx, data):
    if idx % 400 == 0:
        print(" ", idx, end=", ")
    if len(data['BondAngleGraph_edges']) == 0 or len(data['bond_angle']) == 0:
        data['BondAngleGraph_edges'] = [[0, 0]]
        data['bond_angle'] = [0.]
        data['bond_angle'] = np.array(data['bond_angle'], 'float32')
    if len(data['plane_atoms']) == 0 or len(data['plane_atoms']) == 0 or len(data['plane_atoms']) == 0:
        data['plane_atoms'] = [[0, 0, 0, 0]]
        data['plane_in_ring'] = [0]
        data['plane_mass'] = [0.]
        data['plane_mass'] = np.array(data['plane_mass'], 'float32')
    if len(data['DihedralAngleGraph_edges']) == 0 or len(data['DihedralAngleGraph_angles']) == 0:
        data['DihedralAngleGraph_edges'] = [[0, 0]]
        data['DihedralAngleGraph_angles'] = [0.]
        data['DihedralAngleGraph_angles'] = np.array(data['DihedralAngleGraph_angles'], 'float32')


    data['plane_atoms'] = np.array(data['plane_atoms'], 'int64')
    data['plane_in_ring'] = np.array(data['plane_in_ring'], 'int64')
    data['plane_mass'] = np.array(data['plane_mass'], 'float32')

    data['Da_node_i'] = np.array(data['Da_node_i'], 'int64')
    data['Da_node_m'] = np.array(data['Da_node_m'], 'int64')
    data['Da_node_n'] = np.array(data['Da_node_n'], 'int64')
    data['Da_node_j'] = np.array(data['Da_node_j'], 'int64')
    data['Da_dihedral_angle'] = np.array(data['Da_dihedral_angle'], 'float32')
    data['DihedralAngleGraph_edges'] = np.array(data['DihedralAngleGraph_edges'], 'int64')
    data['DihedralAngleGraph_angles'] = np.array(data['DihedralAngleGraph_angles'], 'float32')

    return data






def main(args):
    st = time.time()
    files_list = pickle_load_data(args.pickle_path)
    for fid, file in enumerate(files_list):
        root, name = file  # 
        print("\n\nprocessing %s: %s/%s" % (fid, root, name))

        # repickle data
        new_root = args.pickle_path_new + "/" + "/".join(root.split("/")[-2:])
        mkdir(new_root)
        pd = open(new_root + "/" + name, "wb")

        # load old dataset and generate new dataset
        dataset_new = []
        pkl = open(root + "/" + name, "rb")
        dataset_old = pickle.load(pkl)
        print("  old dataset len: %s" % len(dataset_old))



        # single process transform data dtype
        for idx, data in enumerate(dataset_old):
            data = trasform_data_dtype(idx, data)
            dataset_new.append(data)



        # # single process test
        # for idx, data in enumerate(dataset_old):
        #     data = gen_new_dataset_with_dihedral_angle((idx, data))
        #     dataset_new.append(data)
        #     print(data)
        #     exit(0)
        # exit(0)

        # # multiprocessing data
        # dataset_old = [(idx, data) for idx, data in enumerate(dataset_old)]
        # pool = Pool(args.num_workers)
        # map_results = pool.map_async(gen_new_data_with_dihedral_angle, dataset_old)
        # pool.close()
        # pool.join()
        # for result in map_results.get():
        #     dataset_new.append(result)

        dataset_new = InMemoryDataset(data_list=dataset_new)
        print("\n  new dataset len: %s" % len(dataset_new))
        pickle.dump(dataset_new, pd)
        # print(dataset_new[0])
    print("\n\nFinish!")
    print("Time: %s" % (time.time() - st))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_path", type=str, default='pickle_dataset')
    """
    0_500000, 500000_100000, 1000000_2000000, 2000000_3000000, 3000000_4000000, 
    4000000_5000000, 5000000_6000000, 6000000_7000000, 7000000_8000000, 8000000_9000000, 
    9000000_10000000, 10000000_11000000, 11000000_12000000, 12000000_12525012
    """
    parser.add_argument("--pickle_path_new", type=str, default='pickle_dataset_new')
    parser.add_argument("--num_workers", type=int, default=16)

    args = parser.parse_args()
    main(args)
