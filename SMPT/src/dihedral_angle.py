
import time
import argparse
from util import *
from collections import Counter
from multiprocessing import Pool


def pickle_load_data(path):
    files_list = []
    for root, dirs, files in os.walk(args.pickle_path):
        for name in files:
            if name.endswith(".pkl"):
                files_list.append([root, name]) 
    return files_list



def _get_norm_vector(plane, atom_poses):
    atom0, atom1, atom2 = plane
    atom0_posx, atom0_posy, atom0_posz = atom_poses[atom0]
    atom1_posx, atom1_posy, atom1_posz = atom_poses[atom1]
    atom2_posx, atom2_posy, atom2_posz = atom_poses[atom2]

    norm_vector_x = (atom0_posy - atom1_posy) * (atom2_posz - atom1_posz) \
                    - (atom2_posy - atom1_posy) * (atom0_posz - atom1_posz)
    norm_vector_y = (atom0_posz - atom1_posz) * (atom0_posx - atom1_posx) \
                    - (atom2_posz - atom1_posz) * (atom0_posx - atom1_posx)
    norm_vector_z = (atom0_posx - atom1_posx) * (atom2_posy - atom1_posy) \
                    - (atom2_posx - atom1_posx) * (atom0_posy - atom1_posy)
    norm_vector = [norm_vector_x, norm_vector_y, norm_vector_z]
    return norm_vector



def _get_angle(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    vec1 = vec1 / (norm1 + 1e-5)    # 1e-5: prevent numerical errors
    vec2 = vec2 / (norm2 + 1e-5)
    angle = np.arccos(np.dot(vec1, vec2))
    return angle



def get_dihedral_angle(data):
    Ba_node_i = data['Ba_node_i']
    Ba_node_j = data['Ba_node_j']
    Ba_node_k = data['Ba_node_k']
    atom_poses = data['atom_pos']
    Ba_bond_angle = data['Ba_bond_angle']

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
    node_i_indices = []
    node_m_indices = []
    node_n_indices = []
    node_j_indices = []
    dihedral_angles = []

    for p1 in range(len(Ba_node_i) - 1):
        for p2 in range(p1, len(Ba_node_i)):

            # bong angle = 0
            if Ba_bond_angle[p1] == 0 or Ba_bond_angle[p2] == 0:
                continue

            # plane 1&2
            plane1 = [Ba_node_i[p1], Ba_node_j[p1], Ba_node_k[p1]]
            plane2 = [Ba_node_i[p2], Ba_node_j[p2], Ba_node_k[p2]]

            # plane1 = plane2
            if dict(Counter(plane1)) == dict(Counter(plane2)):
                continue

            # plane1 & plane2 if not share one edge, continue
            shared_nodes = set(plane1).intersection(plane2)  # union set
            if len(shared_nodes) != 2:
                continue

            # plane 1&2 normal vector
            plane1_norm_vector = _get_norm_vector(plane1, atom_poses)
            plane2_norm_vector = _get_norm_vector(plane2, atom_poses)
            # calculate dihedral angle between plane 1&2
            dihedral_angle = _get_angle(plane1_norm_vector, plane2_norm_vector)

            node_i = list(set(plane1).difference(shared_nodes))[0]  # differece set
            node_m = list(shared_nodes)[0]
            node_n = list(shared_nodes)[1]
            node_j = list(set(plane2).difference(shared_nodes))[0]  # differece set

            node_i_indices += [node_i, node_i, node_j, node_j]
            node_m_indices += [node_m, node_n, node_m, node_n]
            node_n_indices += [node_n, node_m, node_n, node_m]
            node_j_indices += [node_j, node_j, node_i, node_i]
            dihedral_angles += [dihedral_angle, dihedral_angle, dihedral_angle, dihedral_angle]

    node_imnj = np.array([node_i_indices, node_m_indices, node_n_indices, node_j_indices])
    uniq_node_imnj, uniq_index = np.unique(node_imnj, return_index=True, axis=1)  # order node_i_indices
    node_i_indices, node_m_indices, node_n_indices, node_j_indices = uniq_node_imnj
    dihedral_angles = np.array(dihedral_angles)[uniq_index]

    return [node_i_indices, node_m_indices, node_n_indices, node_j_indices, dihedral_angles]



def gen_new_dataset_with_dihedral_angle(args):
    idx, data = args
    if idx % 400 == 0:
        print(" ", idx, end=", ")
    node_i, node_m, node_n, node_j, dihedral_angles = get_dihedral_angle(data)
    data['Da_node_i'] = node_i
    data['Da_node_m'] = node_m
    data['Da_node_n'] = node_n
    data['Da_node_j'] = node_j
    data['Da_dihedral_angle'] = dihedral_angles
    return data



def main(args):
    st = time.time()
    files_list = pickle_load_data(args.pickle_path)
    for fid, file in enumerate(files_list):
        root, name = file  
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
        # multiprocessing data
        dataset_old = [(idx, data) for idx, data in enumerate(dataset_old)]
        pool = Pool(args.num_workers)
        map_results = pool.map_async(gen_new_dataset_with_dihedral_angle, dataset_old)
        pool.close()
        pool.join()
        for result in map_results.get():
            dataset_new.append(result)

        dataset_new = InMemoryDataset(data_list=dataset_new)
        print("\n  new dataset len: %s" % len(dataset_new))
        pickle.dump(dataset_new, pd)
    print("Finish!")
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



def backup():
    st = time.time()
    files_list = pickle_load_data(args.pickle_path)
    for file in files_list:
        root, name = file  
        print("processing: %s/%s" % (root, name))

        # repickle data
        new_root = args.pickle_path_new + "/" + "/".join(root.split("/")[-2:])
        mkdir(new_root)
        pd = open(new_root + "/" + name, "wb")

        # load data
        pkl = open(root + "/" + name, "rb")
        dataset = pickle.load(pkl)
        print("  dataset len: %s" % len(dataset))
        for idx, data in enumerate(dataset):
            if idx % 400 == 0:
                print("  ", idx, end=", ")
            node_i, node_m, node_n, node_j, dihedral_angles = get_dihedral_angle(data)
            data['Da_node_i'] = node_i
            data['Da_node_m'] = node_m
            data['Da_node_n'] = node_n
            data['Da_node_j'] = node_j
            data['Da_dihedral_angle'] = dihedral_angles

        pickle.dump(dataset, pd)
    print("Finish!")
    print("Time: %s" % (time.time() - st))