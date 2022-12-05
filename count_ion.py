#!/usr/bin/env python3
import argparse
import mdtraj as md
import numpy as np



def potassium_state_assign_membrane(traj, top_ind, bottom_ind, ion_index):
    """
    :param traj: mdtraj.traj
    :param top_ind:    atom index list
    :param bottom_ind: atom index list
    :return: ions_state_dict
    #
    #      3
    #P#P#P#P#P#P#P#P#P#P#P#P#P#P#P# top_ind
    #
    #        2
    #
    #P#P#P#P#P#P#P#P#P#P#P#P#P#P#P# bottom_ind
    #     1
    #
    """
    # initiate ions_state_dict
    ions_state_dict = {}
    for k in ion_index:
        ions_state_dict[k] = np.zeros(traj.n_frames, dtype=int) + 2
    # check if ion is in 3
    boundary = md.compute_center_of_mass(traj.atom_slice(top_ind))[:, 2]
    for k in ion_index:
        Z_Kion = traj.xyz[:, k, 2]
        ions_state_dict[k][Z_Kion > boundary] = 3
    # check if ion is in 1
    boundary = md.compute_center_of_mass(traj.atom_slice(bottom_ind))[:, 2]
    for k in ion_index:
        Z_Kion = traj.xyz[:, k, 2]
        ions_state_dict[k][Z_Kion < boundary] = 1
    return ions_state_dict


def potassium_state_assign_cylider(top_ind, bottom_ind, center_ind,
                                   traj, ion_index, rad=0.25, ):
    """
    :param top_ind:    atom index list
    :param bottom_ind: atom index list
    :param center_ind: atom index list
    :param ion_index:  atom index list
    :param traj: mdtraj.traj, mind the memory usage
    :param rad: nm
    :return: A dictionary with ion_state in each frame
    #
    #  3
    ############################### top_ind
    #          |       |
    #     2    |   1   |
    #          |       |
    #          |    <->|
    #          |    rad|  center_ind for the xy position
    ############################### bottom_ind
    #     4
    #
    """
    # initiate ions_state_dict
    ions_state_dict = {}
    for k in ion_index:
        ions_state_dict[k] = np.zeros(traj.n_frames, dtype=int) + 2
    # seperate the Cylinder 1 and 2
    center = md.compute_center_of_mass(traj.atom_slice(center_ind))[:, :2]
    for k in ion_index:
        Kion_traj = traj.xyz[:, k, :2]
        xy = Kion_traj - center
        mask1 = xy[:, 0] ** 2 + xy[:, 1] ** 2 < rad ** 2
        ions_state_dict[k][mask1] = 1
    # check if ion is in 3
    boundary = md.compute_center_of_mass(traj.atom_slice(top_ind))[:, 2]
    for k in ion_index:
        Z_Kion = traj.xyz[:, k, 2]
        ions_state_dict[k][Z_Kion > boundary] = 3
    # check if ion is in 4
    boundary = md.compute_center_of_mass(traj.atom_slice(bottom_ind))[:, 2]
    for k in ion_index:
        Z_Kion = traj.xyz[:, k, 2]
        ions_state_dict[k][Z_Kion < boundary] = 4
    return ions_state_dict


def assign_ion_state_chunk(xtc_file, stride, top, chunk, assign_fun=potassium_state_assign_cylider, **kwargs):
    """
    :param xtc_file:
    :param top:
    :param chunk:
    :param assign_fun: must take traj as keyword argument
    :param kwargs: all the keyword argument for 'assigh_fun'
    :return:
    """
    count = 0
    for traj_chunk in md.iterload(xtc_file, top=top, chunk=chunk, stride=stride):
        print(".", end='')
        ions_state_dict_tmp = assign_fun(**kwargs, traj=traj_chunk)
        if count == 0:
            ions_state_dict = ions_state_dict_tmp
            count += 1
        else:
            for k in ions_state_dict:
                ions_state_dict[k] = np.concatenate((ions_state_dict[k], ions_state_dict_tmp[k]))
            count += 1
    print(".")
    return ions_state_dict


def ion_state_short(ion_state_chain, traj_timestep):
    state_0 = ion_state_chain[0]  # previous state
    state_chain = []
    resident_time_chain = []
    end_time_chain = []
    resident_time_tmp = traj_timestep
    for i in range(1, len(ion_state_chain)):
        state_1 = ion_state_chain[i]  # new state
        if state_1 == state_0:
            resident_time_tmp += traj_timestep
        else:
            state_chain.append(state_0)
            resident_time_chain.append(resident_time_tmp)
            end_time_chain.append((i - 1) * traj_timestep)
            state_0 = state_1
            resident_time_tmp = traj_timestep
    state_chain.append(state_0)
    resident_time_chain.append(resident_time_tmp)
    end_time_chain.append(i * traj_timestep)
    return np.array(state_chain, dtype=int), np.array(resident_time_chain), np.array(end_time_chain)


def ion_state_short_map(ions_state_dict, traj_timestep):
    ions_state = {}
    ions_resident_time = {}
    end_time = {}
    for k in ions_state_dict:
        ion_state_chain = ions_state_dict[k]
        ions_state[k], ions_resident_time[k], end_time[k] = ion_state_short(ion_state_chain, traj_timestep)
    return ions_state, ions_resident_time, end_time


def match_sequence_numpy(arr, seq):
    """
    :param arr: 1 D np.array
    :param seq: 1 D np.array
    :return: np.where(match)
    """
    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size
    # Range of sequence
    r_seq = np.arange(Nseq)
    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[np.arange(Na - Nseq + 1)[:, None] + r_seq] == seq).all(1)
    # Get the range of those indices as final output
    return np.where(M)


def match_seqs(ions_state, seq):
    """
    :param ions_state:
    :param seq: 1 D np.array
    :return:
    """
    matched_dict = {}
    for k in ions_state:
        i_state = ions_state[k]
        matched_dict[k] = match_sequence_numpy(i_state, seq)
    return matched_dict


def match_head_tail(ions_state, seq):
    matched_dict_head = {}
    matched_dict_tail = {}
    for k in ions_state:
        i_state = ions_state[k]
        matched_dict_head[k] = (i_state[:seq.size] == seq).all()
        matched_dict_tail[k] = (i_state[-seq.size:] == seq).all()
    return matched_dict_head, matched_dict_tail


def auto_find_SF_index(traj):
    pass


def find_P_index(traj):
    """
    :param traj:
    :return: up_leaf_index, low_leaf_index # 0 base index for mdtraj
    """
    P_index = traj.topology.select("name P")
    P_z = traj.atom_slice(P_index).xyz[0, :, 2]
    middle = np.mean(P_z)
    up_leaf_index = []
    low_leaf_index = []
    for p in P_index:
        if traj.xyz[0, p, 2] > middle:
            up_leaf_index.append(p)
        else:
            low_leaf_index.append(p)
    return up_leaf_index, low_leaf_index


def print_seq(seq_list, ions_state, ions_resident_time, end_time, traj_timestep, voltage):
    for seq, name in seq_list:
        print("\n###############################")
        print("# New sequence start:", seq, name)
        print("#################################")
        count = 0
        length = len(seq)
        matched_dict = match_seqs(ions_state, seq)
        for k in matched_dict:
            # print("K ion index", k)
            #print()
            for i in matched_dict[k][0]:
                print("Perm:",
                      seq,
                      " %5d" % (k + 1),  # 1 base index
                      "resident_time %8d %8d %8d " % tuple(ions_resident_time[k][i:i + length]),
                      "end_t %8d %8d %8d " % tuple(end_time[k][i:i + length]),
                      "frame_num ", end_time[k][i:i + length] / traj_timestep
                      )
                count += 1
        current = count * 1.602176634 / end_time[k][-1] * 100000.0  # pA
        conductance = current * 1000 / voltage  # pS
        print("#################################")
        print("assumed voltage (mV) :", voltage)
        print("simulation time (ns)  :", end_time[k][-1] / 1000)
        print("ion permeation events : %d" % count)
        print("Ave current (pA)      : %.5f" % current)
        print("Ave conductance (pS)  : %.5f" % conductance)
        print("Sequence End")
        print("###############################")
        print("")





seq_list_dict = {"Cylinder": [(np.array([4, 1, 3]), "proper current up"),
                              (np.array([4, 2, 3]), "leak current up"),
                              (np.array([3, 1, 4]), "proper current down"),
                              (np.array([3, 2, 4]), "leak current down"),
                              (np.array([1, 2]), "SF broken, inside-out"),
                              (np.array([2, 1]), "SF broken, outside-in"),
                              ],
                 "Membrane": [(np.array([1, 2, 3]), "current up"),
                              (np.array([3, 2, 1]), "current down")]
                 }



if __name__ == "__main__":
    print("This is the main")
    parser = argparse.ArgumentParser()
    parser.add_argument("-pdb",
                        dest="top",
                        help="Ideally This file should be generated from trjconv as xtc",
                        metavar="top.pdb",
                        type=argparse.FileType('r'),
                        required=True)
    parser.add_argument("-xtc",
                        dest="traj",
                        metavar="traj.xtc",
                        help="this traj should have SF centered",
                        type=argparse.FileType('r'),
                        required=True)
    parser.add_argument("-K",
                        dest="K_name",
                        metavar="AT_name",
                        help="Atom name of Potassium in pdb file. Default K ei. K, POT, Na, SOD",
                        type=str,
                        default="K")
    parser.add_argument("-chunk",
                        dest="chunk",
                        metavar="int",
                        help="number of frame that will be load each time",
                        type=int,
                        default=100)
    parser.add_argument("-volt",
                        dest="volt",
                        metavar="float",
                        help="Voltage in mV",
                        type=float,
                        default=300.)
    parser.add_argument("-cylinderRad",
                        dest="cylRAD",
                        metavar="float",
                        help="Radius of the cylinder in nm. Default 0.25",
                        type=float,
                        default=0.25)
    parser.add_argument("-cylinderTop",
                        dest="cylTOP",
                        metavar="index",
                        help="1 base atom index for the top of the cylinder.such as \"111 247 383 519\"",
                        type=str)
    parser.add_argument("-cylinderBot",
                        dest="cylBOT",
                        metavar="index",
                        help="1 base atom index for the bottom of the cylinder",
                        type=str)
    parser.add_argument("-algorithm",
                        dest="alg",
                        choices=["Cylinder", "Membrane"],
                        type=str,
                        default="Cylinder",
                        help="""Cylinder:
    Check if the ion pass through the cylinder.
    You need to provide the top and bottom of the cylinder
Membrane:
    Check if the ion pass through the Membrane.
    The default boundary is P atom
    """,
                        )
    parser.add_argument("-stride",
                        dest="stride",
                        type=int,
                        default=None,
                        help="Only read every stride-th frame")


    args = parser.parse_args()
    # read arg
    top = args.top.name
    xtc_full_max = args.traj.name
    K_name = args.K_name
    chunk = args.chunk
    cylinderRad = args.cylRAD
    print("#################################################################################")
    print("PDB top file:", args.top.name)
    print("xtc traj file:", xtc_full_max)
    print("Ion name in this pdb should be:", K_name)
    print("The number of frame loading each time will be:", chunk)
    print("The Voltage in this simulation is: ", args.volt, "mV")
    print("#################################################################################")

    traj_pdb = md.load(top)
    # prepare time step
    for tmp in md.iterload(xtc_full_max, top=top, chunk=2):
        traj_timestep = tmp.timestep
        break

    # prepare atom index for cylinder
    if args.alg == "Cylinder":
        if args.cylTOP is None or args.cylBOT is None:
            raise ValueError("Pls give -cylinderTop and -cylinderBot")
        print("#################################################################################")
        top_ind = [int(i) - 1 for i in args.cylTOP.split()]  # mdtraj needs 0 base index
        bottom_ind = [int(i) - 1 for i in args.cylBOT.split()]  # mdtraj needs 0 base index
        center_ind = top_ind + bottom_ind
        ion_index = traj_pdb.topology.select('name ' + K_name)
        print("Number of ions found", len(ion_index))
        print("The ion index (0 base):", ion_index)
        print("Top of the cylinder")
        for at in top_ind:
            print(traj_pdb.topology.atom(at))
        print("Bottom of the cylinder")
        for at in bottom_ind:
            print(traj_pdb.topology.atom(at))

        # load xtc iteratively and assign state for each ion
        print("Assign state to each ion for each frame")
        print("Load xtc traj chunk by chunk.", end="")
        ions_state_dict = assign_ion_state_chunk(
            xtc_file=xtc_full_max,
            top=top,
            stride=args.stride,
            chunk=chunk,
            assign_fun=potassium_state_assign_cylider,
            top_ind=top_ind,
            bottom_ind=bottom_ind,
            center_ind=top_ind + bottom_ind,
            ion_index=ion_index,
            rad=cylinderRad
        )

    elif args.alg == "Membrane":
        print("#################################################################################")
        # Find P atom index and separate them to higher/lower leaflet
        up_leaf_index, low_leaf_index = find_P_index(traj_pdb)
        print("The upper leaf of membrane has (0 base index):", up_leaf_index)
        print("The lower leaf of membrane has (0 base index):", low_leaf_index)
        ion_index = traj_pdb.topology.select('name ' + K_name)
        print("Number of ions found", len(ion_index))
        print("The ion index (0 base):", ion_index)
        # load xtc iteratively and assign state for each ion
        print("Assign state to each ion for each frame")
        print("Load xtc traj chunk by chunk.", end="")
        ions_state_dict = assign_ion_state_chunk(
            xtc_file=xtc_full_max,
            top=top,
            stride=args.stride,
            chunk=chunk,
            assign_fun=potassium_state_assign_membrane,
            top_ind=up_leaf_index,
            bottom_ind=low_leaf_index,
            ion_index=ion_index,
        )

    # short the ion state array and generate with resident time
    ## [000111000] > [010]
    ions_state, ions_resident_time, end_time = ion_state_short_map(ions_state_dict, traj_timestep)

    # find specific state sequence and print/save
    seq_list = seq_list_dict[args.alg]
    print_seq(seq_list, ions_state, ions_resident_time, end_time, traj_timestep, args.volt)
