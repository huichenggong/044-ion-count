#!/usr/bin/env python3
import argparse
import mdtraj as md
import numpy as np
import os

lib = os.path.join(os.path.dirname(__file__), "cpp/cmake-build-debug/")
# print(lib)
import sys

sys.path.append(lib)
from PYSfilter import Sfilter


#
#  3
###############################
#          S01     |
#     2    |   1   |
#          |       |
#          |       |
#     -----S23-----|---
#          |   5   |
#          |       |
#          |    <->|
#          S45  rad|  center_ind for the xy position
###############################
#     4
#

def match_head(ions_state15, seq, forbidden):
    matched = {}
    for k in ions_state15:
        states = ions_state15[k]
        if states[0] == seq[0]:
            for i in range(len(states) - 2):
                if states[i + 1] == forbidden:
                    matched[k] = (np.array([], dtype=int),)
                    break
                elif np.all(states[i:i + 3] == seq):
                    matched[k] = (np.array([i]),)
                    break
        else:
            matched[k] = (np.array([], dtype=int),)
    return matched


def match_tail(ions_state15, seq, forbidden):
    matched = {}
    for k in ions_state15:
        states = ions_state15[k]
        if states[-1] == seq[-1]:
            for i in range(len(states) - 3, -1, -1):
                if states[i + 1] == forbidden:
                    matched[k] = (np.array([], dtype=int),)
                    break
                elif np.all(states[i:i + 3] == seq):
                    matched[k] = (np.array([i]),)
                    break
        else:
            matched[k] = (np.array([], dtype=int),)
    return matched


def auto_find_SF_index(traj, SF_seq=["THR", "VAL", "GLY", "TYR", "GLY"]):
    """
    :param traj: md.traj with proper atom name
    :param SF_seq: name of the sequence in SF, default: ["THR", "VAL", "GLY", "TYR", "GLY"]
    :return: 0 base index of S01, S12, S23, S34, S45
    """
    top = traj.topology
    S00 = []
    S01 = []
    S12 = []
    S23 = []
    S34 = []
    S45 = []

    for chain in top.chains:
        res_list = [res for res in chain.residues]
        for i in range(len(res_list) - 5):
            if res_list[i].is_water:
                break
            bool1 = res_list[i].name == SF_seq[0]
            bool2 = res_list[i + 1].name == SF_seq[1]
            bool3 = res_list[i + 2].name == SF_seq[2]
            bool4 = res_list[i + 3].name == SF_seq[3]
            bool5 = res_list[i + 4].name == SF_seq[4]
            if bool1 and bool2 and bool3 and bool4 and bool5:
                S45 += [atom.index for atom in res_list[i].atoms if atom.name == "OG1"]
                S34 += [atom.index for atom in res_list[i].atoms if atom.name == "O"]
                S23 += [atom.index for atom in res_list[i + 1].atoms if atom.name == "O"]
                S12 += [atom.index for atom in res_list[i + 2].atoms if atom.name == "O"]
                S01 += [atom.index for atom in res_list[i + 3].atoms if atom.name == "O"]
                S00 += [atom.index for atom in res_list[i + 4].atoms if atom.name == "O"]
    if len(S00) != 4 or len(S01) != 4 or len(S12) != 4 or len(S23) != 4 or len(S34) != 4 or len(S45) != 4:
        print(len(S00), len(S01), len(S12), len(S23), len(S34), len(S45))
        raise ValueError("SF auto detection Fail. The number of Oxygen selected for the SF boundary should be 4.")

    return S00, S01, S12, S23, S34, S45


def ion_state_list_2_dict(ion_state_list, ion_index):
    ion_state = np.array(ion_state_list).T
    ion_state_dict = {}
    ion_num, frame_num = ion_state.shape
    for i in range(ion_num):
        ion_state_dict[ion_index[i]] = ion_state[i, :]
    return ion_state_dict


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


def print_seq_str(seq, k, ions_resident_time, end_time, length, i, traj_timestep):
    print("Perm:",
          seq,
          " %5d" % (k + 1),  # 1 base index
          "resident_time %8d %8d %8d " % tuple(ions_resident_time[k][i:i + length]),
          "end_t %8d %8d %8d " % tuple(end_time[k][i:i + length]),
          "frame_num ", end_time[k][i:i + length] / traj_timestep
          )


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

def print_seq(seq_list, ions_state, ions_resident_time, end_time, traj_timestep, voltage, stride=None):
    for seq, name in seq_list:
        print("\n###############################")
        print("# New sequence start:", seq, name)
        print("#################################")
        if stride is None:
            stride = 1
        count = 0
        length = len(seq)
        matched_dict = match_seqs(ions_state, seq)
        for k in matched_dict:
            # print("K ion index", k)
            # print()
            for i in matched_dict[k][0]:
                print_seq_str(seq, k, ions_resident_time, end_time, length, i, traj_timestep)
                count += 1
        current = count * 1.602176634 / end_time[k][-1] * 100000.0  # pA
        conductance = current * 1000 / voltage  # pS
        print_conductance_summary(voltage, end_time, k, count, current, conductance)

def print_conductance_summary(voltage, end_time, k, count, current, conductance):
    print("#################################")
    print("assumed voltage (mV) :", voltage)
    print("simulation time (ns)  :", end_time[k][-1] / 1000)
    print("ion permeation events : %d" % count)
    print("Ave current (pA)      : %.5f" % current)
    print("Ave conductance (pS)  : %.5f" % conductance)
    print("Sequence End")
    print("###############################")
    print("")


if __name__ == "__main__":
    # print("This is the main")
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
    parser.add_argument("-SF_seq",
                        dest="SF_seq",
                        metavar="list of string",
                        help="THR VAL GLY TYR GLY",
                        type=str,
                        nargs=5)

    args = parser.parse_args()
    # read arg
    top = args.top.name
    xtc_full_max = args.traj.name
    K_name = args.K_name
    cylinderRad = args.cylRAD
    SF_seq = args.SF_seq
    print("#################################################################################")
    print("PDB top file:", args.top.name)
    print("xtc traj file:", xtc_full_max)
    print("Ion name in this pdb should be:", K_name)
    print("The Voltage in this simulation is: ", args.volt, "mV")
    print("The sequence of the SF is ", SF_seq)
    print("#################################################################################")

    traj_pdb = md.load(top)
    # prepare atom index for cylinder
    if (args.SF_seq is None):
        raise ValueError("Please provide -SF_seq")
    print("#################################################################################")
    S00, S01, S12, S23, S34, S45 = auto_find_SF_index(traj_pdb, SF_seq)
    ion_index = traj_pdb.topology.select('name ' + K_name)
    print("Number of ions found", len(ion_index))
    print("The ion index (0 base):", ion_index)
    wat_index = traj_pdb.topology.select("water and name O")
    print("Number of water(O) found")
    print("S01 of the cylinder")
    for at in S01:
        print(traj_pdb.topology.atom(at), at)
    print("S23 of the cylinder")
    for at in S23:
        print(traj_pdb.topology.atom(at), at)
    print("S45 of the cylinder")
    for at in S45:
        print(traj_pdb.topology.atom(at), at)

    # load xtc iteratively and assign state for each ion (c++)
    print("Assign state to each ion for each frame")
    SF = Sfilter(xtc_full_max)
    SF.assign_state_double(S01, S23, S45,
                           S01 + S23 + S45,
                           ion_index,
                           wat_index, cylinderRad
                           )
    print("Simulation time (ps)    :", SF.time)
    ions_state_dict = ion_state_list_2_dict(SF.ion_state_list, ion_index)
    wats_state_dict = ion_state_list_2_dict(SF.wat_state_list, wat_index)

    traj_timestep = SF.time / (len(ions_state_dict[ion_index[0]]) - 1)
    print("Traj time step (ps/frame):", traj_timestep)

    ions_state15, ions_resident_time15, end_time15 = ion_state_short_map(ions_state_dict, traj_timestep)
    for k in ions_state_dict:
        states = ions_state_dict[k]
        ions_state_dict[k][states == 5] = 1
    ions_state14, ions_resident_time14, end_time14 = ion_state_short_map(ions_state_dict, traj_timestep)

    ##########################
    # check permeation up
    ##########################
    print("\n###############################")
    print("# New sequence start:", np.array([4, 1, 3]), "proper current up")
    print("#################################")
    # check head/tail
    matched_h = match_head(ions_state15, seq=np.array([5, 1, 3]), forbidden=4)
    matched_t = match_tail(ions_state15, seq=np.array([4, 5, 1]), forbidden=3)
    count = 0
    for matched, seq in [(matched_h, np.array([5, 1, 3])), (matched_t, np.array([4, 5, 1]))]:
        for k in matched:
            # print("K ion index", k)
            # print()
            for i in matched[k][0]:
                print_seq_str(seq, k, ions_resident_time15, end_time15, len(seq), i, traj_timestep)
                count += 1
    # check permeation in the middle
    seq = np.array([4, 1, 3])
    matched_dict = match_seqs(ions_state14, seq)
    for k in matched_dict:
        for i in matched_dict[k][0]:
            print_seq_str(seq, k, ions_resident_time14, end_time14, len(seq), i, traj_timestep)
            count += 1
    current = count * 1.602176634 / end_time14[k][-1] * 100000.0  # pA
    conductance = current * 1000 / args.volt  # pS
    print_conductance_summary(args.volt, end_time14, k, count, current, conductance)

    ##########################
    # check permeation down
    ##########################
    print("\n###############################")
    print("# New sequence start:", np.array([3, 1, 4]), "proper current down")
    print("#################################")
    # check head/tail
    matched_h = match_head(ions_state15, seq=np.array([1, 5, 4]), forbidden=3)
    matched_t = match_tail(ions_state15, seq=np.array([3, 1, 5]), forbidden=4)

    count = 0
    for matched, seq in [(matched_h, np.array([1, 5, 4])), (matched_t, np.array([3, 1, 5]))]:
        for k in matched:
            # print("K ion index", k)
            # print()
            for i in matched[k][0]:
                print_seq_str(seq, k, ions_resident_time15, end_time15, len(seq), i, traj_timestep)
                count += 1
    # check permeation in the middle
    seq = np.array([3, 1, 4])
    matched_dict = match_seqs(ions_state14, seq)
    for k in matched_dict:
        for i in matched_dict[k][0]:
            print_seq_str(seq, k, ions_resident_time14, end_time14, len(seq), i, traj_timestep)
            count += 1
    current = count * 1.602176634 / end_time14[k][-1] * 100000.0  # pA
    conductance = current * 1000 / args.volt  # pS
    print_conductance_summary(args.volt, end_time14, k, count, current, conductance)

    ##########################
    # Safety check
    ##########################
    seq_list = [(np.array([4, 2, 3]), "leak current up"),
                (np.array([3, 2, 4]), "leak current down"),
                (np.array([1, 2]), "SF broken, inside-out"),
                (np.array([2, 1]), "SF broken, outside-in"),
                (np.array([1, 3, 4]), "safety check 1 for current up"),
                (np.array([3, 4, 1]), "safety check 2 for current up"),
                (np.array([4, 3, 1]), "safety check 1 for current down"),
                (np.array([1, 4, 3]), "safety check 2 for current down"),
                ]
    print_seq(seq_list, ions_state14, ions_resident_time14, end_time14, traj_timestep, args.volt)
