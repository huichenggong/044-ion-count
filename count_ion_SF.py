#!/usr/bin/env python3
import count_ion
from count_ion import *


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
            for i in range(len(states)-2):
                if states[i+1] == forbidden:
                    matched[k] = (np.array([], dtype=int),)
                    break
                elif np.all(states[i:i+3] == seq):
                    matched[k] = (np.array([i]), )
                    break
        else:
            matched[k] = (np.array([], dtype=int),)
    return matched


def match_tail(ions_state15, seq, forbidden):
    matched = {}
    for k in ions_state15:
        states = ions_state15[k]
        if states[-1] == seq[-1]:
            for i in range(len(states)-3, -1, -1):
                if states[i+1] == forbidden:
                    matched[k] = (np.array([], dtype=int),)
                    break
                elif np.all(states[i:i+3] == seq):
                    matched[k] = (np.array([i]), )
                    break
        else:
            matched[k] = (np.array([], dtype=int),)
    return matched





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
    parser.add_argument("-SF_seq",
                        dest="SF_seq",
                        metavar="list of string",
                        help="THR VAL GLY TYR GLY",
                        type=str,
                        nargs=5)
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
    SF_seq = args.SF_seq
    print("#################################################################################")
    print("PDB top file:", args.top.name)
    print("xtc traj file:", xtc_full_max)
    print("Ion name in this pdb should be:", K_name)
    print("The number of frame loading each time will be:", chunk)
    print("The Voltage in this simulation is: ", args.volt, "mV")
    print("XTC traj will be strided:", args.stride)
    print("The sequence of the SF is ", SF_seq)
    print("#################################################################################")

    traj_pdb = md.load(top)
    # prepare time step
    for tmp in md.iterload(xtc_full_max, top=top, chunk=2, stride=args.stride):
        traj_timestep = tmp.timestep
        break

    # prepare atom index for cylinder
    if (args.SF_seq is None):
        raise ValueError("Please provide -SF_seq")
    print("#################################################################################")
    S00, S01, S12, S23, S34, S45 = auto_find_SF_index(traj_pdb, SF_seq)
    ion_index = traj_pdb.topology.select('name ' + K_name)
    print("Number of ions found", len(ion_index))
    print("The ion index (0 base):", ion_index)
    print("S01 of the cylinder")
    for at in S01:
        print(traj_pdb.topology.atom(at))
    print("S23 of the cylinder")
    for at in S23:
        print(traj_pdb.topology.atom(at))
    print("S45 of the cylinder")
    for at in S45:
        print(traj_pdb.topology.atom(at))
    # load xtc iteratively and assign state for each ion
    print("Assign state to each ion for each frame")
    print("Load xtc traj chunk by chunk.", end="")
    ions_state_dict = count_ion.assign_ion_state_chunk(
        xtc_file=xtc_full_max,
        top=top,
        stride=args.stride,
        chunk=chunk,
        assign_fun=count_ion.potassium_state_assign_cylider_double,
        S01=S01, S23=S23, S45=S45,
        center_ind=S01 + S23 + S45,
        ion_index=ion_index,
        rad=args.cylRAD,
    )
    ions_state15, ions_resident_time15, end_time15 = ion_state_short_map(ions_state_dict, traj_timestep)

    for k in ions_state_dict:
        states = ions_state_dict[k]
        ions_state_dict[k][states == 5] = 1
    ions_state14, ions_resident_time14, end_time14 = ion_state_short_map(ions_state_dict, traj_timestep)

    matched_h = match_head(ions_state15, seq=np.array([5, 1, 3]), forbidden=4)
    matched_t = match_tail(ions_state15, seq=np.array([4, 5, 1]), forbidden=3)
    #print(matched_h)
    #print(matched_t)
    if args.stride is None:
        stride = 1
    else:
        stride = args.stride
    count = 0

    print("\n###############################")
    print("# New sequence start:", np.array([4, 1, 3]), "proper current up")
    print("#################################")
    # check head/tail
    for matched, seq in [(matched_h, np.array([5, 1, 3])), (matched_t, np.array([4, 5, 1]))]:
        for k in matched:
            # print("K ion index", k)
            # print()
            for i in matched[k][0]:
                count_ion.print_seq_str(seq, k, ions_resident_time15, end_time15, len(seq), i, stride, traj_timestep)
                count += 1
    # check permeation in the middle
    seq = np.array([4, 1, 3])
    matched_dict = match_seqs(ions_state14, seq)
    for k in matched_dict:
        for i in matched_dict[k][0]:
            print_seq_str(seq, k, ions_resident_time14, end_time14, len(seq), i, stride, traj_timestep)
            count += 1
    current = count * 1.602176634 / end_time14[k][-1] * 100000.0  # pA
    conductance = current * 1000 / args.volt  # pS
    count_ion.print_conductance_summary(args.volt, end_time14, k, count, current, conductance)


    seq_list = count_ion.seq_list_dict["Cylinder"][1:]
    count_ion.print_seq(seq_list, ions_state14, ions_resident_time14, end_time14, traj_timestep, args.volt,
                        stride=args.stride)
