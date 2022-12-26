#!/usr/bin/env python3

import argparse


class PermEvent:
    def __init__(self, at_index: int, time, up=True):
        self.at_index = at_index
        self.time = time  # 1 number or a list of 2 numbers
        self.up = up

    def __str__(self):
        s = "Time " + str(self.time)
        s += " %d " % self.at_index
        if self.up:
            s += "up"
        else:
            s += "down"
        return s


def read_perm_up_down(perm_up, perm_down):
    """
    
    :param perm_up: file
    :param perm_down: file
    :return: a list of PermEvent
    """
    perm_list = []
    with open(perm_up) as f:
        for line in f:
            line = line.rstrip()
            words = line.split()
            e = PermEvent(int(words[1]), float(words[0]), True)
            perm_list.append(e)

    with open(perm_down) as f:
        for line in f:
            line = line.rstrip()
            words = line.split()
            e = PermEvent(int(words[1]), float(words[0]), False)
            perm_list.append(e)
    perm_list = sorted(perm_list, key=lambda x: x.time)
    return perm_list


def read_cylinder(cylinder_file):
    """
    :param cylinder_file: 
    :return: a list of PermEvent
    """
    perm_list = []
    with open(cylinder_file) as f:
        for line in f:
            if "Perm: [4 1 3]" in line:
                line = line.rstrip()
                words = line.split()
                at_index = int(words[4])
                time = [float(words[10]), float(words[11])]
                perm_list.append(PermEvent(at_index, time))
    perm_list = sorted(perm_list, key=lambda x: x.time[0])
    return perm_list


def equal(event_xtck: PermEvent, event_cylinder: PermEvent):
    if not event_xtck.at_index == event_cylinder.at_index:
        return False
    if not (event_cylinder.time[0] < event_xtck.time and event_xtck.time < event_cylinder.time[1]):
        return False
    if not event_xtck.up == event_cylinder.up:
        return False
    return True


def longest_common_subsequence(seq_xtck, seq_cylinder):
    if len(seq_xtck) == 0 and len(seq_cylinder) == 0:
        return [], 0
    elif len(seq_cylinder) == 0:
        seq = []
        for i in seq_xtck:
            seq.append([i, None])
        return seq, 0
    elif len(seq_xtck) == 0:
        seq = []
        for i in seq_cylinder:
            seq.append([None, i])
        return seq, 0
    elif equal(seq_xtck[-1], seq_cylinder[-1]):
        seq, length = longest_common_subsequence(seq_xtck[:-1], seq_cylinder[:-1])
        seq.append( [seq_xtck[-1], seq_cylinder[-1]] )
        return seq, length+1
    elif seq_xtck[-1].time > seq_cylinder[-1].time[1]:
        seq, length = longest_common_subsequence(seq_xtck[:-1], seq_cylinder)
        seq.append([seq_xtck[-1], None])
        return seq, length
    elif seq_xtck[-1].time < seq_cylinder[-1].time[0]:
        seq, length = longest_common_subsequence(seq_xtck, seq_cylinder[:-1])
        seq.append([None, seq_cylinder[-1]])
        return seq, length
    elif seq_xtck[-1].up == False:
        seq, length = longest_common_subsequence(seq_xtck[:-1], seq_cylinder)
        seq.append([seq_xtck[-1], None])
        return seq, length
    elif equal(seq_xtck[-2], seq_cylinder[-1]):
        seq, length = longest_common_subsequence(seq_xtck[:-2], seq_cylinder[:-1])
        seq.append([seq_xtck[-2], seq_cylinder[-1]])
        length += 1
        seq.append([seq_xtck[-1], None])
        return seq, length
    elif equal(seq_xtck[-1], seq_cylinder[-2]):
        seq, length = longest_common_subsequence(seq_xtck[:-1], seq_cylinder[:-2])
        seq.append([seq_xtck[-1], seq_cylinder[-2]])
        length += 1
        seq.append([None, seq_cylinder[-1]])
        return seq, length
    else:
        seq1, length1 = longest_common_subsequence(seq_xtck[:-1], seq_cylinder)
        seq2, length2 = longest_common_subsequence(seq_xtck,      seq_cylinder[:-1])
        if length1 > length2:
            seq1.append([seq_xtck[-1], None])
            return seq1, length1
        else:
            seq2.append([None, seq_cylinder[-1]])
            return seq2, length2




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-perm_up",
                        dest="perm_up",
                        help="perm_up.dat from xtck",
                        type=argparse.FileType('r'),
                        required=True)

    parser.add_argument("-perm_down",
                        dest="perm_down",
                        help="perm_down.dat from xtck",
                        type=argparse.FileType('r'),
                        required=True)

    parser.add_argument("-cylinder",
                        dest="cylinder",
                        help="cylinder.dat from Chenggong",
                        type=argparse.FileType('r'),
                        required=True)

    args = parser.parse_args()
    seq_xtck = read_perm_up_down(args.perm_up.name, args.perm_down.name)
    seq_cylinder = read_cylinder(args.cylinder.name)
    e_list, l = longest_common_subsequence(seq_xtck, seq_cylinder)
    for e in e_list:
        for i in e:
            if not (i is None):
                print(i.at_index, end=" ")
                if i.up:
                    print("up   ", end="")
                else:
                    print("down ", end="")
            else:
                print("NONE      ", end="")
        if not e[1] is None:
            print(e[1].time)
        else:
            print()