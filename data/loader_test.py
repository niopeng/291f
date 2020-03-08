import os

def load_drc_split(base_dir, synth_set):
    filename = os.path.join(base_dir, "{}.file".format(synth_set))
    lines = [line.rstrip('\n') for line in open(filename)]

    k = 3  # first 3 are garbage
    split = {}
    while k < len(lines):
        _,_,name,_,_,num = lines[k:k+6]
        k += 6
        num = int(num)
        split_curr = []
        print(num, name)
        # print(a)
        for i in range(num):
            _, _, _, _, model_name = lines[k:k+5]
            k += 5
            split_curr.append(model_name)
            print(model_name)
            print(a)
        split[name] = split_curr

    return split

load_drc_split("./splits", "03001627")



