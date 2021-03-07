import os


def init_acc_regime(dataname):
    if 'RAVEN' in dataname:
        return init_acc_regime_raven()
    else:
        return init_acc_regime_pgm()


def update_acc_regime(dataname, acc_regime, model_output, target, structure_encoded, data_file):
    if 'RAVEN' in dataname:
        update_acc_regime_raven(acc_regime, model_output, target, data_file)
    else:
        update_acc_regime_pgm(acc_regime, model_output, target, structure_encoded)


def init_acc_regime_raven():
    acc_regime = {"center_single": [0, 0],
                  "distribute_four": [0, 0],
                  "distribute_nine": [0, 0],
                  "in_center_single_out_center_single": [0, 0],
                  "in_distribute_four_out_center_single": [0, 0],
                  "left_center_single_right_center_single": [0, 0],
                  "up_center_single_down_center_single": [0, 0],
                  }
    return acc_regime


def update_acc_regime_raven(acc_regime, model_output, target, data_file):
    acc_one = model_output.data.max(1)[1] == target
    for i in range(model_output.shape[0]):
        regime = data_file[i].split('\\' if os.name == 'nt' else '/')[0]
        acc_regime[regime][0] += acc_one[i].item()
        acc_regime[regime][1] += 1


number = {1: 'one', 2: 'two', 3: 'three', 4: 'four'}


def init_acc_regime_pgm():
    acc_regime = {"one": [0, 0],
                  "two": [0, 0],
                  "three": [0, 0],
                  "four": [0, 0],
                  "shape": [0, 0],
                  "line": [0, 0],
                  "both": [0, 0],
                  "shape-color-prog": [0, 0],
                  "shape-color-xor": [0, 0],
                  "shape-color-or": [0, 0],
                  "shape-color-and": [0, 0],
                  "shape-color-union": [0, 0],
                  "shape-number-prog": [0, 0],
                  "shape-number-xor": None,
                  "shape-number-or": None,
                  "shape-number-and": None,
                  "shape-number-union": [0, 0],
                  "shape-position-prog": None,
                  "shape-position-xor": [0, 0],
                  "shape-position-or": [0, 0],
                  "shape-position-and": [0, 0],
                  "shape-position-union": None,
                  "shape-size-prog": [0, 0],
                  "shape-size-xor": [0, 0],
                  "shape-size-or": [0, 0],
                  "shape-size-and": [0, 0],
                  "shape-size-union": [0, 0],
                  "shape-type-prog": [0, 0],
                  "shape-type-xor": [0, 0],
                  "shape-type-or": [0, 0],
                  "shape-type-and": [0, 0],
                  "shape-type-union": [0, 0],
                  "line-color-prog": [0, 0],
                  "line-color-xor": [0, 0],
                  "line-color-or": [0, 0],
                  "line-color-and": [0, 0],
                  "line-color-union": [0, 0],
                  "line-type-prog": None,
                  "line-type-xor": [0, 0],
                  "line-type-or": [0, 0],
                  "line-type-and": [0, 0],
                  "line-type-union": [0, 0]
                  }
    return acc_regime


def update_acc_regime_pgm(acc_regime, model_output, target, structure_encoded):
    acc_one = model_output.data.max(1)[1] == target
    meta_target = structure_encoded.sum(1) > 0
    num_rules = (structure_encoded.sum(2) > 0).sum(1)

    for i in range(model_output.shape[0]):
        # Num rules
        acc_regime[number[num_rules[i].item()]][0] += acc_one[i].item()
        acc_regime[number[num_rules[i].item()]][1] += 1
        # Line
        if meta_target[i, 0] == 0:
            acc_regime["line"][0] += acc_one[i].item()
            acc_regime["line"][1] += 1

        # Shape
        if meta_target[i, 1] == 0:
            acc_regime["shape"][0] += acc_one[i].item()
            acc_regime["shape"][1] += 1

        # Both
        if meta_target[i, 0] != 0 and meta_target[i, 1] != 0:
            acc_regime["both"][0] += acc_one[i].item()
            acc_regime["both"][1] += 1

        if num_rules[i].item() == 1:
            # Shape
            if meta_target[i, 0]:
                # Color
                if meta_target[i, 2]:
                    if meta_target[i, 7]:  # progression
                        acc_regime["shape-color-prog"][0] += acc_one[i].item()
                        acc_regime["shape-color-prog"][1] += 1
                    if meta_target[i, 8]:  # xor
                        acc_regime["shape-color-xor"][0] += acc_one[i].item()
                        acc_regime["shape-color-xor"][1] += 1
                    if meta_target[i, 9]:  # or
                        acc_regime["shape-color-or"][0] += acc_one[i].item()
                        acc_regime["shape-color-or"][1] += 1
                    if meta_target[i, 10]:  # and
                        acc_regime["shape-color-and"][0] += acc_one[i].item()
                        acc_regime["shape-color-and"][1] += 1
                    if meta_target[i, 11]:  # union
                        acc_regime["shape-color-union"][0] += acc_one[i].item()
                        acc_regime["shape-color-union"][1] += 1
                # Number
                if meta_target[i, 3]:
                    if meta_target[i, 7]:  # progression
                        acc_regime["shape-number-prog"][0] += acc_one[i].item()
                        acc_regime["shape-number-prog"][1] += 1
                    if meta_target[i, 8]:  # xor
                        raise Exception('No shape-number-xor')
                    if meta_target[i, 9]:  # or
                        raise Exception('No shape-number-or')
                    if meta_target[i, 10]:  # and
                        raise Exception('No shape-number-and')
                    if meta_target[i, 11]:  # union
                        acc_regime["shape-number-union"][0] += acc_one[i].item()
                        acc_regime["shape-number-union"][1] += 1
                # Position
                if meta_target[i, 4]:
                    if meta_target[i, 7]:  # progression
                        raise Exception('No shape-position-prog')
                    if meta_target[i, 8]:  # xor
                        acc_regime["shape-position-xor"][0] += acc_one[i].item()
                        acc_regime["shape-position-xor"][1] += 1
                    if meta_target[i, 9]:  # or
                        acc_regime["shape-position-or"][0] += acc_one[i].item()
                        acc_regime["shape-position-or"][1] += 1
                    if meta_target[i, 10]:  # and
                        acc_regime["shape-position-and"][0] += acc_one[i].item()
                        acc_regime["shape-position-and"][1] += 1
                    if meta_target[i, 11]:  # union
                        raise Exception('No shape-position-union')
                # Size
                if meta_target[i, 5]:
                    if meta_target[i, 7]:  # progression
                        acc_regime["shape-size-prog"][0] += acc_one[i].item()
                        acc_regime["shape-size-prog"][1] += 1
                    if meta_target[i, 8]:  # xor
                        acc_regime["shape-size-xor"][0] += acc_one[i].item()
                        acc_regime["shape-size-xor"][1] += 1
                    if meta_target[i, 9]:  # or
                        acc_regime["shape-size-or"][0] += acc_one[i].item()
                        acc_regime["shape-size-or"][1] += 1
                    if meta_target[i, 10]:  # and
                        acc_regime["shape-size-and"][0] += acc_one[i].item()
                        acc_regime["shape-size-and"][1] += 1
                    if meta_target[i, 11]:  # union
                        acc_regime["shape-size-union"][0] += acc_one[i].item()
                        acc_regime["shape-size-union"][1] += 1
                # Type
                if meta_target[i, 6]:
                    if meta_target[i, 7]:  # progression
                        acc_regime["shape-type-prog"][0] += acc_one[i].item()
                        acc_regime["shape-type-prog"][1] += 1
                    if meta_target[i, 8]:  # xor
                        acc_regime["shape-type-xor"][0] += acc_one[i].item()
                        acc_regime["shape-type-xor"][1] += 1
                    if meta_target[i, 9]:  # or
                        acc_regime["shape-type-or"][0] += acc_one[i].item()
                        acc_regime["shape-type-or"][1] += 1
                    if meta_target[i, 10]:  # and
                        acc_regime["shape-type-and"][0] += acc_one[i].item()
                        acc_regime["shape-type-and"][1] += 1
                    if meta_target[i, 11]:  # union
                        acc_regime["shape-type-union"][0] += acc_one[i].item()
                        acc_regime["shape-type-union"][1] += 1
            # Line
            if meta_target[i, 1]:
                # Color
                if meta_target[i, 2]:
                    if meta_target[i, 7]:  # progression
                        acc_regime["line-color-prog"][0] += acc_one[i].item()
                        acc_regime["line-color-prog"][1] += 1
                    if meta_target[i, 8]:  # xor
                        acc_regime["line-color-xor"][0] += acc_one[i].item()
                        acc_regime["line-color-xor"][1] += 1
                    if meta_target[i, 9]:  # or
                        acc_regime["line-color-or"][0] += acc_one[i].item()
                        acc_regime["line-color-or"][1] += 1
                    if meta_target[i, 10]:  # and
                        acc_regime["line-color-and"][0] += acc_one[i].item()
                        acc_regime["line-color-and"][1] += 1
                    if meta_target[i, 11]:  # union
                        acc_regime["line-color-union"][0] += acc_one[i].item()
                        acc_regime["line-color-union"][1] += 1
                # Type
                if meta_target[i, 6]:
                    if meta_target[i, 7]:  # progression
                        raise Exception('No line-type-progression')
                    if meta_target[i, 8]:  # xor
                        acc_regime["line-type-xor"][0] += acc_one[i].item()
                        acc_regime["line-type-xor"][1] += 1
                    if meta_target[i, 9]:  # or
                        acc_regime["line-type-or"][0] += acc_one[i].item()
                        acc_regime["line-type-or"][1] += 1
                    if meta_target[i, 10]:  # and
                        acc_regime["line-type-and"][0] += acc_one[i].item()
                        acc_regime["line-type-and"][1] += 1
                    if meta_target[i, 11]:  # union
                        acc_regime["line-type-union"][0] += acc_one[i].item()
                        acc_regime["line-type-union"][1] += 1