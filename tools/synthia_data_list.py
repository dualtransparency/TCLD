import os

from importmagician import import_from
with import_from('./'):
    from configs.semantic_segmentation.common.datasets._utils import SYNTHIA_ROOT as base


# Pad with 0
def pad(x):
    zero = '0'
    length = len(x)
    if length < 7:
        x = zero * (7 - length) + x
    x += '\n'

    return x


# Count
start = 0
end = 9399
train_list = [pad(str(x)) for x in range(start, end + 1)]
print('Whole training set size: ' + str(len(train_list)))

# Save training list
lists_dir = os.path.join(base, "data_lists")
if not os.path.exists(lists_dir):
    os.makedirs(lists_dir)
with open(os.path.join(lists_dir, "train.txt"), "w") as f:
    f.writelines(train_list)
print("Complete.")
