import splitfolders

#### input dataset that want to split
# input_folder = "/home/v-boli7/azure_storage/data/imagenet-sketch/sketch"

# output_folder = "/home/v-boli7/azure_storage/data/imagenet-sketch-split"

# splitfolders.ratio(input_folder, output=output_folder, seed=1337, ratio=(0.8, 0.1, 0.1))

name_list = []
with open("/home/v-boli7/projects/Genforce/caltech-256/labels.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.split(".")[1].replace('-101', '')
        name_list.append(line.strip())

print(name_list)