import os
import subprocess

dataset_list = [
    "caltech-101",
    "cifar-10",
    "cifar-100",
    "country211",
    "dtd",
    "eurosat_clip",
    "fer-2013",
    "fgvc-aircraft-2013b-variants102",
    "food-101",
    "gtsrb",
    "hateful-memes",
    "kitti-distance",
    "mnist",
    "oxford-flower-102",
    "oxford-iiit-pets",
    "patch-camelyon",
    "rendered-sst2",
    "resisc45_clip",
    "stanford-cars",
    "voc-2007-classification",
]

cmd = "python pipeline.py --dataset_name {} --ref_image=retrieval --ref_shot=20 --ref_image_path=/home/v-boli7/azure_storage/cvinwild/ic_benchmark/bamboo_vitb16/t2t_sep_prompt_30nn_archive/images --output_dir /home/v-boli7/azure_storage/cvinwild/ic_benchmark/bamboo_vitb16/SD_i2i_ret20 --elevator_root_path /home/v-boli7/azure_storage/cvinwild/ic_benchmark/bamboo_vitb16 --option=textual_inversion_prepare"
for ds in dataset_list:
    print(ds)
    subprocess.call(cmd.format(ds), shell=True)
