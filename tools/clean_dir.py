import os
import uuid
from PIL import Image

for category in os.listdir('/home/v-boli7/azure_storage/cvinwild/ic_benchmark/bamboo_vitb16/TI_original_ref/mnist'):
    for img in os.listdir('/home/v-boli7/azure_storage/cvinwild/ic_benchmark/bamboo_vitb16/TI_original_ref/mnist/' + category):
        img_file = Image.open(f'/home/v-boli7/azure_storage/cvinwild/ic_benchmark/bamboo_vitb16/TI_original_ref/mnist/{category}/{img}').convert('RGB')
        img_file.save(f'/home/v-boli7/azure_storage/cvinwild/ic_benchmark/bamboo_vitb16/TI_original_ref/mnist/{category}/{uuid.uuid4()}.jpg')
    