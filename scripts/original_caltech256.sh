cd ..;

python -m vision_benchmark.commands.linear_probe \
    --ds=./vision_benchmark/resources/datasets/sun397.yaml \
    --model=./vision_benchmark/resources/model/vitb32_CLIP.yaml \
    --no-tuning=False --lr=0.1 --l2=1e-6 \
    LOG_LEVEL INFO MODEL.CLIP_FP32 True \
    DATASET.NUM_SAMPLES_PER_CLASS 5 \
    DATASET.ROOT /home/v-boli7/azure_storage/cvinwild/ic_benchmark/bamboo_vitb16/datasets \
    OUTPUT_DIR /home/v-boli7/azure_storage/cvinwild/ic_benchmark/bamboo_vitb16/datasets/log \
    DATASET.RANDOM_SEED_SAMPLING 0 \
    DATASET.MERGE_TRAIN_VAL_FINAL_RUN True \
    TRAIN.FREEZE_IMAGE_BACKBONE True \
    TRAIN.MULTI_TASK False TRAIN.INIT_HEAD_WITH_TEXT_ENCODER True \
    TRAIN.MERGE_ENCODER_AND_HEAD_PROJ False \
    TRAIN.WITH_GENERATED_IMAGES False \
    KNOWLEDGE.WORDNET.USE_HIERARCHY False \
    KNOWLEDGE.WORDNET.USE_DEFINITION False \
    KNOWLEDGE.WIKITIONARY.USE_DEFINITION False \
    KNOWLEDGE.GPT3.USE_GPT3 False \
    KNOWLEDGE.AGGREGATION.NUM_GPT3_ITEMS 0