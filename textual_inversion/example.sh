export MODEL_NAME="stabilityai/stable-diffusion-2-1"
# model_id = "stabilityai/stable-diffusion-2-1"
export DATA_DIR="./cat_statue"

accelerate launch --config_file ../accelerate_config.yaml train_textual_inversion.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_data_dir=$DATA_DIR \
    --learnable_property="object" \
    --placeholder_token="<cat-toy>" --initializer_token="toy" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=3000 \
    --learning_rate=5.0e-04 --scale_lr \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --output_dir="textual_inversion_cat"

cd ./textual_inversion && CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file accelerate_config.yaml train_textual_inversion.py --pretrained_model_name_or_path='stabilityai/stable-diffusion-2' --train_data_dir=/home/v-boli7/azure_storage/cvinwild/ic_benchmark/bamboo_vitb16/textual_inversion_generated/voc-2007-classification/bicycle_instance_dir --learnable_property='object' --placeholder_token='<voc-2007-classification bicycle>' --initializer_token=bicycle --resolution=512 --train_batch_size=4 --gradient_accumulation_steps=1 --max_train_steps=500 --checkpointing_steps=500 --learning_rate=5.0e-04 --scale_lr --lr_scheduler='constant' --lr_warmup_steps=0 --output_dir=/home/v-boli7/azure_storage/cvinwild/ic_benchmark/bamboo_vitb16/textual_inversion_generated/voc-2007-classification/bicycle_model_output --lambda_global_clip_loss 1.0
