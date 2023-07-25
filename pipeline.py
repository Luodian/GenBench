# from tools.auxillary_func import multi_gpu_launcher, prompt_generator, convert_dirs_to_zip, write_txt
from vision_benchmark.datasets import class_map, template_map
from vision_benchmark.common.constants import get_dataset_hub, VISION_DATASET_STORAGE
from loguru import logger
import sys
import os
import shutil
import json
import argparse
import glob

logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD:HH:mm:ss}</green> | <cyan>{name}</cyan><cyan>:line {line}</cyan>: <level>{message}</level>", level="INFO")
os.environ["WANDB_API_KEY"] = "abc1859572354a66fc85b2ad1d1009add929cbfa"
import os
from loguru import logger
import shutil
from generation_utils import sampling_images, dreambooth_generate_images, textual_inversion_prepare_images, sampling_images_i2i, sampling_images_i2v
from packing_utils import packing_retrieval_images, packing_generated_images
from evaluation_utils import run_elevator, zeroshot_metric, feature_distances
from common_utils import configure_meta_info
import wandb

def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def parse_args():
    parser = argparse.ArgumentParser()
    # choices=["generate", "pack_generated", "pack_retrieval", "elevator", "configure", "all"]
    # basic args
    parser.add_argument("--option", type=lambda s: [item for item in s.split(",")], default="pack_retrieval,elevator")
    parser.add_argument("--dataset_name", type=str, default="hateful-memes")
    parser.add_argument("--dataset_type", type=str, default=None)
    parser.add_argument("--language_enc", type=str, default='lang_enc')
    parser.add_argument("--neg_prompts", type=boolean_string, default=False)
    parser.add_argument("--output_dir", type=str, default="./work_dirs")
    parser.add_argument("--workspace_dir", type=str, default="./dreambooth")
    # args for stable diffusion
    parser.add_argument("--generation_model_path", type=str, default="/home/v-baike/azure_storage/models/512-base-ema.ckpt")
    # args for textual inversion
    parser.add_argument("--sharp_focus", type=boolean_string, default=False)
    parser.add_argument("--sentence_expansion", type=boolean_string, default=False)
    parser.add_argument("--unique_id", type=str, default="retrieval")
    parser.add_argument("--total_num_temps", type=int, default=16, help="total number of templates, maximum range")
    parser.add_argument("--sample_num", type=int, default=4)
    parser.add_argument("--sliced_num", type=int, default=8, help="sampling prompts per gpu, bsc = sliced_num * sample_num")
    parser.add_argument("--lambda_global_clip_loss", type=float, default=1.0)
    # args for packing
    parser.add_argument("--clip_selection", type=int, default=0, help="0: random, 1-x: clip selection with x shots")
    parser.add_argument("--random_selection_num", type=int, default=5, help="0: no clip, 1-x: clip selection with x shots")
    parser.add_argument("--force_repack", type=boolean_string, default=False)
    parser.add_argument("--force_regen", type=boolean_string, default=False)
    parser.add_argument("--ref_image", type=str, default="original")
    parser.add_argument("--ref_image_path", type=str, default="/home/v-boli7/azure_storage/cvinwild/ic_benchmark/bamboo_vitb16/t2t_sep_prompt_30nn")
    parser.add_argument("--ref_shot", type=int, default=5)
    parser.add_argument("--steps", type=int, default=500)
    # args for elevator
    parser.add_argument("--elevator_root_path", type=str, default="./vision_benchmark/outputs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--model", type=str, default="./vision_benchmark/resources/model/vitb32_CLIP.yaml")
    parser.add_argument("--ds", type=str, default="./vision_benchmark/resources/datasets/cifar-10.yaml")
    parser.add_argument("--wandb_tag", type=str, default="Genforce")
    parser.add_argument("--no_tuning", type=boolean_string, default=False)
    parser.add_argument("--with_ext_train", type=str, default="extended")
    parser.add_argument("--with_generated", type=boolean_string, default=False)
    parser.add_argument("--with_retrieval", type=boolean_string, default=False)
    parser.add_argument('--text_feature_only', help='consider text feature or not.', default=False, action='store_true')
    parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)   
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    args = parse_args()

    logger.info(args)
    # check option issues
    if "pack_retrieval" in args.option and "pack_generated" in args.option:
        raise ValueError("Can not pack both generated and retrieval images")
    if "generate" in args.option and "pack_retrieval" in args.option:
        raise ValueError("Can not both generate images and pack retrieval images")

    elevator_root_path = args.elevator_root_path
    dataset_hub = get_dataset_hub()
    from vision_datasets import Usages

    manifest = dataset_hub.create_dataset_manifest(VISION_DATASET_STORAGE, local_dir=f"{elevator_root_path}/datasets/", name=args.dataset_name, usage=Usages.TRAIN_PURPOSE)
    template_names = template_map[args.dataset_name]
    if manifest:
        class_names = manifest[0].labelmap
        elevator_dataset_full_path = os.path.join(elevator_root_path, "datasets", manifest[1].root_folder)
        args.elevator_dataset_full_path = elevator_dataset_full_path
        logger.info(f"Meta info of {args.dataset_name} is ready to use.")

    try:
        concept_list = class_map[args.dataset_name]
        template_list = template_map[args.dataset_name]
    except:
        logger.info(f"Dataset {args.dataset_name} not supported yet.")
        logger.info("Supported datasets are: ", list(class_map.keys()))
    
    args.output_dir = os.path.join(args.output_dir, f"{args.dataset_name}")
    if "configure" in args.option:
        configure_meta_info(args=args)

    if "textual_inversion_prepare" in args.option:
        textual_inversion_prepare_images(manifest=manifest, args=args)

    if "sample_images" in args.option:
        sampling_images(args=args)

    if "pack_generated" in args.option:
        packing_generated_images(class_names=concept_list, args=args)

    if "pack_retrieval" in args.option:
        packing_retrieval_images(class_names=concept_list, args=args)
        # configure_meta_info(dataset_type=args.dataset_type)

    if "elevator" in args.option:
        run_elevator(
            elevator_root_path=elevator_root_path,
            args = args
        )
        
    if "zeroshot_metric" in args.option:
        args.ds = os.path.join('./vision_benchmark/resources/datasets', f"{args.dataset_name}.yaml")
        wandb.init(project="zeroshot_metric", entity="drluodian", name=f"{args.dataset_name}-{args.dataset_type}", tags=[args.wandb_tag], config=args)
        zeroshot_metric(args = args)

    if "proxy_distance" in args.option:
        args.ds = os.path.join('./vision_benchmark/resources/datasets', f"{args.dataset_name}.yaml")
        feature_distances(args = args)
