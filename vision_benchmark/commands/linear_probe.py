"""
Linear Probe with sklearn Logistic Regression or linear model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from loguru import logger
import sys

import numpy as np
import random

import os
import wandb
os.environ['WANDB_API_KEY'] = 'abc1859572354a66fc85b2ad1d1009add929cbfa'

from vision_benchmark.utils import comm, create_logger
from vision_benchmark.evaluation import construct_dataloader, full_model_finetune, construct_IC_multitask_dataloader, IC_Multitask_full_model_finetune
from vision_benchmark.config import config, update_config

# These 2 lines are a walk-around for "Too many open files error". Refer: https://github.com/pytorch/pytorch/issues/11201
import torch.multiprocessing
from ..common.utils import log_arg_env_config, submit_predictions

torch.multiprocessing.set_sharing_strategy("file_system")

MULTILABEL_DATASETS = {"chestx-ray8"}


def add_linear_probing_args(parser):
    parser.add_argument("--ds", required=False, help="Evaluation dataset configure file name.", type=str)
    parser.add_argument("--model", required=True, help="Evaluation model configure file name", type=str)
    parser.add_argument("--submit-predictions", help="submit predictions and model info to leaderboard.", default=False, action="store_true")
    parser.add_argument("--submit-by", help="Person who submits the results.", type=str)

    parser.add_argument("--no-tuning", help="No hyperparameter-tuning.", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument(
        "--l2", help="(Inverse) L2 regularization strength. This option is only useful when option --no-tuning is True.", default=0.316, type=float
    )
    parser.add_argument(
        "--lr", help="Test with a specific learning rate. This option is only useful when option --no-tuning is True.", default=0.001, type=float
    )
    parser.add_argument("--run", help="Run id", default=1, type=int)
    parser.add_argument("--fix_seed", help="Fix the random seed. [-1] not fixing the seeds", default=0, type=int)
    parser.add_argument("--save-predictions", help="save predictions logits for analysis.", default=True, action="store_true")

    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)


def main():
    parser = argparse.ArgumentParser(description="Test a classification model, with linear probing.")
    add_linear_probing_args(parser)
    args = parser.parse_args()

    args.cfg = args.ds
    update_config(config, args)
    args.cfg = args.model
    update_config(config, args)
    config.defrost()
    config.NAME = ""
    config.freeze()

    if args.submit_predictions:
        assert args.submit_by

    if args.fix_seed != -1:
        random.seed(args.fix_seed)
        np.random.seed(args.fix_seed)
        torch.manual_seed(args.fix_seed)
        torch.cuda.manual_seed_all(args.fix_seed)

    n_samples = str(config.DATASET.NUM_SAMPLES_PER_CLASS) if config.DATASET.NUM_SAMPLES_PER_CLASS >= 0 else "full"
    exp_name = f"{config.MODEL.NAME}_{config.DATASET.DATASET}_{n_samples}shot"
    if config.TRAIN.WITH_GENERATED_IMAGES:
        exp_name += f"_wgen_{config.TRAIN.WGEN_PER_CLASS}"
    elif config.TRAIN.WITH_RETRIEVAL_IMAGES:
        exp_name += f"_wret_{config.TRAIN.WRET_PER_CLASS}"

    if config.DATASET.NUM_SAMPLES_PER_CLASS == 1:
        config.defrost()
        config.DATASET.NUM_SAMPLES_PER_CLASS = 2
        config.DATASET.MERGE_TRAIN_VAL_FINAL_RUN = False
        config.freeze()

    # Follow MAE's design choice: not using global pool in linear probe
    if config.MODEL.NAME.startswith("mae_"):
        config.defrost()
        config.MODEL.SPEC.GLOBAL_POOL = False
        config.freeze()

    final_output_dir = create_logger(config, exp_name)
    if comm.is_main_process():
        log_arg_env_config(args, config, final_output_dir)

    if config.DATASET.DATASET == "patch-camelyon" and config.DATASET.NUM_SAMPLES_PER_CLASS == -1:
        # deal with patch camelyon large dataset (search using 10000-shot subset, final run with the full dataset)
        logger.info(f"Detecting large dataset with {config.DATASET.NUM_SAMPLES_PER_CLASS}-shot.")
        config.defrost()
        config.DATASET.NUM_SAMPLES_PER_CLASS = 10000
        config.freeze()
        logger.info(f"Used the subset ({config.DATASET.NUM_SAMPLES_PER_CLASS}-shot) to train the model.")

    wandb_tag = config.WANDB_TAG
    wandb.init(project="Genforce", entity="drluodian", name=exp_name, config=config, tags=[wandb_tag])

    if config.TRAIN.MULTI_TASK:
        # loader is a list of multi tasks
        task_list, train_dataloader, val_dataloader, test_dataloader = construct_IC_multitask_dataloader(config)
        if config.TRAIN.MULTI_TASK_HEAD == 'split':
            assert type(train_dataloader) == list
        elif config.TRAIN.MULTI_TASK_HEAD == 'shared':
            assert type(train_dataloader) == torch.utils.data.DataLoader    
        best_acc, model_info = IC_Multitask_full_model_finetune(
            train_dataloader, val_dataloader, test_dataloader, task_list, args.no_tuning, args.lr, args.l2, config
        )
    else:
        train_dataloader, val_dataloader, test_dataloader = construct_dataloader(config)
        best_acc, model_info = full_model_finetune(train_dataloader, val_dataloader, test_dataloader, args.no_tuning, args.lr, args.l2, config)
        wandb.log({"best_acc": best_acc})
        test_predictions = model_info["best_logits"]
    # Run linear probe

    if args.save_predictions:
        import json

        # a hack to control the json dump float accuracy
        # if you find the accuracy is not enough, pleae consider increasing `prec`.
        def json_prec_dump(data, prec=6):
            return json.dumps(json.loads(json.dumps(data), parse_float=lambda x: round(float(x), prec)))

        results_dict = {
            "model_name": config.MODEL.NAME,
            "dataset_name": config.DATASET.DATASET,
            "num_trainable_params": model_info.get("n_trainable_params", None),
            "num_params": model_info.get("n_params", None),
            "num_visual_params": model_info.get("n_visual_params", None),
            "num_backbone_params": model_info.get("n_backbone_params", None),
            "n_shot": config.DATASET.NUM_SAMPLES_PER_CLASS,
            "rnd_seeds": [config.DATASET.RANDOM_SEED_SAMPLING],
            "predictions": [test_predictions.tolist()],
        }
        json_string = json_prec_dump(results_dict)

        prediction_folder = os.path.join(config.OUTPUT_DIR, "predictions", exp_name)
        os.makedirs(prediction_folder, exist_ok=True)
        with open(os.path.join(prediction_folder, f"seed{config.DATASET.RANDOM_SEED_SAMPLING}_{config.DATASET.DATASET}.json"), "w") as outfile:
            outfile.write(json_string)
        logger.info(f"Saved predictions to {os.path.join(prediction_folder, f'seed{config.DATASET.RANDOM_SEED_SAMPLING}_{config.DATASET.DATASET}.json')}")


if __name__ == "__main__":
    main()
