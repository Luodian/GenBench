import codecs
import json
import os
import random
import shutil
import zipfile
from collections import defaultdict
from threading import ThreadPoolExecutor
from typing import List

import numpy as np
import torch
from accelerate import Accelerator
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from keytotext import pipeline as keytotext_pipeline
from loguru import logger
from PIL import Image

from common_utils import process_namings
from vision_benchmark.datasets import class_map, template_map

torch.backends.cuda.matmul.allow_tf32 = True
import uuid
from concurrent.futures import ThreadPoolExecutor


def textual_inversion_prepare_images(manifest, args=None):
    concept_list = class_map[args.dataset_name]
    template_list = template_map[args.dataset_name]
    os.makedirs(args.output_dir, exist_ok=True)

    if args.ref_image == "retrieval":
        per_category_selected = defaultdict(list)

        parent_output_dir = os.path.dirname(args.ref_image_path)
        image_path = os.path.join(parent_output_dir, f"images/{args.dataset_name}")
        meta_info_path = os.path.join(parent_output_dir, f"metas/{args.dataset_name}")
        ref_shot = args.ref_shot if args.ref_shot else 5
        for concept in concept_list:
            logger.info(f"Processing concept {concept} ...")
            saved_concept_dir = process_namings(concept)
            ref_concept_dir = os.path.join(args.output_dir, f"{saved_concept_dir}_instance_dir")
            if os.path.exists(ref_concept_dir):
                shutil.rmtree(ref_concept_dir)
            os.makedirs(ref_concept_dir)

            with open(os.path.join(meta_info_path, f"{saved_concept_dir}.json"), "r") as f:
                meta_info = json.load(f)
                logger.info(f"category: {saved_concept_dir}, meta_info: {len(meta_info.keys())}")
                for name in meta_info:
                    meta_info[name]["avg_dist"] = np.mean([float(x["dist"]) for x in meta_info[name]["search_meta"]])
                sorted_meta_info = {
                    k: v for k, v in sorted(meta_info.items(), key=lambda x: x[1]["avg_dist"], reverse=True)
                }
                per_category_selected[saved_concept_dir] = list(sorted_meta_info.keys())[:ref_shot]
                logger.info(f"category: {saved_concept_dir}, selected: {len(per_category_selected[saved_concept_dir])}")

            for idx, img_name in enumerate(per_category_selected[saved_concept_dir]):
                img_path = os.path.join(image_path, saved_concept_dir, img_name)
                img = Image.open(img_path).resize((224, 224))
                img.save(os.path.join(ref_concept_dir, img_name))
                logger.info(f"Saving image {img_name} to {ref_concept_dir}")
            # assert len(os.listdir(ref_concept_dir)) == ref_shot, f"Number of images in {ref_concept_dir} is not {ref_shot}."

    elif args.ref_image == "original":
        random_seed = 1337
        dataset_info = manifest[1]
        dataset_manifest = manifest[0].sample_few_shot_subset(20, random_seed)
        category_list = defaultdict(list)
        for image in dataset_manifest.images:
            try:
                cate_name = concept_list[image.labels[0]]
                image_tuple = (image.img_path, image.labels[0])
                category_list[cate_name].append(image_tuple)
            except:
                continue

        train_zipfile_path = dataset_manifest.images[0].img_path.split("@")[0]
        train_zipfile = zipfile.ZipFile(train_zipfile_path, "r")
        ref_shot = args.ref_shot if args.ref_shot else 5
        # train_zip_infolist = train_zipfile.infolist()
        for concept in concept_list:
            logger.info(f"Processing concept {concept} ...")
            saved_concept_dir = process_namings(concept)
            ref_concept_dir = os.path.join(args.output_dir, f"{saved_concept_dir}_instance_dir")
            os.makedirs(ref_concept_dir, exist_ok=True)

            # select 5 images from the concept as instance images
            ref_shot = ref_shot if len(category_list[concept]) >= ref_shot else len(category_list[concept])
            selected_instance = random.sample(category_list[concept], ref_shot)
            for idx, (img_path, label) in enumerate(selected_instance):
                img_file = train_zipfile.open(img_path.split("@")[1])
                img = Image.open(img_file).resize((224, 224))
                img.save(os.path.join(ref_concept_dir, f"{idx}.png"))
                logger.info(f"Saved image {idx} to {ref_concept_dir}")


def textual_inversion_sampling(args, template_list, concept_list):
    logger.info(f"Concept list: {concept_list}")
    logger.info(f"Template list: {template_list}")
    logger.info(f"Number of per category prompts: {len(template_list) * args.sample_num}")

    total_num_temps = 16
    expand_ratio = total_num_temps // len(template_list) + 1
    template_list = template_list * expand_ratio
    random.shuffle(template_list)
    template_list = template_list[:total_num_temps]

    model_id = args.generation_model_path
    logger.info("sampling image ...")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto").to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    print(pipe.unet.conv_out.state_dict()["weight"].stride())  # (2880, 9, 3, 1)
    pipe.unet.to(memory_format=torch.channels_last)  # in-place operation
    print(
        pipe.unet.conv_out.state_dict()["weight"].stride()
    )  # (2880, 1, 960, 320) having a stride of 1 for the 2nd dimension proves that it works

    if type(concept_list[0]) is list:
        concept_list = [c[0] for c in concept_list]

    for concept in concept_list:
        saved_concept_dir = process_namings(concept)
        unique_concept = f"{args.unique_id} {concept}"
        prompt_list = [template.format(unique_concept) for template in template_list]
        category_output_dir = os.path.join(args.output_dir, saved_concept_dir)
        os.makedirs(category_output_dir, exist_ok=True)
        logger.info(f"For concept {concept} ...")
        for prompt in prompt_list:
            images = pipe(
                prompt,
                num_inference_steps=50,
                guidance_scale=7.5,
                num_images_per_prompt=args.sample_num,
                height=512,
                width=512,
            ).images
            assert len(images) == args.sample_num
            saved_prompt = process_namings(prompt)
            for idx, image in enumerate(images):
                logger.info(f"Saving image of {saved_prompt} to {category_output_dir}")
                image.save(os.path.join(category_output_dir, f"{uuid.uuid4()}.png"))


def initialize_nlp():
    return keytotext_pipeline("mrm8488/t5-base-finetuned-common_gen")


def process_classnames(classnames: List[str], nlp, num=200):
    sentence_dict = {}
    for n in classnames:
        if isinstance(n, list):
            n = process_namings(n)
        sentences = [nlp([n], num_return_sequences=1, do_sample=True) for _ in range(num + 50)]
        sentence_dict[n] = list(set(sentences))
    return sentence_dict


def sample_sentences(sentence_dict: dict, num=200):
    return {k: random.sample(v, num) for k, v in sentence_dict.items()}


def load_model(args):
    try:
        logger.info(f"Loading model from {args.generation_model_path}")
        return StableDiffusionPipeline.from_pretrained(
            args.generation_model_path, torch_dtype=torch.float16, safety_checker=None, device_map="auto"
        )
    except:
        logger.info(f"Loading model from stabilityai/stable-diffusion-2-1-base")
        return StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16, safety_checker=None, device_map="auto"
        )


def save_image(saved_category_dir):
    def _save_image(img):
        img = img.resize((224, 224))
        img.save(os.path.join(saved_category_dir, f"{uuid.uuid4()}.png"))

    return _save_image


# read class names
def sampling_images(args=None):
    nlp = initialize_nlp()
    concept_list = class_map[args.dataset_name]
    if args.language_enc == "model_enc":
        sentence_dict = process_classnames(concept_list, nlp, num=args.total_num_temps)
        sampled_dict = sample_sentences(sentence_dict, num=args.total_num_temps)
    elif args.language_enc == "lang_enc":
        template_list = template_map[args.dataset_name]
    elif args.language_enc == "simple_template":
        # using simple prompt
        template_list = [
            "{}",
        ]
    else:
        raise NotImplementedError

    logger.info(f"Concept list: {concept_list}")
    logger.info(f"Template list: {template_list}")
    logger.info(f"Number of per category prompts: {len(template_list) * args.sample_num}")

    if args.sharp_focus is True:
        for idx in range(len(template_list)):
            template_list[idx] = template_list[idx] + ",highly detailed,hires,8k,sharp focus"

    expand_ratio = args.total_num_temps // len(template_list) + 1
    template_list = template_list * expand_ratio
    random.shuffle(template_list)
    template_list = template_list[: args.total_num_temps]

    nlp = keytotext_pipeline("mrm8488/t5-base-finetuned-common_gen")

    os.makedirs(args.output_dir, exist_ok=True)
    sliced_num = args.sliced_num

    accelerator = Accelerator()
    pipe = load_model(args)
    pipe = accelerator.prepare(pipe).to("cuda")

    if type(concept_list[0]) is list:
        concept_list = [c[0] for c in concept_list]

    for category_name in concept_list:
        if sampled_dict is not None:
            prompt_list = sampled_dict[category_name]
        else:
            prompt_list = []
            for template in template_list:
                if args.dataset_type == "ti_generated" or args.dataset_type == "ti_generate":
                    prompt_unique_id = f"{args.unique_id} {category_name}"
                else:
                    prompt_unique_id = category_name
                prpt = template.format(prompt_unique_id)
                if args.sentence_expansion is True:
                    prpt = nlp([prpt], num_return_sequences=1, do_sample=True)
                prompt_list.append(prpt)

        if args.neg_prompts is True:
            negative_prompt_list = []
            for c in concept_list:
                if c != category_name:
                    negative_prompt_list.append(f"{c}")

            if len(negative_prompt_list) > 10:
                negative_prompt = random.sample(negative_prompt_list, 10)
            negative_prompt = ", ".join(negative_prompt_list)
            negative_prompt += ", ((disfigured)), ((bad art)), ((deformed)), ((poorly drawn)), ((extra limbs)), ((close up)), ((b&w)), weird colors, blurry, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artefacts, signature, watermark, username, blurry, out of focus, censorship,"
        else:
            negative_prompt = None

        saved_category_dir = os.path.join(args.output_dir, process_namings(category_name))
        if os.path.exists(saved_category_dir) and args.force_regen:
            shutil.rmtree(saved_category_dir)
        os.makedirs(saved_category_dir, exist_ok=True)

        def save_image(img):
            img = img.resize((224, 224))
            img.save(os.path.join(saved_category_dir, f"{uuid.uuid4()}.png"))

        with torch.no_grad():
            images = []
            for i in range(args.total_num_temps // sliced_num):
                negative_prompt_list = [negative_prompt] * sliced_num if negative_prompt is not None else None
                images.extend(
                    pipe(
                        prompt_list[i * sliced_num : (i + 1) * sliced_num],
                        negative_prompt=negative_prompt_list,
                        num_images_per_prompt=args.sample_num,
                    ).images
                )

        logger.info(f"Saving {len(images)} images for category {category_name} to {saved_category_dir}...")
        with ThreadPoolExecutor(max_workers=24) as executor:
            executor.map(save_image, images)
