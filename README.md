# Benchmarking and Analyzing Generative Data for Visual Recognition

## Motivation and TLDR
Machine learning (ML) community has successfully utilized datasets, though their creation often demands substantial time and resources. Despite this, there's a lack of efficient, open data engineering tools to streamline these processes, leading to increased costs. This prompts an exploration into the potential benefits of generative data, a topic investigated in this paper.

With the repo, you can easily:

(1) sample images from generative models;

(2) pack images for various evaluations;

(3) run CLER score/Elevater evaluations;

(4) analyze images with other metrics (FID, CLIP score, proxy-A distance etc.);

## Overview

Large pre-trained generative models have made remarkable progress in generating realistic and diverse images. This ability opens up the possibility of utilizing large pre-trained generative models as efficient data generators to enhance visual recognition. In this work, we aim to rigorously benchmark and analyze the effect of generative images, with an emphasis on comparative studies among different paradigms leveraging external data (i.e. generative vs. retrieval vs. original). Our contributions include:

**(1) Benchmarking with wide coverage:** We construct GenBench, a comprehensive benchmark consisting of 22 datasets with 2548 categories to cover a diverse range of visual recognition tasks to evaluate the benefits of generative data.

**(2) New metric tailored for recognition:** Existing metrics in generative models (\eg, FID and CLIP score) have no strong correlation for the downstream recognition performance. Here we propose a training-free metric, CLER score, to efficiently indicate the effectiveness of generative data for recognition performance before actually training on the downstream tasks.

**(3) New reference baselines:** Leveraging external training data includes retrieval-based methods. We compare generative data with retrieved data from the same external pool to highlight the unique characteristics of generative data.

**(4) Investigating external knowledge injection in generative models:** We fine-tune special token embeddings for each category in a dataset by injecting retrieval and original data via Textual Inversion. This approach leads to improved performance across 17 datasets, with a notable exception in the case of low-resolution reference images.

Overall, our comprehensive benchmark, in-depth analysis, and proposed techniques highlight the opportunities of generative data for visual recognition, while also identifying critical challenges for future research.

<style>
.centered {
    display: flex;
    justify-content: center;
    align-items: center;
}
</style>

<div style="background-color: white;">
    <figure>
        <div class="has-text-centered">
        <img src="https://i.postimg.cc/NFXk9M8H/rel-improvement.png" alt="Image description" width="100%">
        </div>
        <figcaption style="font-size: 1.2em; color: black;"><strong>Table: Left:</strong> CLIP ViT-B/32 linear probing results for all datasets on GenBench, arranged in descending order of improvement over the zero-shot accuracy. The results are based on 500-shot generative data per category with best strategy for each dataset. <br><strong>Right:</strong> The average results using different external data sources on the 22 datasets, along with sample images for different categories, are shown on the right.</figcaption>
    </figure>
</div>

<div style="background-color: white;">
    <figure>
        <div class="centered">
            <img src="https://i.postimg.cc/jdVXVHKQ/metric-correlation.png" alt="Image description" width="75%">
        </div>
        <figcaption style="font-size: 1.2em; color: black;"><strong>Figure:</strong> Correlation between three evaluation metrics, CLER score, CLIP score, and FID, with the linear probing accuracy.</figcaption>
    </figure>
</div>

<div style="background-color: white;">
    <figure>
        <div class="centered">
            <img src="https://i.postimg.cc/3RsgThMJ/appendix-examples.png" alt="Image description" width="75%">
        </div>
        <figure class="centered">
            <figcaption style="font-size: 1.2em;color: black;"><strong>Figure:</strong> Generative images with different prompt strategies.</figcaption>
        </figure>
    </figure>
</div>

<div style="background-color: white;">
    <figure>
        <div class="centered">
            <img src="https://i.postimg.cc/mrr33VNR/ti-teaser.png" alt="Image description" width="75%">
        </div>
        <figure class="centered">
            <figcaption style="font-size: 1.2em;color: black;"><strong>Figure:</strong>Comparison of the visualization results between direct sampling from Stable Diffusion and finetuned after Textual Inversion on original and retrieval data.</figcaption>
        </figure>
    </figure>
</div>

## Prerequisites

- Python 3.7+
- Loguru
- Wandb
- Vision_Benchmark (https://github.com/Computer-Vision-in-the-Wild/Vision_Benchmark)

## Usage

> This project provides a flexible pipeline for sampling images from generative models, packing images for evaluation, and running CLER score evaluations along with various other evaluation methods. The script `pipeline.py` guides the entire process. 

You can run the `pipeline.py` with different options. Here are the options and their usage:

1. **configure dataset meta info**: `--option configure` Use this option to setup the dataset meta information.

2. **prepare retrieval/original images for textual inversion reference**: `--option textual_inversion_prepare` This will prepare retrieval or original images for textual inversion reference.

3. **sampling images from generative models (stable diffusion, or with injected token embeddings)**: `--option sample_images` This option will sample images from the generative models. You can choose to sample images from stable diffusion or with injected token embeddings.

4. **pack generated/retrieval images for evaluation**: `--option pack_generated` or `--option pack_retrieval` Use these options to pack the generated or retrieval images for evaluation.

5. **run evaluation with Elevater**: `--option elevator` This will run the evaluation with Elevater toolkit.

6. **run zeroshot evaluation with CLER Score and FID metric**: `--option zeroshot_metric` This will run a zeroshot evaluation using CLER Score and FID metric.

7. **calculate proxy_a_instance**: `--option proxy_distance` This will calculate the proxy_a_instance.

Please note that for most of these options, additional arguments may be needed. Refer to the `pipeline.py` script for the full list of arguments.

## Example

The above commands can also be combined to form a pipeline execution.

Here is an example command to run the `pipeline.py` script:

```bash
python pipeline.py --option configure,sample_images,pack_generated,elevator,zeroshot_metric,proxy_distance --dataset_name hateful-memes --language_enc lang_enc --output_dir ./work_dirs
```

This command will configure the dataset meta info, sample images, pack the generated images, run evaluation with Elevater, run zeroshot evaluation, and calculate the proxy_a_instance. The dataset used is `hateful-memes` and the generation model is `stable-diffusion-2.1` in default. The output directory is set to `./work_dirs`.

## Contact

If you have any questions or issues, please open an issue in this repository or contact `drluodian@gmail.com` for help.