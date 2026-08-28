#!/bin/bash

set -e

echo "Running Z-Image-Turbo..."
python -m experiments.test_scripts.test_judge_models

echo "Running Stable Diffusion XL..."
python -m experiments.test_scripts.test_judge_models \
    --annotations_file ./data/to_annotate/stable_diffusion_annotations.xlsx \
    --images_folder ./data/to_annotate/stable_diffusion

echo "Running Stable Cascade..."
python -m experiments.test_scripts.test_judge_models \
    --annotations_file ./data/to_annotate/stable_cascade_annotations.xlsx \
    --images_folder ./data/to_annotate/stable_cascade

echo "Running FLUX 1.0-dev..."
python -m experiments.test_scripts.test_judge_models \
    --annotations_file ./data/to_annotate/flux_annotations.xlsx \
    --images_folder ./data/to_annotate/flux

echo "Running DALLE-3..."
python -m experiments.test_scripts.test_judge_models \
    --annotations_file ./data/to_annotate/dalle_annotations.xlsx \
    --images_folder ./data/to_annotate/dalle

echo "Running GPT-Image-1.5..."
python -m experiments.test_scripts.test_judge_models \
    --annotations_file ./data/to_annotate/gpt-image-1.5.xlsx \
    --images_folder ./data/to_annotate/gpt-image-1.5

echo "Running Gemini-Flash-3.1-preview..."
python -m experiments.test_scripts.test_judge_models \
    --annotations_file ./data/to_annotate/gemini_flash_annotations.xlsx \
    --images_folder ./data/to_annotate/gemini_flash

echo "All jobs completed successfully."