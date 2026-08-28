## Key files

### Sample Data

Sample images generated with Z-Image-Turbo [(HuggingFace link)](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) are available under [data/to_annotate/z_image_turbo/](../../data/to_annotate/z_image_turbo/). An example of annotation file is provided at [data/to_annotate/annotations_example.xlsx](../../data/to_annotate/annotations_example.xlsx).

### Scripts

#### Image generator: [test_gen_images.py](./test_gen_images.py)

This script contains the image generation pipelines used throughout our experiments. Different models rely on different execution environments and providers:

- **HuggingFace/Diffusers dependent models:** Stable Diffusion XL, Z-Image-Turbo, Stable Cascade, Animagine 
- **Stable-Diffusion-cpp dependent models:** Qwen-Image, FLUX 1.0
- **API-based models:** DallE-3, GPT-Image-1.5. 

##### Arguments
```bash
Test script for generating images. Default or (inferred) best paramaters for each model are used for each generator.

options:
  -h, --help            show this help message and exit
  --gpu {0,1,2}         Preferred GPU number.
  --model {all,firelfy,dalle,kandinsky,runway,stable_cascade,stable_diffusion,gpt-image,z_image,qwen,flux}
                        Generator to use.
  --seed SEED           The generator seed.
```

##### Important notes 
1. We use [SD-Embed](https://github.com/xhinker/sd_embed) to support for longer prompts for specific models. 
2. For SD-Embed-based models and models relying on Stable-Diffusion-cpp, model weights must be downloaded locally. The corresponding paths defined in the script should then be updated accordingly.



#### Automated evaluator: [test_judge_models.py](./test_judge_models.py)

This script implements the automated evaluation pipeline used in our experiments.

Contributions are welcome, particularly regarding prompt engineering improvements and more robust extraction methodologies.

> **Hardware requirements:** Peak VRAM consumption during automated annotation was approximately 48 GiB.

From the root of the repository: ```python -m experiments.test_scripts.test_judge_models```

```bash

options:
  -h, --help            show this help message and exit
  --model {all,firelfy,dalle,kandinsky,runway_ml,stable_cascade,stable_diffusion,z_image_turbo,qwen_image,flux}
  --device {cuda,cpu}
  --seed SEED           The generator seed.
  --object_detection {yolo26n,yolo26s,yolo26m,yolo26l,yolo26x,detectron}
                        The method used for depth estimation.
  --depth_estimation {yolo26n,yolo26s,yolo26m,yolo26l,yolo26x,detectron}
                        The method used for object detection.
  --text_extraction {paddleocr,qwen-vl}
                        The method used for text extraction.
  --emotion_extraction {small,large}
                        The VLM model used for emotion extraction.
  --annotations_file ANNOTATIONS_FILE
                        Path to the file to annotate.
  --images_folder IMAGES_FOLDER
                        The folder that contains images to annotate.
  --verbose {0,1}       The verbosity controls if the extraction results are shown or not.
```

##### Current supported models and methods for extraction

| Skill        | Method                                         | Models                                                                                    |
|--------------|------------------------------------------------|-------------------------------------------------------------------------------------------|
| Text         | VLM inference (HuggingFace)                    |  Qwen/Qwen3-VL-8B-Instruct (https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)             |
| Counting     | Object detection                               | YOLO26 models (https://docs.ultralytics.com/tasks/detect)                                 |
| Size         | Instance segmentation                          | YOLO26-seg models (https://docs.ultralytics.com/tasks/segment)                            |
| Spatial      | Object detection + Monocular depth estimation  | YOLO26-seg & YOLO26-depth models (https://docs.ultralytics.com/tasks/depth)                 |
| Emotion      | VLM inference (Sglang)                         | Qwen3-VL-30B-A3B-Instruct-FP8 (https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8) |
| Color        | Instance segmentation + Object isolation + VLM inference (Sglang) | YOLO26 & Qwen3-VL-30B-A3B-Instruct-FP8                                                  |
| Cohesiveness | VLM inference (Sglang)                         | Qwen3-VL-30B-A3B-Instruct-FP8                                                             |

> If you download the Qwen3-VL-30B-A3B-Instruct-FP8 from hugging face, ensure that the following files are present in the local Qwen directory: preprocessor_config, vocab.json, tokenizer_config, tokeniser, video_processor_config, chat_template, in the qwen directory.

Based on our experiments, the automated evaluation pipeline achieves performance comparable to human annotation for *Counting, Spatial, Relationships, Size, Emotion,* and *Text* skills. Performance is less reliable for *Cohesiveness*, for which human evaluation remains the recommended approach. Nevertheless, contributions aimed at improving automated assessment are highly encouraged.

> In case you use your own .xlsx files for automated evaluation, your file name should follow the convention: ```[model_name]_[skill_code]_[level]_[prompt_number]_[robust or nothing].png``` and follow the same columns as in the example above.



#### Important notes 

For the automated evaluator, we used this specific configuration GPU-wise:

```bash
Cuda compilation tools, release 12.4, V12.4.99
Build cuda_12.4.r12.4/compiler.33961263_0
NVIDIA-SMI 580.173.02
CUDA Version: 13.0 
```
If you identify additional compatible configurations that do not introduce workflow regressions, feedback and contributions are welcome.

### Notebooks 

> **Note:** Please download these .csv and .pkl assets and place them in the locations expected by the notebooks before running the examples provided in the repository.


#### .yaml files generator for various-lengths skill combinations: [test_yaml_generation.ipynb](./test_yaml_generation.ipynb)

Demonstrates how YAML configuration files are automatically generated for all valid skill combinations across multiple complexity levels.

If a skill is added or modified, rerun the entire notebook to regenerate the corresponding configuration files.

#### Skill codes generation: [test_codes_generation.ipynb](./test_codes_generation.ipynb)

Demonstrates the generation of skill identifiers from combinations of skills defined within a prompt configuration file.
