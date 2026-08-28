![Static Badge](https://img.shields.io/badge/Python-3.11.4-brightgreen?style=for-the-badge&logo=Python) ![Static Badge](https://img.shields.io/badge/PyTorch-2.10.0%2Bcu128-brightgreen?style=for-the-badge&logo=Pytorch)



# Imag-Eval [EMNLP 2026]

Official repository for the paper "[Imag-Eval A language-grounded framework for interpretable Text-to-Image instruction following evaluation](link_to_the_paper.com)" (link to come soon). (MOHAMED SEROUIS et al., EMNLP 2026)

**TL;DR:** Imag-Eval is a skill-based evaluation framework for Text-to-Image (T2I) models that measures instruction-following capabilities across compositional visual reasoning skills, while explicitly controlling for prompt complexity and minimizing error propagation during evaluation; we are rying to shift evaluation paradigms towards more controlled increases in complexity.

<details>
 <summary>
 <b>Drop-down to show the current leaderboard (Last updated 17/08/2026). </b>
 </summary>
 
## 🏆 Image Generation Leaderboard (all skills)

Current evaluation settings: 
```json
  {
    "test": "all skills", # Color + Counting + Size + Spatial + Emotion + Text + Cohesiveness,
    "image_width": 1024, 
    "image_height": 1024,
    "annotation_type": "manual",
    "seed": 42,
    "guidance_scale": inferred from the technical paper/model documentation,
    "inference_steps": inferred from the technical paper/model documentation,
  }
```
<table>
<thead>
<tr style="background:#eef2ff;">
<th>Rank</th>
<th>Model</th>
<th>Counting</th>
<th>Spatial</th>
<th>Size</th>
<th>Emotion</th>
<th>Color</th>
<th>Cohesiveness</th>
<th>Text (WER ↓)</th>
</tr>
</thead>

<tbody>

<tr style="background:rgba(255,215,0,0.15);">
<td>🥇</td>
<td><strong>Gemini-Flash-3.1-preview</strong></td>
<td><b>0.70</b></td>
<td>0.73</td>
<td><b>0.85</b></td>
<td><b>0.98</b></td>
<td>0.96</td>
<td>0.68</td>
<td><b>0.29</b></td>
</tr>

<tr style="background:rgba(192,192,192,0.15);">
<td>🥈</td>
<td><strong>Z-Image-Turbo</strong></td>
<td>0.55</td>
<td><b>0.79</b></td>
<td>0.84</td>
<td>0.80</td>
<td><b>0.98</b></td>
<td>0.50</td>
<td>0.49</td>
</tr>

<tr style="background:rgba(205,127,50,0.15);">
<td>🥉</td>
<td><strong>FLUX 1.0</strong></td>
<td>0.56</td>
<td>0.69</td>
<td>0.62</td>
<td>0.67</td>
<td>0.88</td>
<td>0.76</td>
<td>2.46</td>
</tr>

<tr>
<td>4</td>
<td>DALL·E 3</td>
<td>0.42</td>
<td>0.59</td>
<td>0.70</td>
<td>0.28</td>
<td>0.69</td>
<td><b>0.99</b></td>
<td>4.25</td>
</tr>

<tr>
<td>5</td>
<td>Stable Cascade</td>
<td>0.13</td>
<td>&lt;0.01</td>
<td>0.20</td>
<td>0.10</td>
<td>0.36</td>
<td>0.35</td>
<td>4.31</td>
</tr>

<tr>
<td>6</td>
<td>Stable Diffusion XL</td>
<td>0.17</td>
<td>0.34</td>
<td>0.15</td>
<td>0.09</td>
<td>0.16</td>
<td>0.26</td>
<td>5.55</td>
</tr>

</tbody>
</table>

> *Comprehensive experimental results are available in the paper.*
---
 <br>
</details>

> To ensure leaderboard integrity and reproducibility, submissions must include the generated images, generation seed, the method used for annotation, and all relevant inference parameters. Reported results will be independently verified, and entries whose reproduced results closely match the submitted scores will be added to the leaderboard.

## 🚀 Recent News
- **[Aug 2026]** 🎉 IMAG-EVAL has been accepted to **EMNLP 2026**.
- **[Aug 2026]** 📊 Released the first version of the IMAG-EVAL benchmark, comprising **1,140 prompts**, **8,842 evaluation rules**, and **7 evaluation dimensions**: Counting, Spatial Relations, Size Relations, Color Attribution, Emotion Attribution, Text Rendering, and Cohesiveness.
- **[Future]** 🌱 Planned extensions include backend support for MPS and AMD, improved extraction pipelines, additional evaluation skills, and expanded leaderboard support.

## Motivation

Existing Text-to-Image benchmarks often conflate prompt length, linguistic complexity, and instruction-following difficulty. As a result, it is frequently unclear whether a model fails because of compositional reasoning limitations or simply because it struggles with longer prompts. Imag-Eval addresses this limitation through a skill-based evaluation framework that tries to disentangle surface-level linguistic complexity from compositional instruction complexity, which we refer to as **compositional load**.

### Core Principles

**1. Controlled prompt complexity:** Instead of comparing arbitrary short and long prompts, additional prompt length is introduced primarily through task-relevant constraints associated with the evaluated skills.
**2. Multi-factorial compositional difficulty scaling:** We control compositional difficulty by providing tests with various combinations of skills (e.g, color+size vs color+size+emotion+spatial), or by increasing both the number of rules and instances to generate for the same test (e.g, 3 spatial relationships + 3 emotion rules and 3 humans for Hard, but 2 spatial relationships + 2 emotions rules + 1 human for Easy).
**3. Minimal error propagation:** Evaluation is performed at the rule level whenever possible. For instance, the absence of a generated object does not automatically invalidate unrelated constraints associated with that object, enabling to better pinpoint *where* the model failed.

Our experiments, together with an additional analysis of more than 2,000 prompts collected from prior literature, suggest that instruction-following difficulty is primarily driven by the number of grounded constraints and instance bindings rather than prompt length alone.

## Dataset Statistics

Imag-Eval consists of **3,778 image-generation JSON evaluation rules** and **186 text-rendering rules**, covering a diverse set of instruction-following skills.

| Skill | JSON rules (meta-prompts)  | Total combined rules (synthetic prompts)  |
|----------------------|------------------------------|------------------------------|
| Counting  | 2,228 | 2,228 |
| Color | 558 | 1,660 |
| Spatial | 310 | 2,199  |
| Emotion | 372 | 372 |
| Size  | 310 | 2,197 |
| Text  | 186 | 186 |
| **Total**  |  **3,964**  |  **8,842** |

> **Note:** A detailed prompt analysis is available in [evaluation/prompts_analysis.ipynb](./evaluation/prompts_analysis.ipynb). There is also a notebook in [evaluation/evaluate_results_merged_all.ipynb](./evaluation/evaluate_results_merged_all.ipynb) that showcases how we use annotation results in a .csv or .xlsx format to compute relevant metrics. 

### Experimental Environment

- **Python version:** 3.11.4
- **GPU:** NVIDIA RTX 6000 Ada (48 GB VRAM), NVIDIA RTX 5000 Ada * 2 (32 GB VRAM * 2). 
- **CPU:** Intel Xeon w5-3435X (4.70 Ghz) CPU featuring 32 threads
- **RAM:** 252 GB

The configuration above reflects the environment used during experimentation and should not be interpreted as a minimum system requirement. Hardware requirements vary significantly across models and are typically documented in their respective technical reports, Hugging Face model cards, or ComfyUI repositories.

Most experiments only require a single GPU. Multi-GPU execution was primarily used for larger models such as full-scale Qwen-Image variants.

#### Specific environment variables

Create a `.env` file in the project root and replace placeholder values with your own setup, depending on the experiments that you want to run:

```bash
CUDA_LAUNCH_BLOCKING=1
TORCH_USE_CUDA_DSA=1
PYTORCH_ALLOC_CONF=expandable_segments:True
AZURE_OPENAI_ENDPOINT=[our API endpoint]
AZURE_OPENAI_API_KEY=[our API key]
DALLE_ENDPOINT=[our API endpoint]
DALLE_KEY=[our API key]
GPT_IMAGE_ENDPOINT=[our API endpoint]
GPT_IMAGE_KEY=[our API key]
OPENAI_API_VERSION=[2023-05-15]
HF_TOKEN=[our Hugging-Face token]
MOUNTED_COMFY_UI_PATH=[our ComfyUI path]
QWEN_VL_FP8_PATH=[our Qwen3-VL-30B-A3B-Instruct-FP8 path]
```

> **Important note:** The aforementioned configuration is not a minimum requirement for a specific model. Model requirements are ususally listed on HuggingFace, the technical paper or ComfyUI. Moreoever, most experiments only made use of one of the GPUs, except for larger models such as the full QWEN-Image.
 


## Quick Start 

### Prompts: location and structure

All prompts are available at [outputs/prompts](./outputs/prompts/), as JSON files for each set of tests. Prompt files follow the structure below:

```json
  {
    "experiment_id": [names of the skills to evaluate], 
    "timestamp": [start time of the generation], 
    "prompts": [
      {
        "id": "prompt_[skill_code]_[level]_[names of the skill to evaluate]" # Example: prompt_001_hard_color+size, 
        "skills": [list of skills], 
        "level": [level] # Same structure for each level in the defined levels (default = easy, medium and hard)
        "meta-prompt": [text of the meta-prompt (template) used to generate final prompts],
        "scene": {
          "emotion": [ # IF emotion rules are defined
            "emotion_1", 
            "..."
          ],
          "objects": [ # IF counting rules are defined
            {
              "object": [object_name], 
              "count": [object_count], # If a count is not specified, then there is only one instance of the object to generate
              "color" : [object_color] # Only if a color has been specified
            }
          ], 
          "spatial_relations": [ # IF spatial rules are defined
            [
              "object_A", 
              "to the right of", # Example of spatial rule
              "object_B"
            ],
            ...,
          ], 
          "size_relations": [ # IF size rules are defined
            [
              "object_B", 
              "smaller than", # Example of size rule
              "object_A"
            ], 
            ...,
          ], 
          "text": { # IF text rules are defined
            "word_count": [word_count], 
            "objects": [
              "object_A", 
              "object_B"
              "...",
            ]
          }
        }, 
        "synthetic_prompts": [
          "Prompt 1 generated from the above elements", 
          "...."
        ]
      }, 
      {....Same structure for the other levels....}
    ]
  }
```

> **Jump to [test_scripts](./experiments/test_scripts/) README for the most interesting part.**


### How to (re)generate the synthetic prompts

```bash
python main.py generate_all 
```

This will generate structured prompts based on all the configuration files in `config/prompt_generation` and save them to `outputs/prompts/` as json files.

You can also generate prompts for a specific configuration file, using:

```bash
python main.py generate \
--config config/prompt_generation/[name of the configuration file]
```

However, the original prompts are already available at `outputs/prompts/`.

> **Reproducibility Note:** Proprietary LLM APIs may evolve over time, which can introduce minor variations in prompt generation. Nevertheless, our experiments indicate that the relative differences in compositional load are generally preserved across such variations.


### How to run the evaluator 

> Before running the evaluator, please read the documentation available in [this Readme](./experiments/test_scripts/) first. 

```bash
python -m experiments.test_scripts.test_judge_models
```


## Configuration

### Prompt generation configuration

All prompt generation configuration files can be found in `config/prompt_generation/`, as .yaml files. However, you can tune them to vary the exact skills for specific levels, modify the generator, and so on. An example of prompt generation configuration file is available below:

```yaml
experiment:
  name: color+emotion_robust # Name of the experiment
prompt_generation:
  metadata_config:
    name: coco # Dataset where to sample objects from
  skills: # This is where to define various levels. It can also be used to create or declare an intermediate or custom level
  - level: hard 
    skills: [color, emotion] # Skills to evalue
    robustness: [typos, consistency] # Types of robustness tests
    k: 5 # Max number of synethtic prompts to generate for each meta-prompt
  - level: medium
    skills: [color, emotion]
    robustness: [typos, consistency]
    k: 3
  - level: easy
    skills: [color, emotion]
    robustness: [typos, consistency]
    k: 2
  llm_model: # Configuring the generator
    name: openai # API name
    model: gpt-5-mini # Name of the model
    temperature: 1 # Generation temperature
  samples_per_skill: 1 # Max number of meta-prompts to generate for each setup
output:
  file: outputs/prompts/color+emotion_robust.json
```

## Project Structure

```
root
├── config/ # Configuration files 
│   ├── evaluation/
│   └── prompt_evaluation/
├── data/
│   └── COCO/
├── evaluation/
├── experiments/
│   ├── test_scripts/
│   └── experiment_runner.py
├── llm_interfaces/
├── metadata/
├── prompt_generation/
├── utils/
│   └── skills.py
├── .env
├── main.py
└── requirements.txt
└── ...
```

## Main Dependencies

- **PyTorch + Torchvision**: Deep learning framework.
- **Ultralytics/YOLO:** Official framework for YOLO models.
- **OpenAI**: GPT API access for LLM evaluation. Used for generating prompts, although you can easily tailor the script to fit your own model. 
- **OpenCV**: Some image processing. 
- **COCO Tools**: COCO dataset utilities. 
- **NLTK**: for processing/extracting details about some prompts. 

## Contributing

We welcome research and engineering contributions that improve the benchmark, evaluation pipeline, or reproducibility of reported results. Areas of particular interest include: 

- **Leaderboard** 
  - Interactive filtering by skill combinations, difficulty levels, evaluation settings, and models. 
- **Additional benchmark results** 
  - Evaluations of new Text-to-Image models, alternative seeds, and additional experimental settings. 
- **Automated evaluation improvements** 
  - More reliable Cohesiveness and Emotion assessment models. 
  - Improved alignment with human annotations. 
- **Benchmark extensibility** 
  - Modular support for adding or removing evaluation skills. 
- **Robustness testing** 
  - Fine-grained control over perturbation placement, intensity, and frequency. 
- **Engineering improvements** 
  - Bug fixes, performance optimization
  - MPS or AMD support
  - Additional model integrations
  - Documentation enhancements

This project has been built iteratively, brick by brick, and has been maintained primarily by a single contributor. As a result, occasional mistakes, inconsistencies, or suboptimal implementations may still be present. Contributions, bug reports, and suggestions are always welcome, and all pull requests will be reviewed with great care and attention.

### Corresponding author

**Ibrahim MOHAMED SEROUIS, PhD.**
Multimodal AI Researcher
- **Email:** ibrahim.mohamed-serouis@talan.com. 
- **LinkedIn:** https://www.linkedin.com/in/ibrahim-serouis/

## License

The dataset (prompt collections in JSON format), generation scripts, and annotation guidelines are released under the **CC BY-NC 4.0 License** (Creative Commons Attribution-NonCommercial).

These resources are intended exclusively for research and non-commercial R&D purposes.

## Citation

If you use our code or benchmark in your research, please cite:

```bibtex
[Citation text coming soon]
```

## Use of AI Assistants

AI assistants were used to help draft and rephrase portions of this README, as well as the docstrings of a small number of functions. All technical decisions, implementations, experiments, and evaluations were designed and reviewed by the author.

While every effort has been made to verify the accuracy of the generated text, occasional inconsistencies or inaccuracies may remain. If you encounter any such issues, please feel free to open an issue or contact me directly.