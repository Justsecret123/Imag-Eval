import pandas as pd
import os 
import json
from pathlib import Path


def extract_info(code_file, file_name: str):
    """
    Extract benchmark metadata from an IMAG-EVAL image filename.

    The function parses an image name following the IMAG-EVAL naming
    convention and retrieves the associated skill combination, difficulty
    level, prompt identifier, synthetic prompt identifier, and
    robustness setting. It also maps the extracted skill code to its
    corresponding skill-combination name using the provided code table.

    Args:
        code_file (pd.DataFrame): Mapping between skill-combination codes
            and skill names.
        file_name (str): Image filename following the IMAG-EVAL naming
            scheme.

    Returns:
        tuple:
            - str: Skill-combination name associated with the image.
            - str: Prompt difficulty level.
            - str: Prompt identifier.
            - str: Synthetic prompt identifier.
            - bool: Whether the image belongs to a robustness evaluation
              setting.
    """

    # Check if robust in string or not
    robust = "robust" in file_name
    name = file_name[:-11] if robust else file_name[:-4]
    name = name.split("_")
    # Retrive the skill code
    skill_code = name[-4]
    # Retrieve the prompt number
    prompt_number = name[-2]
    # Retrieve the synthetic prompt number
    synthetic_number = name[-1]
    # Retrieve the level
    level = name[-3]
    #print(f"Skill code : {skill_code}")
    # Retrieve the skill name
    skill_name = code_file.loc[code_file["code"]==int(skill_code), "skill"]
    # Check if the code is not empty
    if not skill_name.empty: 
        skill_name = skill_name.iloc[0]
    else: 
        skill_name = -1

    return skill_name,level,prompt_number,synthetic_number,robust


def get_file_names(model_name: str):
    """
    Retrieve the list of benchmark images associated with a specific model.

    The function scans the IMAG-EVAL image directory corresponding to the
    selected model, collects all PNG files, and returns their paths in
    sorted order. The resulting list can be used for annotation,
    evaluation, or analysis workflows.

    Args:
        model_name (str): Name of the model whose generated images should
            be retrieved.

    Returns:
        list[str]: Sorted list of image file paths associated with the
        specified model.
    """

    # Retrieving images list
    images_list = Path(f"../../data/images_refactored/{model_name}")
    images_list = [p._str for p in images_list.glob("*") if p.is_file() and ".png" in os.path.basename(p)]
    images_list.sort()

    return images_list

def get_images_info(code_file, images_list):
    """
    Extract benchmark metadata for a collection of IMAG-EVAL images.

    The function parses each image filename using the IMAG-EVAL naming
    convention and gathers the associated metadata, including the skill
    combination, difficulty level, prompt identifier, synthetic prompt
    identifier, and robustness setting.

    Args:
        code_file (pd.DataFrame): Mapping between skill-combination codes
            and skill names.
        images_list (list[str]): Collection of image file paths to
            process.

    Returns:
        dict: Mapping between image paths and their associated metadata.
        Each entry contains:

            - `skill_name`: Skill combination associated with the image.
            - `level`: Prompt difficulty level.
            - `prompt_number`: Prompt identifier.
            - `synthetic_prompt`: Synthetic prompt identifier.
            - `robust`: Whether the image belongs to a robustness
              evaluation setting.
    """

    images_info = dict()
    # Loop through the list of images
    for image in images_list: 
        # Retrieve the info
        skill_name, level, prompt_number, synthetic_prompt,robust = extract_info(code_file,image)
        # Retrieve the skill name
        images_info[image] = {
            "skill_name": skill_name,
            "level": level,
            "prompt_number": prompt_number,
            "synthetic_prompt": synthetic_prompt, 
            "robust": robust
        }
    
    return images_info

def get_prompts(images_info: dict):
    """
    Retrieve the synthetic prompts associated with a collection of
    IMAG-EVAL images.

    The function maps each image to its corresponding synthetic prompt by
    parsing the image metadata, locating the appropriate prompt file, and
    extracting the prompt variant used to generate the image. Both
    standard and robustness-evaluation prompts are supported.

    Args:
        images_info (dict): Mapping between image paths and their
            associated metadata, including skill combination, difficulty
            level, synthetic prompt identifier, and robustness setting.

    Returns:
        dict: Mapping between image paths and their corresponding
        synthetic prompts.
    """

    # Set the matches between levels and 
    level_match = {"hard": 0, "medium": 1, "easy": 2}
    # Set the matches dict
    image_matches = dict()
    # Loop through the files
    for image, image_info in images_info.items():
        # Open the corresponding file
        try:
            # Set the json file name
            json_file_name = ""
            if image_info["robust"]:
                json_file_name = f"../../outputs/prompts/{image_info['skill_name']}_robust.json" 
            else:
                json_file_name = f"../../outputs/prompts/{image_info['skill_name']}.json" 
            # Open the json file
            with open(json_file_name, "rb") as file:
                # Load data into a json object
                data = json.load(file)
                # Get the exact synthetic prompt
                level = image_info["level"]
                synthetic_prompt_position = image_info["synthetic_prompt"]
                synthetic_prompt = data["prompts"][level_match[level]]["synthetic_prompts"][int(synthetic_prompt_position)]
                # Add the match
                image_matches[image] = synthetic_prompt
        except Exception:
            # Display error message
            print(f"Error for image: {image} and skill {image_info['skill_name']}")
    
    return image_matches

def write_results(image_matches: dict, model_name: str):
    """
    Export image-to-prompt matches for a specific model.

    The function writes two aligned output files: one containing the
    generated image filenames and another containing the corresponding
    prompt identifiers. These files can be used for benchmark analysis,
    manual inspection, or downstream evaluation workflows.

    Existing output files are automatically replaced before writing the
    new results.

    Args:
        image_matches (dict): Mapping between image filenames and their
            associated prompts.
        model_name (str): Name of the model whose matches are being
            exported.

    Returns:
        None: Match files are written to the output directory.
    """

    # Set the output names 
    prompts_file = f"../../outputs/matches/{model_name}_prompts.txt"
    files_file = f"../../outputs/matches/{model_name}_files.txt"
    # Removing files if they already exist
    if os.path.exists(prompts_file):
        os.remove(prompts_file)
    if os.path.exists(files_file):
        os.remove(files_file)

    # Loop through the elements
    for file_name,prompt in image_matches.items():
        # Write the file names into a file
        with open(files_file, "a+", encoding="utf-8") as file:
            file.write(f"{os.path.basename(file_name)}\n")
        with open(prompts_file, "a+", encoding="utf-8") as file2:
        # Write into the file
            file2.write(f"{os.path.basename(prompt)}\n")
    # Display success message
    print(f"Files written for: {model_name}.")


def main():
    """
    """
    # Set the list of models
    MODELS = ["adobe_firefly",
              "animagine", 
              "dalle", 
              "flux",
              "kandinsky", 
              "qwen_image", 
              "runway_ml", 
              "stable_cascade", 
              "stable_diffusion", 
              "z_image_turbo"]
    # Read the code file
    code_file = pd.read_csv("../../skills_code.csv", encoding="utf-8")
    # Loop through the models
    for model in MODELS:
        # Get the file names
        images_list = get_file_names(model)
        # Get the images info
        images_info = get_images_info(code_file,images_list)
        # Get the matches between prompts and prompts
        image_matches = get_prompts(images_info)
        # Write results
        write_results(image_matches,model)


if __name__=="__main__":
    main()