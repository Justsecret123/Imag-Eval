import pandas as pd
import os 
import json
from pathlib import Path


def extract_info(code_file,file_name:str):
    """
    Extracts the info from the file name
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


def get_file_names(model_name:str):
    """
    """

    # Retrieving images list
    images_list = Path(f"../../data/images_refactored/{model_name}")
    images_list = [p._str for p in images_list.glob("*") if p.is_file() and ".png" in os.path.basename(p)]
    images_list.sort()

    return images_list

def get_images_info(code_file,images_list):
    """
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

def get_prompts(images_info:dict):
    """
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

def write_results(image_matches:dict,model_name:str):
    """
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