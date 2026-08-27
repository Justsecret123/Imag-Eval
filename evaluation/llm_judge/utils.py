
import json
import math
import os
from .codes_info import SKILLS, MODELS
from dotenv import load_dotenv
import numpy as np
import pandas as pd

def process_annotations_file(annotations_file:pd.DataFrame):
    """_summary_

    Args:
        annotations_file (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """

    # Set the columns as strings
    annotations_file["Counting"] = annotations_file["Counting"].astype("string")
    annotations_file["Colors"] = annotations_file["Colors"].astype("string")
    annotations_file["Spatial relationships"] = annotations_file["Spatial relationships"].astype("string")
    annotations_file["Size"] = annotations_file["Size"].astype("string")
    annotations_file["Emotion"] = annotations_file["Emotion"].astype("string")
    annotations_file["Text"] = annotations_file["Text"].astype("string")
    annotations_file["Cohesiveness"] = annotations_file["Cohesiveness"].astype("string")


    return annotations_file

def load_logs(logs_path:str):

    # Initialize the logs
    json_logs = dict()
    
    # Check if the file doesn't exist
    if not os.path.exists(logs_path): 
        # Set the logs
        for skill in SKILLS:
             json_logs[skill] = "KO"
        # Write the logs to a file
        with open(logs_path, "w+") as file:
            json.dump(json_logs,file)
    # If the file aready exists
    else:
        # Load the object
        with open(logs_path, "r", encoding="utf-8") as file:
            json_logs = json.load(file)

    return json_logs

def load_YOLO_CLASSES(path:str):
    """_summary_

    Args:
        path (str): _description_

    Returns:
        _type_: _description_
    """    
    # Initialize the yolo dict ({codes:class_names})
    YOLO_CLASSES = dict()
    # Initialize the inverted YOLO ({class_names:codes})
    INVERTED_YOLO = dict()
    # Load the data
    with open(path) as file:
        YOLO_CLASSES = json.load(file)
        YOLO_CLASSES = YOLO_CLASSES["class"]
    # Set the inverted yolo dict
    for key, value in YOLO_CLASSES.items():
        INVERTED_YOLO[value] = key

    return YOLO_CLASSES, INVERTED_YOLO


def get_level(name:str):
    """_summary_

    Args:
        name (str): _description_

    Returns:
        _type_: _description_
    """

    # Initialize the variable
    image_name = name
    # Retrieve models name
    for model in MODELS:
        if model in image_name:
            # Remove the model name string
            image_name = image_name.replace(model+"_","")
            # Remove the extension
            image_name = image_name.replace(".png","")
            break
    # Extract the level
    level = image_name.split("_")[1]
    
    return level


def get_skill_code(name:str):
    """_summary_

    Args:
        name (str): _description_

    Returns:
        _type_: _description_
    """

    # Initialize the variable
    image_name = name
    # Retrieve models name
    for model in MODELS:
        if model in image_name:
            # Remove the model name string
            image_name = image_name.replace(model+"_","")
            # Remove the extension
            image_name = image_name.replace(".png","")
            break
    # Extract the skill code
    skill_code = image_name.split("_")[0]
    
    return skill_code


def extract_robustness(image_name:str):
    """_summary_

    Args:
        image_name (str): _description_

    Returns:
        _type_: _description_
    """

    return "robust" in image_name

def load_environment() -> str:
    """
    Loads the environment from the .env file in os system variables.
    
    return: (str) Access token. 
    
    """

    # Load the environment files
    load_dotenv(".env")
    # Load the access token
    access_token = os.environ["HF_TOKEN"] if os.environ["HF_TOKEN"] else False
    # Load the QWEN_VL_FP8 path
    qwen_path = os.environ["QWEN_VL_FP8_PATH"] if os.environ["QWEN_VL_FP8_PATH"] else False

    return access_token, qwen_path

def get_skill_code(name:str):
    """_summary_

    Args:
        name (str): _description_

    Returns:
        _type_: _description_
    """

    # Initialize the variable
    image_name = name
    # Retrieve models name
    for model in MODELS:
        if model in image_name:
            # Remove the model name string
            image_name = image_name.replace(model+"_","")
            # Remove the extension
            image_name = image_name.replace(".png","")
            break
    # Extract the skill code
    skill_code = image_name.split("_")[0]
    
    return skill_code


def extract_prompt_file(row, code_file):
    """_summary_

    Args:
        row (_type_): _description_
    """
    # Extract the skill code
    skill_code = int(get_skill_code(row['Image']))
    # Get the skill name
    skill_name = list(code_file[code_file["code"]==skill_code]["skill"])[0]
    # Extract the level
    level = get_level(row['Image'])
    # Extract the meta prompt position
    position = 0
    if level=="medium":
        position = 1
    elif level=="easy": 
        position = 2
    # Extract the robustness
    robust = extract_robustness(row['Image'])
    # Initialize the synthetic prompt number
    synthetic_number = -1
    # Extract the prompt number
    image = row['Image'].split("_")
    # Initialize the prompt file name
    prompt_file = ""
    # Extract the synthetic prompt number
    if robust:
        synthetic_number = image[-2]
        prompt_file = f"./outputs/prompts/{skill_name}_robust.json"
    else:
        synthetic_number = image[-1]
        prompt_file = f"./outputs/prompts/{skill_name}.json"

    return prompt_file, position

def extract_counting_rules(prompt_file:str, position:int):
    """_summary_

    Args:
        prompt_file (str): _description_
        position (int): _description_

    Returns:
        _type_: _description_
    """

    # Open the associated prompt file
    with open(prompt_file, "rb") as file:
        # Load the data
        prompt_data = json.load(file)
        # Initialize the classes 
        objects_to_generate = dict()
        # Get the classes to detect
        objects = prompt_data["prompts"][position]["scene"]["objects"]
        # Check if there is an emotion
        emotion = "emotion" in prompt_data["prompts"][position]["skills"]
        emotion_count = 0
        # Loop through the objects
        for object in objects:
            # Check if it's a single instance or not
            if "count" in object:
                # Get the object name and count
                objects_to_generate[object["object"]] = object["count"]
            else:
                # Set the object name and count
                objects_to_generate[object["object"]] = 1
            # Check if there is an emotion rule     
            if emotion: 
                # Check the number of people to generate
                emotion_count = len(prompt_data["prompts"][position]["scene"]["emotion"])
                # Append to the objects to generate
                objects_to_generate["person"] = emotion_count

    return objects_to_generate


def extract_size_rules(prompt_file:str, position:int):
    """_summary_

    Args:
        prompt_file (str): _description_
        position (int): _description_

    Returns:
        _type_: _description_
    """

    # Initialize the size relations
    size_relations = list()
    # Initialize the list of objects
    objects = set()

    # Open the associated prompt file
    with open(prompt_file, "rb") as file:
        # Load the data
        prompt_data = json.load(file)
        # Get the sizes relationships
        size_relations = prompt_data["prompts"][position]["scene"]["size_relations"]
        # Extract the objects to evaluate
        for object_A, relationship, object_B in size_relations:
            # Append the objects to the set of objects
            objects.add(object_A)
            objects.add(object_B)


    return size_relations, list(objects)

def extract_spatial_rules(prompt_file:str, position:int):
    """_summary_

    Args:
        prompt_file (str): _description_
        position (int): _description_

    Returns:
        _type_: _description_
    """

    # Initialize the spatial relations
    spatial_relations = list()
    # Initialize the list of objects
    objects = set()

    # Open the associated prompt file
    with open(prompt_file, "rb") as file:
        # Load the data
        prompt_data = json.load(file)
        # Get the spatial relationships
        spatial_relations = prompt_data["prompts"][position]["scene"]["spatial_relations"]
        # Extract the objects to evaluate
        for object_A, relationship, object_B in spatial_relations:
            # Append the objects to the set of objects
            objects.add(object_A)
            objects.add(object_B)


    return spatial_relations, list(objects)

def extract_emotion_rules(prompt_file:str, position:int):
    """_summary_

    Args:
        prompt_file (str): _description_
        position (int): _description_

    Returns:
        _type_: _description_
    """

    # Open the associated prompt file
    with open(prompt_file, "rb") as file:
        # Load the data
        prompt_data = json.load(file)
        # Get the emotions rules
        emotions = prompt_data["prompts"][position]["scene"]["emotion"]

    return emotions



def process_size_relations(areas_A:list, areas_B:list)->list:
    """_summary_

    Args:
        areas_A (list): _description_
        areas_B (list): _description_

    Returns:
        list: _description_
    """

    # Initialize the list of relationships
    relations = list()

    # Loop through the areas
    for area_A in areas_A:
        for area_B in areas_B:
            if area_A > area_B:
                relations.append("larger")
            if area_A < area_A:
                relations.append("smaller")
            if area_A==area_B:
                relations.append("equal")

    return relations

def extract_color_rules(prompt_file:str, position:int):
    """_summary_

    Args:
        prompt_file (str): _description_
        position (int): _description_

    Returns:
        _type_: _description_
    """

    # Open the associated prompt file
    with open(prompt_file, "rb") as file:
        # Load the data
        prompt_data = json.load(file)
        # Initialize the classes 
        colors = dict()
        # Initialize the list of objects
        objects_list = set()
        # Get the classes to detect
        objects = prompt_data["prompts"][position]["scene"]["objects"]
        # Loop through the objects
        for object in objects:
            # Check if there is a color rule
            if "color" in object:
                # Set the object color in the dict
                colors[object["object"]] = object["color"]
                # Append the object name
                objects_list.add(object["object"])


    return colors, objects_list


def compute_inverse_size_rule(rule:str):
    """_summary_

    Args:
        rule (str): _description_
    """
    # Larger ---> smaller
    if rule=="larger":
        return "smaller"
    # Smaller ---->  larger
    if rule=="smaller":
        return "larger"

    return rule


def get_box_info(box):
    """_summary_

    Args:
        box (_type_): _description_

    Returns:
        _type_: _description_
    """

    x1, y1, x2, y2 = map(float, box)

    return {
        # Top-left corner
        "x1": x1,
        "y1": y1,
        # Bottom-right corner
        "x2": x2,
        "y2": y2,
        # Centers
        "cx": (x1 + x2) / 2,
        "cy": (y1 + y2) / 2,
        # Width, height
        "w": x2 - x1,
        "h": y2 - y1
        }

def is_left_of(boxA, boxB):
    """_summary_

    Args:
        boxA (_type_): _description_
        boxB (_type_): _description_

    Returns:
        _type_: _description_
    """
    return boxA["cx"] < boxB["cx"]

def is_right_of(boxA, boxB):
    """_summary_

    Args:
        boxA (_type_): _description_
        boxB (_type_): _description_

    Returns:
        _type_: _description_
    """
    return boxA["cx"] > boxB["cx"]

def is_above(boxA, boxB):
    """_summary_

    Args:
        boxA (_type_): _description_
        boxB (_type_): _description_

    Returns:
        _type_: _description_
    """
    return boxA["cy"] < boxB["cy"]

def is_below(boxA, boxB):
    """_summary_

    Args:
        boxA (_type_): _description_
        boxB (_type_): _description_

    Returns:
        _type_: _description_
    """
    return boxA["cy"] > boxB["cy"]




def is_next_to(boxA, boxB, image_width):
    """_summary_

    Args:
        boxA (_type_): _description_
        boxB (_type_): _description_
        image_width (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Computing the threshold
    threshold = 0.15 * image_width

    # Computing the minimum edge-to-edge distance
    # Using centers would penalize large objects
    dx = max(
        boxA["x1"] - boxB["x2"],
        boxB["x1"] - boxA["x2"],
        0
    )

    dy = max(
        boxA["y1"] - boxB["y2"],
        boxB["y1"] - boxA["y2"],
        0
    )

    distance = math.sqrt(dx**2 + dy**2)

    return distance < threshold

def horizontal_overlap(boxA, boxB):
    """_summary_

    Args:
        boxA (_type_): _description_
        boxB (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Compute the overlap
    overlap = max(
        0,
        min(boxA["x2"], boxB["x2"]) -
        max(boxA["x1"], boxB["x1"])
    )

    return overlap

def is_on_top_of(boxA, boxB, threshold=0.2):
    """_summary_

    Args:
        boxA (_type_): _description_
        boxB (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Threshold = 20% of the width of the smallest object
    return (
        boxA["cy"] < boxB["cy"] and
        horizontal_overlap(boxA, boxB) >
        threshold * min(boxA["w"], boxB["w"])
    )

def is_under(boxA, boxB):
    """_summary_

    Args:
        boxA (_type_): _description_
        boxB (_type_): _description_

    Returns:
        _type_: _description_
    """
    return is_on_top_of(boxB, boxA)

def is_inside(A, B):
    """_summary_

    Args:
        A (_type_): _description_
        B (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Get the coordinates
    ## A
    ax1 = float(A["x1"])
    ax2 = float(A["x2"])
    ay1 = float(A["y1"])
    ay2 = float(A["y2"])
    ## B
    bx1 = float(B["x1"])
    bx2 = float(B["x2"])
    by1 = float(B["y1"])
    by2 = float(B["y2"])

    return (
        ax1 >= bx1 and
        ay1 >= by1 and
        ax2 <= bx2 and
        ay2 <= by2
    )


def object_depth(box, depth_map):
    """_summary_

    Args:
        box (_type_): _description_
        depth_map (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Retrieve the coordinates
    x1 = int(box["x1"])
    y1 = int(box["y1"])
    x2 = int(box["x2"])
    y2 = int(box["y2"])
    # Retrieve width and height
    w = int(box["w"])
    h = int(box["h"])
    # Retrieve the centers
    cx1 = int(x1 + 0.25 * w)
    cx2 = int(x1 + 0.75 * w)
    cy1 = int(y1 + 0.25 * h)
    cy2 = int(y1 + 0.75 * h)
    # Compute the region
    region = depth_map[cy1:cy2, cx1:cx2]

    # Return the median of the region
    # More potent than the mean
    return np.median(region)

def is_in_front_of(depthA, depthB, margin=0.5):
    """_summary_

    Args:
        depthA (_type_): _description_
        depthB (_type_): _description_
        margin (float, optional): _description_. Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    return depthA < depthB - margin

def is_behind(depthA, depthB, margin=0.5):
    """_summary_

    Args:
        depthA (_type_): _description_
        depthB (_type_): _description_
        margin (float, optional): _description_. Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    return depthA > depthB + margin




    
    
