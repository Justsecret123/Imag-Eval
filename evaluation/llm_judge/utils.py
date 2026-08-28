
import json
import math
import os
from .codes_info import SKILLS, MODELS
from dotenv import load_dotenv
import numpy as np
import pandas as pd

def process_annotations_file(annotations_file: pd.DataFrame):
    """
    Standardize annotation column types for downstream processing.

    The function converts all benchmark annotation columns to Pandas
    string dtype, ensuring a consistent representation across evaluation,
    aggregation, and post-processing stages.

    Args:
        annotations_file (pd.DataFrame): Annotation dataframe containing
            raw or partially processed benchmark results.

    Returns:
        pd.DataFrame: Annotation dataframe with all evaluation columns
        converted to string dtype.
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

def load_logs(logs_path: str):
    """
    Load or initialize the benchmark execution logs.

    The function retrieves the status of previously executed evaluation
    skills from a JSON log file. If the file does not exist, a new log
    structure is created and initialized with a default status for all
    supported skills.

    Args:
        logs_path (str): Path to the JSON log file.

    Returns:
        dict: Dictionary containing the execution status associated with
        each benchmark skill.
    """

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

def load_YOLO_CLASSES(path: str):
    """
    Load YOLO class mappings and construct forward and reverse lookup
    dictionaries.

    The function reads a JSON file containing YOLO class identifiers and
    their corresponding class names, then creates an additional inverted
    mapping to support class-name-to-id lookups during benchmark
    evaluation.

    Args:
        path (str): Path to the JSON file containing the YOLO class
            definitions.

    Returns:
        tuple:
            - dict: Mapping from YOLO class identifiers to class names.
            - dict: Reverse mapping from class names to YOLO class
              identifiers.
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


def get_level(name: str):
    """
    Extract the difficulty level from an IMAG-EVAL image filename.

    The function parses an image name following the IMAG-EVAL naming
    convention, removes the model identifier and file extension, and
    retrieves the difficulty level associated with the corresponding
    benchmark prompt.

    Args:
        name (str): Image filename following the IMAG-EVAL naming scheme.

    Returns:
        str: Difficulty level encoded in the filename
        (e.g., "easy", "medium", or "hard").
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


def get_skill_code(name: str):
    """
    Extract the skill-combination code from an IMAG-EVAL image filename.

    The function parses an image name following the IMAG-EVAL naming
    convention, removes the model identifier and file extension, and
    retrieves the code corresponding to the evaluated skill combination.

    Args:
        name (str): Image filename following the IMAG-EVAL naming scheme.

    Returns:
        str: Skill-combination code associated with the image.
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


def extract_robustness(image_name: str):
    """
    Determine whether an image belongs to a robustness evaluation setting.

    The function inspects an IMAG-EVAL image filename and checks whether
    it corresponds to a robustness test case, identified by the presence
    of the `robust` tag in the standardized naming convention.

    Args:
        image_name (str): Image filename following the IMAG-EVAL naming
            scheme.

    Returns:
        bool: True if the image belongs to a robustness evaluation
        setting, otherwise False.
    """

    return "robust" in image_name

def load_environment() -> str:
    """
    Load environment variables required by the IMAG-EVAL evaluation
    pipeline.

    The function reads the project's `.env` file and retrieves the
    credentials and configuration parameters needed to access protected
    models and resources.

    Returns:
        tuple:
            - str | bool: Hugging Face access token (`HF_TOKEN`) if
              available, otherwise `False`.
            - str | bool: Path to the Qwen-VL FP8 model
              (`QWEN_VL_FP8_PATH`) if available, otherwise `False`.
    """

    # Load the environment files
    load_dotenv(".env")
    # Load the access token
    access_token = os.environ["HF_TOKEN"] if os.environ["HF_TOKEN"] else False
    # Load the QWEN_VL_FP8 path
    qwen_path = os.environ["QWEN_VL_FP8_PATH"] if os.environ["QWEN_VL_FP8_PATH"] else False

    return access_token, qwen_path

def get_skill_code(name: str):
    """
    Extract the skill-combination code from an IMAG-EVAL image filename.

    The function parses an image name following the IMAG-EVAL naming
    convention, removes the model identifier and file extension, and
    retrieves the code associated with the evaluated skill combination.

    Args:
        name (str): Image filename following the IMAG-EVAL naming scheme.

    Returns:
        str: Skill-combination code encoded in the filename.
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
    """
    Retrieve the prompt file and prompt position associated with an
    annotation entry.

    The function parses an IMAG-EVAL image filename to determine the
    corresponding skill combination, difficulty level, and robustness
    setting. Using this information, it identifies the prompt JSON file
    from which the image originated and computes the position of the
    associated prompt group within the benchmark structure.

    Args:
        row (pd.Series): Annotation row containing image metadata.
        code_file (pd.DataFrame): Mapping between skill-combination codes
            and skill names.

    Returns:
        tuple:
            - str: Path to the prompt JSON file associated with the image.
            - int: Position of the prompt level within the prompt file
              (hard=0, medium=1, easy=2).
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

def extract_counting_rules(prompt_file: str, position: int):
    """
    Extract object-count constraints from a benchmark prompt definition.

    The function loads a prompt configuration file and retrieves the
    counting rules associated with a specific prompt entry. For each
    object, it determines the required number of instances to generate.
    When Emotion evaluation is enabled, the required number of person
    instances is also derived from the emotion annotations.

    Args:
        prompt_file (str): Path to the prompt JSON file containing the
            benchmark definitions.
        position (int): Index of the prompt entry to extract within the
            prompt collection.

    Returns:
        dict: Mapping between object names and the required number of
        instances to generate.
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


def extract_size_rules(prompt_file: str, position: int):
    """
    Extract size-related constraints from a benchmark prompt definition.

    The function loads a prompt configuration file and retrieves the
    relative size rules associated with a specific prompt entry. It also
    collects the set of objects involved in at least one size
    relationship, which are later used during evaluation.

    Args:
        prompt_file (str): Path to the prompt JSON file containing the
            benchmark definitions.
        position (int): Index of the prompt entry to extract within the
            prompt collection.

    Returns:
        tuple:
            - list: Collection of size rules represented as
              object-relation-object triplets.
            - list: Objects participating in the extracted size
              relationships.
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

def extract_spatial_rules(prompt_file: str, position: int):
    """
    Extract spatial relationship constraints from a benchmark prompt
    definition.

    The function loads a prompt configuration file and retrieves the
    spatial relations associated with a specific prompt entry. It also
    identifies all objects participating in at least one spatial rule,
    which are subsequently used during spatial-relation evaluation.

    Args:
        prompt_file (str): Path to the prompt JSON file containing the
            benchmark definitions.
        position (int): Index of the prompt entry to extract within the
            prompt collection.

    Returns:
        tuple:
            - list: Collection of spatial rules represented as
              object-relation-object triplets.
            - list: Objects referenced by the extracted spatial
              relationships.
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

def extract_emotion_rules(prompt_file: str, position: int):
    """
    Extract emotion constraints from a benchmark prompt definition.

    The function loads a prompt configuration file and retrieves the
    emotion annotations associated with a specific prompt entry. These
    annotations serve as the ground-truth targets used during Emotion
    evaluation.

    Args:
        prompt_file (str): Path to the prompt JSON file containing the
            benchmark definitions.
        position (int): Index of the prompt entry to extract within the
            prompt collection.

    Returns:
        list: Collection of emotions specified for the selected prompt.
    """

    # Open the associated prompt file
    with open(prompt_file, "rb") as file:
        # Load the data
        prompt_data = json.load(file)
        # Get the emotions rules
        emotions = prompt_data["prompts"][position]["scene"]["emotion"]

    return emotions



def process_size_relations(areas_A: list, areas_B: list) -> list:
    """
    Derive relative size relationships between two sets of object
    instances based on their segmented areas.

    The function compares every area in `areas_A` against every area in
    `areas_B` and assigns a relative size relationship
    ("larger", "smaller", or "equal") for each pair. These derived
    relations are subsequently compared against the ground-truth size
    constraints defined by the benchmark.

    Args:
        areas_A (list): Segmented areas associated with instances of the
            first object.
        areas_B (list): Segmented areas associated with instances of the
            second object.

    Returns:
        list: Collection of inferred size relationships between the two
        object groups.
    """

    # Initialize the list of relationships
    relations = list()

    # Loop through the areas
    for area_A in areas_A:
        for area_B in areas_B:
            if area_A > area_B:
                relations.append("larger")
            if area_A < area_B:
                relations.append("smaller")
            if area_A==area_B:
                relations.append("equal")

    return relations

def extract_color_rules(prompt_file: str, position: int):
    """
    Extract color constraints from a benchmark prompt definition.

    The function loads a prompt configuration file and retrieves all
    object-color associations defined for a specific prompt entry. It
    also identifies the set of objects involved in color evaluation,
    which are subsequently used during Color assessment.

    Args:
        prompt_file (str): Path to the prompt JSON file containing the
            benchmark definitions.
        position (int): Index of the prompt entry to extract within the
            prompt collection.

    Returns:
        tuple:
            - dict: Mapping between object names and their required
              colors.
            - set: Objects associated with at least one color
              constraint.
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


def compute_inverse_size_rule(rule: str):
    """
    Compute the inverse of a relative size relationship.

    The function converts a size constraint into its opposite relation,
    enabling bidirectional comparison of predicted and ground-truth size
    rules during benchmark evaluation.

    Args:
        rule (str): Relative size relationship to invert.

    Returns:
        str: Inverse size relationship. Returns "smaller" for
        "larger", "larger" for "smaller", and the original value for
        any other relationship.
    """

    # Larger ---> smaller
    if rule=="larger":
        return "smaller"
    # Smaller ---->  larger
    if rule=="smaller":
        return "larger"

    return rule


def get_box_info(box):
    """
    Convert a bounding box into a structured geometric representation.

    The function transforms a bounding box defined by its corner
    coordinates into a dictionary containing spatial attributes such as
    position, dimensions, and center coordinates. These derived
    properties are used during IMAG-EVAL spatial-relationship and
    depth-based evaluations.

    Args:
        box (array-like): Bounding box coordinates in the format
            `(x1, y1, x2, y2)`.

    Returns:
        dict: Dictionary containing the bounding box coordinates,
        dimensions, and center position with the following fields:

            - `x1`, `y1`: Top-left corner coordinates.
            - `x2`, `y2`: Bottom-right corner coordinates.
            - `cx`, `cy`: Center coordinates.
            - `w`, `h`: Width and height of the bounding box.
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
    """
    Determine whether one object is positioned to the left of another.

    The function compares the horizontal center coordinates of two
    bounding boxes and evaluates the corresponding spatial relationship
    used in IMAG-EVAL spatial-relation assessment.

    Args:
        boxA (dict): Bounding-box representation of the first object.
        boxB (dict): Bounding-box representation of the second object.

    Returns:
        bool: True if the center of the first object is located to the
        left of the center of the second object, otherwise False.
    """
    return boxA["cx"] < boxB["cx"]


def is_right_of(boxA, boxB):
    """
    Determine whether one object is positioned to the right of another.

    The function compares the horizontal center coordinates of two
    bounding boxes and evaluates the corresponding spatial relationship
    used in IMAG-EVAL spatial-relation assessment.

    Args:
        boxA (dict): Bounding-box representation of the first object.
        boxB (dict): Bounding-box representation of the second object.

    Returns:
        bool: True if the center of the first object is located to the
        right of the center of the second object, otherwise False.
    """
    return boxA["cx"] > boxB["cx"]


def is_above(boxA, boxB):
    """
    Determine whether one object is positioned above another.

    The function compares the vertical center coordinates of two
    bounding boxes and evaluates the corresponding spatial relationship
    used in IMAG-EVAL spatial-relation assessment.

    Args:
        boxA (dict): Bounding-box representation of the first object.
        boxB (dict): Bounding-box representation of the second object.

    Returns:
        bool: True if the center of the first object is located above the
        center of the second object, otherwise False.
    """
    return boxA["cy"] < boxB["cy"]

def is_below(boxA, boxB):
    """
    Determine whether one object is positioned below another.

    The function compares the vertical center coordinates of two
    bounding boxes and evaluates the corresponding spatial relationship
    used in IMAG-EVAL spatial-relation assessment.

    Args:
        boxA (dict): Bounding-box representation of the first object.
        boxB (dict): Bounding-box representation of the second object.

    Returns:
        bool: True if the center of the first object is located below the
        center of the second object, otherwise False.
    """
    return boxA["cy"] > boxB["cy"]


def is_next_to(boxA, boxB, image_width):
    """
    Determine whether two objects are spatially close to one another.

    The function computes the minimum edge-to-edge distance between two
    bounding boxes and evaluates whether that distance falls below a
    threshold proportional to the image width. This criterion is used to
    assess the "next to" spatial relationship in IMAG-EVAL.

    Unlike center-based distance measures, the implementation relies on
    object edges to avoid penalizing large objects whose centers may be
    far apart despite being adjacent.

    Args:
        boxA (dict): Bounding-box representation of the first object.
        boxB (dict): Bounding-box representation of the second object.
        image_width (int | float): Width of the original image, used to
            compute the adjacency threshold.

    Returns:
        bool: True if the objects are considered next to each other,
        otherwise False.
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
    """
    Compute the horizontal overlap between two bounding boxes.

    The function measures the extent to which two objects overlap along
    the horizontal axis. The resulting value is used as a geometric
    feature for evaluating spatial relationships between detected
    objects in IMAG-EVAL.

    Args:
        boxA (dict): Bounding-box representation of the first object.
        boxB (dict): Bounding-box representation of the second object.

    Returns:
        float: Horizontal overlap between the two bounding boxes,
        expressed in image-coordinate units. Returns 0 when the boxes do
        not overlap horizontally.
    """
    # Compute the overlap
    overlap = max(
        0,
        min(boxA["x2"], boxB["x2"]) -
        max(boxA["x1"], boxB["x1"])
    )

    return overlap

def is_on_top_of(boxA, boxB, threshold=0.2):
    """
    Determine whether one object is positioned on top of another.

    The function evaluates the "on top of" spatial relationship by
    combining two criteria: the first object must be vertically above
    the second object, and the two objects must exhibit sufficient
    horizontal overlap. The overlap threshold is expressed as a fraction
    of the width of the smaller object.

    Args:
        boxA (dict): Bounding-box representation of the first object.
        boxB (dict): Bounding-box representation of the second object.
        threshold (float, optional): Minimum horizontal overlap ratio
            required to consider the objects vertically aligned.
            Defaults to 0.2.

    Returns:
        bool: True if the first object is considered to be on top of the
        second object, otherwise False.
    """

    # Threshold = 20% of the width of the smallest object
    return (
        boxA["cy"] < boxB["cy"] and
        horizontal_overlap(boxA, boxB) >
        threshold * min(boxA["w"], boxB["w"])
    )

def is_under(boxA, boxB):
    """
    Determine whether one object is positioned under another.

    The function evaluates the "under" spatial relationship by checking
    whether the first object's bounding box satisfies the inverse of the
    "on top of" relationship relative to the second object.

    Args:
        boxA (dict): Bounding-box representation of the first object.
        boxB (dict): Bounding-box representation of the second object.

    Returns:
        bool: True if the first object is considered to be under the
        second object, otherwise False.
    """
    return is_on_top_of(boxB, boxA)

def is_inside(A, B):
    """
    Determine whether one object is entirely contained within another.

    The function evaluates the "inside" spatial relationship by checking
    whether the bounding box of the first object lies completely within
    the bounding box of the second object. This criterion is used during
    IMAG-EVAL spatial-relationship evaluation.

    Args:
        A (dict): Bounding-box representation of the object expected to
            be inside.
        B (dict): Bounding-box representation of the enclosing object.

    Returns:
        bool: True if the first bounding box is fully contained within
        the second bounding box, otherwise False.
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
    """
    Estimate the representative depth of a detected object from a depth map.

    The function extracts a central region within the object's bounding
    box and computes its median depth value. Using the median rather than
    the mean reduces the influence of outliers and noisy depth
    predictions, providing a more robust estimate of the object's
    distance from the camera.

    The resulting depth value is used during IMAG-EVAL evaluation of
    depth-dependent spatial relationships such as "in front of" and
    "behind".

    Args:
        box (dict): Bounding-box representation of the object containing
            coordinates and dimensions.
        depth_map (numpy.ndarray): Predicted depth map associated with
            the image.

    Returns:
        float: Representative depth value for the object, computed as the
        median depth within the central region of its bounding box.
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
    """
    Determine whether one object is positioned in front of another based
    on estimated depth values.

    The function compares the representative depths of two objects and
    evaluates the "in front of" spatial relationship. An object is
    considered to be in front of another when its estimated depth is
    sufficiently smaller, according to the specified depth margin.

    This criterion is used during IMAG-EVAL evaluation of depth-based
    spatial relationships.

    Args:
        depthA (float): Estimated depth of the first object.
        depthB (float): Estimated depth of the second object.
    """
    return depthA < depthB - margin

def is_behind(depthA, depthB, margin=0.5):
    """
    Determine whether one object is positioned behind another based on
    estimated depth values.

    The function compares the representative depths of two objects and
    evaluates the "behind" spatial relationship. An object is
    considered to be behind another when its estimated depth is
    sufficiently larger, according to the specified depth margin.

    This criterion is used during IMAG-EVAL evaluation of depth-based
    spatial relationships.

    Args:
        depthA (float): Estimated depth of the first object.
        depthB (float): Estimated depth of the second object.
        margin (float, optional): Minimum depth difference required to
            consider the first object behind the second. Defaults
            to 0.5.

    Returns:
        bool: True if the first object is considered to be behind the
        second object, otherwise False.
    """
    return depthA > depthB + margin




    
    
