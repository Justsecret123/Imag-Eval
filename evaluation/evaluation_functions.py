import numpy as np
import pandas as pd
import werpy
from evaluation.llm_judge.codes_info import MODELS

# Set the list of texts
texts = pd.read_csv("../text.csv", encoding="latin1")


def compute_WER(row: pd.Series):
    """
    Compute the Word Error Rate (WER) for a benchmark annotation.

    The function matches a generated image with its corresponding target
    text prompt, retrieves the transcribed text annotation, and computes
    the Word Error Rate (WER) between the generated and reference texts.

    Images without an associated reference text, missing annotations, or
    images containing no text are assigned a missing value.

    Args:
        row (pd.Series): Annotation row containing image metadata and
            extracted text predictions.

    Returns:
        float | numpy.nan: Word Error Rate for the corresponding image.
        Returns `numpy.nan` when no valid comparison can be performed.
    """

    # Initialize the image name
    image_name = dict(row)["Image"]
    # Initialize the WER
    WER = np.nan
    # Retrieve models name
    for model in MODELS:
        if model in image_name:
            # Remove the model name string
            image_name = image_name.replace(model+"_","")
            # Remove the extension
            image_name = image_name.replace(".png","")
            break

    # Retrieve the row's text
    row_text = str(dict(row)["Text"])
    # Get the matching text
    matching_text = list(texts[texts["id"]==image_name]["text"])
    # Check if the row has a matching text
    if matching_text!=[]:
        # If the text is N/A
        if pd.isna(row_text) or "There is no text in this image".lower() in row_text.lower().strip():
            WER = np.nan
        else:
            # Compute the wer
            WER = werpy.wer(row_text,matching_text[0])

    return WER

def get_level(name: str):
    """
    Extract the difficulty level associated with a benchmark image.

    The function parses an IMAG-EVAL image filename, removes the model
    identifier and file extension, and retrieves the prompt difficulty
    level encoded in the standardized naming convention.

    Args:
        name (str): Image filename following the IMAG-EVAL naming scheme.

    Returns:
        str: Prompt difficulty level associated with the image
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
    retrieves the skill code associated with the evaluated prompt
    configuration.

    Args:
        name (str): Image filename following the IMAG-EVAL naming scheme.

    Returns:
        str: Skill code associated with the image.
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

def compute_mean_accuracy(elements):
    """
    Compute the mean counting accuracy from an IMAG-EVAL counting annotation.

    Counting annotations are represented as a sequence of
    `generated,required` pairs separated by semicolons. An object is
    considered correctly generated when the number of generated instances
    exactly matches the required count. The final score corresponds to
    the proportion of correctly generated objects.

    Args:
        elements (str): Counting annotation string following the
            IMAG-EVAL format
            (`generated,required;generated,required;...`).

    Returns:
        float | numpy.nan: Mean counting accuracy rounded to two decimal
        places, or `numpy.nan` if the annotation cannot be processed.
    """

    # Initialize the average
    average = np.nan
    
    if isinstance(elements,str):
        try:
            # Remove extra spaces
            elements = elements.rstrip()
            # Initialize the count of objects
            objects_count = 0
            # Initialize the list of items
            items = list()
            # Initialize the list of elements
            elements = elements.split(";")
            # Loop through the elements
            for element in elements:
                # Get the parts
                parts = element.split(",")
                if len(parts)>1:
                    # Get the values
                    value_1 = int(parts[0])
                    value_2 = int(parts[1])
                    # Increase the number of objects
                    objects_count+=value_2
                    # Add to the list of elements
                    items.append((value_1,value_2))
            # Compute the average precision
            average = round(sum(1 for generated,asked in items if generated==asked)/len(items),2)
        except: 
            return np.nan

    return average

def process_cohesiveness(text: str):
    """
    Convert cohesiveness annotations to a standardized boolean format.

    The function normalizes cohesiveness values originating from manual
    annotations or automated evaluations. It supports textual
    representations (e.g., "true", "false", "vrai"), boolean values, and
    numeric encodings (e.g., 1 and 0).

    Args:
        text (str): Cohesiveness annotation to process.

    Returns:
        bool | float | int: Normalized boolean value when a valid
        cohesiveness annotation is provided. Missing values are preserved
        unchanged.
    """

    if not isinstance(text,str):
        # Bool
        if isinstance(text,bool):
            return text
        # Float or int and not na
        if (isinstance(text,float) or isinstance(text,int)) and pd.notna(text):
            return int(text)==1
        elif (isinstance(text,float) or isinstance(text,int)) and not pd.notna(text):
            return text

    return ("vrai" in text.strip().lower() or "true" in text.strip().lower()) and pd.notna(text)

def extract_robustness(image_name: str):
    """
    Determine whether an image belongs to a robustness evaluation setting.

    The function inspects an IMAG-EVAL image filename and checks whether
    it corresponds to a robustness test case, identified by the presence
    of the `_robust` suffix in the standardized naming convention.

    Args:
        image_name (str): Image filename following the IMAG-EVAL naming
            scheme.

    Returns:
        bool: True if the image belongs to a robustness evaluation,
        otherwise False.
    """

    return "robust" in image_name


def compute_combinations(annotations: pd.DataFrame, test: str):
    """
    Identify and display the best-performing skill combinations for a
    given evaluation setting.

    The function retrieves the highest-performing configurations for
    each evaluated skill (Counting, Spatial, Size, Emotion, and Color)
    as well as the lowest Word Error Rate (WER). It then reports the
    corresponding skill-combination codes associated with these best
    results.

    Args:
        annotations (pd.DataFrame): DataFrame containing benchmark
            evaluation results and skill scores.
        test (str): Name of the evaluation setting being analyzed
            (e.g., a model, difficulty level, or benchmark subset).

    Returns:
        int: Returns 0 after displaying the best-performing skill
        combinations.
    """

    # Extract the best parameters (values)
    max_counting_test = annotations["Counting"].max(axis=0)
    min_wer_test = annotations["WER"].min(axis=0)
    max_spatial_test = annotations["Spatial"].max(axis=0)
    max_size_test = annotations["Size"].max(axis=0)
    max_emotion_test = annotations["Emotion"].max(axis=0)
    max_colors_test = annotations["Colors"].max(axis=0)

    print(f"Max counting: {max_counting_test} -  Min WER: {min_wer_test} - Max spatial: {max_spatial_test} -\
          \nMax size: {max_size_test} - Max emotion: {max_emotion_test} - Max colors: {max_colors_test} ")
    # Extract lines that fit the best parameters
    best_counting_level = annotations[annotations["Counting"]==max_counting_test]
    best_wer_level = annotations[annotations["WER"]==min_wer_test]
    best_spatial_level = annotations[annotations["Spatial"]==max_spatial_test]
    best_size_level = annotations[annotations["Size"]==max_size_test]
    best_emotion_level = annotations[annotations["Emotion"]==max_emotion_test]
    best_colors_level = annotations[annotations["Colors"]==max_colors_test]
    # Displaying results
    print(f"Skill codes ({test}):\n- Counting ({test}): {list(best_counting_level['Code'].unique())}\
        \n- WER ({test}) : {list(best_wer_level['Code'].unique())}\
        \n- Spatial ({test}) : {list(best_spatial_level['Code'].unique())}\
        \n- Size ({test}) : {list(best_size_level['Code'].unique())}\
        \n- Emotion ({test}) : {list(best_emotion_level['Code'].unique())}\
        \n- Colors ({test}) : {list(best_colors_level['Code'].unique())}")
    
    return 0



def majority_or_mean(series):
    """
    Aggregate multiple annotations using majority voting when possible.

    The function first checks whether a strict majority exists among the
    non-missing values. If a majority is found, the majority value is
    returned. Otherwise, for numeric data, the arithmetic mean is used as
    an aggregate estimate. If neither condition can be satisfied, a
    missing value is returned.

    Args:
        series (pd.Series): Collection of annotations or scores to
            aggregate.

    Returns:
        object: Majority value when a strict majority exists, mean value
        for numeric series without a majority, or `numpy.nan` when no
        valid aggregation can be computed.
    """
    
    # Drop NaN for majority check
    counts = series.value_counts(dropna=True)
    total = len(series.dropna())

    # If all values are NaN
    if total == 0:
        return np.nan

    # Strict majority
    if counts.iloc[0] > total / 2:
        return counts.index[0]

    # No majority, we return mean if numeric
    if pd.api.types.is_numeric_dtype(series):
        return series.mean()

    # Otherwise, we return nan
    return np.nan