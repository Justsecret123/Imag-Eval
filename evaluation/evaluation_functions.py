import numpy as np
import pandas as pd
import werpy
from evaluation.llm_judge.codes_info import MODELS

# Set the list of texts
texts = pd.read_csv("../text.csv", encoding="latin1")


def compute_WER(row:pd.Series):
    """Computes the WER for a specific row. 

    Args:
        row (pd.Series): _description_

    Returns:
        _type_: _description_
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
        if pd.isna(row_text):
            WER = np.nan
        else:
            # Compute the wer
            WER = werpy.wer(row_text,matching_text[0])

    return WER

def get_level(name:str):
    """Extract the level of a specific row

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
    """Extracts the skill code from the image name"""

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
    """Computes the mean accuracy of a list of tuples."""

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

def process_cohesiveness(text:str):
    """
    Transforms the text in cohesiveness into booleans.
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

    return ("Vrai" in text.strip() or "True" in text.strip()) and pd.notna(text)

def extract_robustness(image_name:str):
    """Extracts the robustness of the test from the image name.
    Returns True if any, False not."""

    return "robust" in image_name


def compute_combinations(annotations:pd.DataFrame, test:str):
    """
    Computes the best combinations for the specified test.
    Displays the combinations and returns nothing.
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