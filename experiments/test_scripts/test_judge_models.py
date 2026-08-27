import argparse
from collections import Counter, defaultdict
import itertools
import json
import gc
import os
import traceback
import cv2
import numpy as np
import pandas as pd
import torch
import evaluation.llm_judge.utils as utils
from evaluation.llm_judge.codes_info import TEXTS_CODES, SIZE_CODES, SPATIAL_CODES, \
    COUNTING_CODES, COLOR_CODES, EMOTION_CODES, VALID_EMOTIONS, VALID_COLORS, VALID_COHESIVENESS
from transformers import AutoProcessor
from sglang import Engine
from sglang import Engine
from qwen_vl_utils import process_vision_info
from ultralytics import YOLO


# Initialize the seed
SEED = 42
# Initialize the verbosity
VERBOSE = 1
# Load the judges LOGS
LOGS = dict()
# Initialize the qwen-VL-FP8 path
QWEN_VL_FP8_PATH = ""


def extract_texts(annotations_file:pd.DataFrame, images_folder:str, device:str, output_file:str, access_token="", text_extraction="qwen-vl"):
    """_summary_

    Args:
        annotations_file (pd.DataFrame): _description_
        images_folder (str): _description_
        image_model (str): _description_
        device (str): _description_
        output_file (str): _description_
        access_token (str, optional): _description_. Defaults to "".
        text_extraction (str, optional): _description_. Defaults to "qwen-vl".

    Returns:
        _type_: _description_
    """

    # Qwen-VL extraction
    # Load libraries 
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    # Loading the model
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct", 
        torch_dtype=torch.bfloat16, 
        device_map=device, 
        token=access_token,
            #attn_implementation="flash_attention_2"
        )
    # Loading the processor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    
    # Loop through the rows
    for idx,row in annotations_file.iterrows():
        # Extract the skill code
        skill_code = int(utils.get_skill_code(row["Image"]))
        # Check if a text was asked
        if skill_code in TEXTS_CODES:
            try:
                # Set a message
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": f"{images_folder}/{row['Image']}",
                            },
                            {"type": "text", "text": "Extact the text from this image. Output only the text."},
                        ],
                    }
                ]
                # Preparation for inference
                text = processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                # Process the vision info
                image_inputs, video_inputs = process_vision_info(messages)
                # Setup the inputs
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                # Placing inputs on GPU
                inputs = inputs.to(device)
                # Inference: Generation of the output
                generated_ids = model.generate(**inputs, 
                                               max_new_tokens=128, 
                                               temperature=1)

                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                # Get the output text
                output_text = processor.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )
                # Get the extracted text
                extracted_text = output_text[0].replace("\n"," ").strip()
                # Display a message
                if VERBOSE:
                    print(f'Extracted text: {extracted_text}')
                # Annotate the row
                annotations_file.at[idx,"Text"] = extracted_text

            except Exception:
                # Display error message
                print(f"Error when predicting for image {row['Image']}")
                traceback.print_exc()
    
    # Save results
    annotations_file.to_excel(output_file, index=False)
    # Free memory
    del model, processor
    free_memory()

    return annotations_file


def extract_objects_yolo(annotations_file:pd.DataFrame, output_file:str, code_file:pd.DataFrame, images_folder:str, detection_model:str):
    """_summary_

    Args:
        annotations_file (pd.DataFrame): _description_
        code_file (pd.DataFrame): _description_
        images_folder (str): _description_
        detection_model (str): _description_

    Returns:
        _type_: _description_
    """


    # Load a model
    model = YOLO(f"{detection_model}.pt")

    # Loop through the rows
    for idx,row in annotations_file.iterrows():
        # Extract the skill code
        skill_code = int(utils.get_skill_code(row["Image"]))
        # Extract the prompt file and the meta-prompt position
        prompt_file, position = utils.extract_prompt_file(row,code_file)
        # Check if a counting rule was asked
        if skill_code in COUNTING_CODES:
            try:
                # Extract the counting rules
                counting_rules = utils.extract_counting_rules(prompt_file, position)
                # Perform YOLO inference 
                results = model(f"{images_folder}/{row['Image']}")
                # Initialize the annotation for the current line
                annotation = ""
                # Access the results
                for result in results: 
                    # Get the object names 
                    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]
                    names_dict = Counter(names)
                    if VERBOSE:
                        print(f"File name: {row['Image']}, objects extracted: {names}")
                        print(f"Counting rules: {counting_rules}")
                    # Loop through the counting rules
                    # Couting annotations = (instances_generated,instances_required)
                    for object, count in counting_rules.items():
                        # Check if the object has a correspondance in the extracted objecs
                        if object in names:
                            # We set the annotation accordingly
                            annotation+=f"{names_dict[object]},{count};"
                        else:
                            annotation+=f"0,{count};"
                    # Delete the trailing semicolon
                    if annotation[-1]==";":
                        annotation = annotation[:-1]
                    # Annotate the row
                    annotations_file.at[idx,"Counting"] = annotation
                    if VERBOSE:
                        print(f"Counting annotation for this line: {annotation}")
            except: 
                print(f"Error when predicting for image {row['Image']}")
                traceback.print_exc()

    # Save results
    annotations_file.to_excel(output_file, index=False)
    # Free memory
    del model
    free_memory()

    return annotations_file


def extract_sizes(annotations_file:pd.DataFrame, output_file:str, code_file:pd.DataFrame, images_folder:str, detection_model:str):
    """_summary_

    Args:
        annotations_file (pd.DataFrame): _description_
        code_file (pd.DataFrame): _description_
        images_folder (str): _description_
        detection_model (str): _description_

    Returns:
        _type_: _description_
    """

    # Load a model
    model = YOLO(f"{detection_model}-seg.pt")

    # Loop through the rows
    for idx,row in annotations_file.iterrows():
        # Extract the skill code
        skill_code = int(utils.get_skill_code(row["Image"]))
        # Extract the prompt file and the meta-prompt position
        prompt_file, position = utils.extract_prompt_file(row,code_file)

        if skill_code in SIZE_CODES:
            try:
                # Extract the size rules
                size_rules, objects = utils.extract_size_rules(prompt_file,position)
                # Initialize the generated sizes
                generated_areas = defaultdict(list)
                # Initialize the results
                size_results = []
                # Perform YOLO inference
                results = model(f"{images_folder}/{row['Image']}")
                # Loop through the results
                for result in results:
                    # Skip if the mask is empty
                    if result.masks is None:
                        continue 
                    # Extract the masks
                    masks = result.masks.data
                    # Compute areas
                    areas = masks.sum(dim=(1,2))
                    # Retrieve the instances names
                    for i, area in enumerate(areas):
                        class_id = int(result.boxes.cls[i])
                        class_name = result.names[class_id]
                        # Display results
                        print(f"Object: {i}, Class={class_name}, Area={area}")
                        # Filtering out the unwanted objects
                        if class_name in objects:
                            print(f"Object: {i}, Class={class_name}, Area={area}")
                            generated_areas[class_name].append(area)
                    # Get the combinations of objects
                    objects_combinations = itertools.combinations(objects,2)

                    for object_A, object_B in objects_combinations:
                        # Compute the size relations between objects
                        relations = utils.process_size_relations(generated_areas[object_A], 
                                                                     generated_areas[object_B])
                        # Verify with the original rules
                        # Case 0 : equal areas (doesn't exist)
                        for relation in relations:
                            if relation=="equal":
                                # Failure
                                size_results.append(False)
                            # Case 1 : the relationship exists
                            elif [object_A, relation, object_B] in size_rules:
                                # Success
                                size_results.append(True)
                            # Case 2 : the inverse relationship exists
                            elif [object_A, utils.compute_inverse_size_rule(relation), object_B] in size_rules:
                                # Failure 
                                size_results.append(False)
                            # Case 3 : a relationship exists between the objects, but reversed
                            elif [object_B, relation, object_A] in size_rules:
                                # Failure 
                                size_results.append(False)
                            # Case 4 : the inverse relationship exists between the objects
                            elif [object_B, utils.compute_inverse_size_rule(relation), object_A] in size_rules:
                                # Success 
                                size_results.append(True)
                    # Compute the success rate : we avoid error propagation between counting and sizes
                    # The rate is only computed when the objects are effectively detected
                    success_rate = "" if len(size_results)==0 else round(sum(size_results)/len(size_results),2)
                    # Annotate the row
                    annotations_file.at[idx,"Size"] = str(success_rate)
                    # Display results
                    if VERBOSE:
                        print(f"Size rules for : {size_rules}")
                        print(f"\n\nSize (success) rate for the current line: {success_rate}\n")
            except Exception:
                # Display error message
                print(f"Error when predicting for image {row['Image']}")
                traceback.print_exc()

    # Save results
    annotations_file.to_excel(output_file, index=False)
    # Free memory
    del model
    free_memory()


    return annotations_file


def extract_spatial_yolo(annotations_file:pd.DataFrame, output_file:str, code_file:pd.DataFrame, images_folder:str, object_detection:str, depth_estimation:str):
    """_summary_

    Args:
        annotations_file (pd.DataFrame): _description_
        code_file (pd.DataFrame): _description_
        images_folder (str): _description_
        object_detection (str): _description_
        depth_estimation (str): _description_

    Returns:
        _type_: _description_
    """

    # Loop through the rows
    for idx,row in annotations_file.iterrows():
        # Extract the skill code
        skill_code = int(utils.get_skill_code(row["Image"]))
        # Extract the prompt file and the meta-prompt position
        prompt_file, position = utils.extract_prompt_file(row,code_file)
        # Initialize the bounding boxes
        bounding_boxes = defaultdict(list)
        # Initialize the depth maps
        depths = defaultdict(list)
        # Check if a counting rule was asked
        if skill_code in SPATIAL_CODES:
            try:
                compute_depth = False
                depth_results = []
                # Initialize the success rate
                success = []
                # Extract the spatial rules
                spatial_rules, objects = utils.extract_spatial_rules(prompt_file, position)
                # Check if there are depth-related rules 
                # in front, behind, inside
                for object_A, rule, object_B in spatial_rules:
                    if rule in ["in front of", "behind", "inside"]:
                        compute_depth = True
                        break
                # Load the object detection model
                detection_model = YOLO(f"{object_detection}.pt")
                # Perform object detection inference
                detection_results = detection_model(f"{images_folder}/{row['Image']}")
                # Extract the original width of the image (I use it to evaluate 'next to')
                image_width = detection_results[0].orig_shape[1]

                if compute_depth:
                    # Load the depth detection model
                    depth_model = YOLO(f"{depth_estimation}-depth.pt")
                    # Perform depth estimation inference
                    depth_results = depth_model(f"{images_folder}/{row['Image']}")

                # Depth-unrelated rules to evaluate
                for result in detection_results:
                    # Get the bounding boxes : top-left-x, top-left-y, bottom-right-x, bottom-right-y
                    boxes = result.boxes.xyxy
                    # Get the object names 
                    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]
                    # Loop through the names
                    for i,name in enumerate(names):
                        # Check if the object has a spatial rule
                        if name in objects:
                            # Display a detection result
                            if VERBOSE:
                                print(f"Detected object: {name}")
                            bounding_boxes[name].append(utils.get_box_info(boxes[i]))
                    # Loop through the rules
                    for object_A, rule, object_B in spatial_rules:
                        # Get the bounding boxes
                        box_A = bounding_boxes[object_A]
                        box_B = bounding_boxes[object_B]
                        # Loop through the objects
                        for A in box_A:
                            for B in box_B:
                                # Evaluate left
                                if rule=="to the left of":
                                    success.append(utils.is_left_of(A,B))
                                # Evaluate right
                                elif rule=="to the right of":
                                    success.append(utils.is_right_of(A,B))
                                # Evaluate above
                                elif rule=="above":
                                    success.append(utils.is_above(A,B))
                                # Evaluate under
                                elif rule=="below":
                                    success.append(utils.is_below(A,B))
                                # Evaluate 'next to'
                                elif rule=="next to":
                                    success.append(utils.is_next_to(A,B, image_width))
                                # Evaluate 'on top of'
                                elif rule=="on top of":
                                    success.append(utils.is_on_top_of(A,B))
                                # Evaluate 'under'
                                elif rule=="under":
                                    success.append(utils.is_under(A,B))
                                # Evaluate 'inside'
                                elif rule=="inside":
                                    success.append(utils.is_inside(A,B))

                # Depth-related rules
                for result in depth_results:
                    # Get the depth map
                    depth_map = result.depth.data.cpu().numpy()
                    # Loop through the results to get the names and obejcts depths
                    for name, boxes in bounding_boxes.items():
                        for box in boxes: 
                            # Display a detection result
                            if VERBOSE:
                                print(f"Detected object: {name}")
                            depths[name].append(utils.object_depth(box, depth_map))
                    # Loop through the rules
                    for object_A, rule, object_B in spatial_rules:
                        # Skip if it's not a depth-related rule
                        if rule not in["in front of", "behind"]:
                            continue
                        
                        # Get the bounding boxes
                        box_A = depths[object_A]
                        box_B = depths[object_B]
                        # Loop through the obejcts
                        for A in box_A:
                            for B in box_B:
                                # Evaluate 'in front of'
                                if rule=="in front of":
                                    success.append(utils.is_in_front_of(A,B))
                                # Evaluate 'behind'
                                elif rule=="behind":
                                    success.append(utils.is_behind(A,B))

                # Compute the success rate : we avoid error propagation between counting and sizes
                # The rate is only computed when the objects are effectively detected
                success_rate = round(sum(success)/len(success),2) if len(success) > 0 else ""
                # Annotate the row
                annotations_file.at[idx,"Spatial relationships"] = str(success_rate)
                # Display results
                if VERBOSE:
                    print(f"File name: {row['Image']}")
                    print(f"Spatial rules: {spatial_rules}") 
                    print(f"Successes: {success}")
                    print(f"Success rate: {success_rate}")
            except: 
                print(f"Error when predicting for image {row['Image']}")
                traceback.print_exc()
    # Save results
    annotations_file.to_excel(output_file, index=False)
    # Free memory
    del depth_model, detection_model
    free_memory()

    return annotations_file


def extract_emotions(annotations_file:pd.DataFrame, output_file:str, code_file:pd.DataFrame, images_folder:str, access_token:str, qwen_path:str, detection_model:str, processor, llm):
    """_summary_

    Args:
        annotations_file (pd.DataFrame): _description_
        output_file (str): _description_
        code_file (pd.DataFrame): _description_
        images_folder (str): _description_
        access_token (str): _description_
        qwen_path (str): _description_
        detection_model (str): _description_
        processor (_type_): _description_
        llm (_type_): _description_

    Returns:
        _type_: _description_
    """


    # Load the object detection model
    model = YOLO(f"{detection_model}.pt")

    # Loop through the rows
    for idx,row in annotations_file.iterrows():
        # Extract the skill code
        skill_code = int(utils.get_skill_code(row["Image"]))
        # Extract the prompt file and the meta-prompt position
        prompt_file, position = utils.extract_prompt_file(row,code_file)
        # Initialize the list of detected emotions
        detected_emotions = list()
        # Initialize the success_rate
        annotation = ""
        # Check if an emotion rule was asked
        if skill_code in EMOTION_CODES:
            try:
                # Initialize the annotation for this row
                annotation = ""
                # Initialize the boudning bxoes for human instances
                person_boxes = []
                # Extract the emotion rules
                emotion_rules = utils.extract_emotion_rules(prompt_file, position)
                emotions_rules_orig = emotion_rules.copy()
                # Perform YOLO inference 
                results = model(f"{images_folder}/{row['Image']}")
                # Access the results
                for result in results:
                    # Loop through the bounding bxoes
                    for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                        # Get the class name
                        class_name = result.names[int(cls)]
                        # Check if it's a person
                        if class_name=="person":
                            # Get the bounding boxes
                            person_boxes.append(box.cpu().numpy().tolist())
                    # Display results
                    if VERBOSE:
                        print(f"File name: {row['Image']}, persons extracted: {person_boxes}")
                # Case 0 : no human detected
                if len(person_boxes)<1:
                    continue
                # Case 2 : human detected
                else:
                    # Load the image
                    image = cv2.imread(f"{images_folder}/{row['Image']}")
                    # Loop through the bounding boxes
                    for i,(x1,y1,x2,y2) in enumerate(person_boxes):
                        # Get the cropped image
                        cropped_region = image[int(y1):int(y2), int(x1):int(x2)]
                        # Save the temporary image file
                        cv2.imwrite(f"./temp_{i}.png", cropped_region)
                        # Prepare the message
                        message = [
                            {
                                "role": "user", 
                                "content": [
                                    {
                                        "type": "image", 
                                        "image": f"./temp_{i}.png"
                                    }, 
                                    {
                                        "type": "text", 
                                        "text": "Which emotion does this character express ? Possible answers : ['trust', 'disgust', 'anticipation', 'joy', 'surprise', 'fear', 'anger', 'sadness', 'unknown']. Return exactly one word."
                                    }
                                ]
                            }
                        ]
                        # Prepare the text processor
                        text = processor.apply_chat_template(
                            message, 
                            tokenize=False, 
                            add_prompt_generation=True
                        )
                        # Process the inputs
                        image_inputs, _ = process_vision_info(message)
                        # Setup the parameters
                        sampling_params = {"max_new_tokens": 20, "temperature": 0}
                        # Perform inference
                        response = llm.generate(prompt=text, 
                                                image_data=image_inputs,
                                                sampling_params=sampling_params)
                        # Process the response
                        ## Removing caps
                        response = str(response["text"]).lower()
                        # Remove trailing and leading whitespaces, etc
                        response = response.strip()
                        ## Only keeping the emotion text
                        for e in VALID_EMOTIONS:
                            # Check if the emotion is included in the response
                            if e in response:
                                response = e

                        # Append the result to the list
                        detected_emotions.append(response)
                        # Display results
                        if VERBOSE:
                            print(f"Detected emotion: {response}")
                        # Free memory
                        del response, image_inputs, _, text, cropped_region
                        gc.collect()
                        # Free cuda memory if available
                        free_memory()
                        # Delete the temporary file 
                        os.remove(f"./temp_{i}.png")
                    # Initiliaze the successes
                    success = []
                    # Initialize the list of deleted emotions
                    deleted = list()
                    # Loop through the detected emotions
                    for emotion in detected_emotions:
                        if emotion in emotion_rules and emotion not in deleted:
                            # Append to the list of emotions
                            success.append(True)
                            deleted.append(emotion)
                            # Delete from the list of emotions to check
                            emotion_rules.remove(emotion)
                        else:
                            success.append(False)

                    if len(detected_emotions) > 0:
                        # Compute the success rate
                        annotation = round(sum(success)/len(success), 2)
                    else:
                        annotation = ""
                    # Annotate the row
                    annotations_file.at[idx,"Emotion"] = str(annotation)
                    # Display results 
                    if VERBOSE:
                        print(f"Emotions rule for this row: {emotions_rules_orig}")
                        print(f"Detected emotions: {detected_emotions}")
                        print(f"Annotation for this row: {annotation}")
            except:
                # Display error message
                print(f"Error for image: {row['Image']}")
                traceback.print_exc()
    # Save results
    annotations_file.to_excel(output_file, index=False)
    # Free memory
    del model, processor, llm
    free_memory()
    
    return annotations_file

def extract_colors(annotations_file:pd.DataFrame, code_file:pd.DataFrame, output_file:str, device:str, images_folder:str, access_token:str, qwen_path:str, segmentation_model:str, processor, llm):
    """_summary_

    Args:
        annotations_file (pd.DataFrame): _description_
        code_file (pd.DataFrame): _description_
        output_file (str): _description_
        device (str): _description_
        images_folder (str): _description_
        access_token (str): _description_
        qwen_path (str): _description_
        segmentation_model (str): _description_
        processor (_type_): _description_
        llm (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Load the object segmentation model
    seg_model = YOLO(f"{segmentation_model}-seg.pt")

    # Loop through the rows
    for idx,row in annotations_file.iterrows():
        # Extract the skill code
        skill_code = int(utils.get_skill_code(row['Image']))
        # Extract the prompt file and the meta-prompt position
        prompt_file, position = utils.extract_prompt_file(row,code_file)
        # Check if the skill code is included
        if skill_code in COLOR_CODES:
            try:
                # Extract the color rules
                color_rules, objects = utils.extract_color_rules(prompt_file, position)
                # Initializee the list of detected colors
                detected_colors = defaultdict(list)
                # Initialize the success rate
                success = list()
                # Perform inference
                results = seg_model(f"{images_folder}/{row['Image']}")
                # Loop through the results 
                for result in results:
                    # Skip if the mask is empty
                    if result.masks is None: 
                        continue 
                    # Load the image
                    image = np.copy(result.orig_img)
                    # Loop through each detected object
                    for i, c in enumerate(result):
                        # Get the class name
                        label = c.names[c.boxes.cls.tolist().pop()]
                        if label in objects:
                            # Build a binary mask
                            binary_mask = np.zeros(image.shape[:2], np.uint8)
                            # Extract the bounding boxes coordinates
                            x1, y1, x2, y2 = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
                            # Isolate the image with a transparent background 
                            isolated = np.dstack([image, binary_mask])
                            # Cropping the isolated image to the object region
                            isolated = isolated[y1:y2, x1:x2]
                            # Save the temporary result
                            cv2.imwrite(f"./temp_{i}.png", isolated)
                            # Prepare the message
                            message = [
                                    {
                                        "role": "user", 
                                        "content": [
                                            {
                                                "type": "image", 
                                                "image": f"./temp_{i}.png"
                                            }, 
                                            {
                                                "type": "text", 
                                                "text": f"What is the color of this {label} ? Possible answers : ['white', 'black', 'blue', 'purple', 'red', 'green', 'brown', 'pink', 'yellow', 'gray', 'orange', 'unknown']. Return exactly one word."
                                            }
                                        ]
                                    }
                                ]
                            # Prepare the text processor
                            text = processor.apply_chat_template(
                                message, 
                                tokenize=False, 
                                add_prompt_generation=True
                            )
                            # Process the inputs
                            image_inputs, _ = process_vision_info(message)
                            # Setup the parameters
                            sampling_params = {"max_new_tokens": 20, "temperature": 0}
                            # Perform inference
                            response = llm.generate(prompt=text, 
                                                    image_data=image_inputs,
                                                    sampling_params=sampling_params)
                            # Process the response
                            ## Removing caps
                            response = str(response["text"]).lower()
                            # Remove trailing and leading whitespaces, etc
                            response = response.strip()
                            ## Only keeping the colors text
                            for e in VALID_COLORS:
                                # Check if the color is included in the response
                                if e in response:
                                    response = e
                            # Append to the detected colors
                            detected_colors[label].append(response)
                            # Free memory
                            del response, image_inputs, text, _
                            free_memory()
                            # Delete the temporary file
                            os.remove(f"./temp_{i}.png")

                if len(detected_colors)>0:
                    for object, colors in detected_colors.items():
                        # Loop through the detected colors
                        for color in colors:
                            # Check if it complies with the rules
                            if color==color_rules[object]:
                                success.append(True)
                            else:
                                success.append(False)
                # Compute the success rate 
                success_rate = "" if len(detected_colors)<1 else round(sum(success)/len(success),2)
                # Annotate the row
                annotations_file.at[idx,"Colors"] = str(success_rate)
                # Display results 
                if VERBOSE: 
                    print(f"Color rules: {color_rules}")
                    print(f"Detected colors: {detected_colors}")    
                    print(f"Annotation for this line: {success_rate}") 
                
            except Exception:
                # Display error message
                print(f"Error for image: {row['Image']}")
                traceback.print_exc()

    # Save results
    annotations_file.to_excel(output_file, index=False)
    # Free memory
    del seg_model, processor, llm
    free_memory()


    return annotations_file
    


def extract_cohesiveness(annotations_file:pd.DataFrame, output_file:str, images_folder:str, access_token:str, qwen_path:str, processor, llm):
    """_summary_

    Args:
        annotations_file (pd.DataFrame): _description_
        images_folder (str): _description_
        access_token (str): _description_
        qwen_path (str): _description_
    """

    # Loop through the rows
    for idx,row in annotations_file.iterrows():
        try:
            # Prepare the message
            message = [
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "image", 
                            "image": f"{images_folder}/{row['Image']}",
                        }, 
                        {
                            "type": "text", 
                            "text": f"Determine whether the anatomy and structure in the image appear realistic and coherent.\n\
Answer 'true' if there are no obvious anatomical anomalies or structural anomalies.\n\
Answer 'false' only if there are clear and significant errors.\n\
Output exactly one word: true or false."
                        }
                        ]
                }]
            # Prepare the text processor
            text = processor.apply_chat_template(
                message, 
                tokenize=False, 
                add_prompt_generation=True
                )
            # Process the inputs
            image_inputs, _ = process_vision_info(message)
            # Setup the parameters
            sampling_params = {"max_new_tokens": 20, "temperature": 0}
            # Perform inference
            response = llm.generate(prompt=text, 
                                    image_data=image_inputs,
                                    sampling_params=sampling_params)
            # Process the response
            ## Removing caps
            response = str(response["text"]).lower()
            # Remove trailing and leading whitespaces, etc
            response = response.strip()
            ## Only keeping the colors text
            for e in VALID_COHESIVENESS:
                # Check if the color is included in the response
                if e in response:
                    response = e
            # Annotate the row
            annotations_file.at[idx,"Cohesiveness"] = response
            if VERBOSE:
                # Display the result
                print(f"Cohesiveness for {row['Image']} : {response}")
            # Free memory
            del response, image_inputs, text, _
            free_memory()

        except Exception:
            print(f"Error for image: {row['Image']}")
            traceback.print_exc()

    # Save results
    annotations_file.to_excel(output_file, index=False)
    # Free memory
    del llm, processor
    free_memory()

    return annotations_file


def free_memory():
    """_summary_
    """

    gc.collect()
    # Free cuda memory if available
    if torch.cuda.is_available() and torch.version.cuda is not None:
        # Empty cuda cache
        torch.cuda.empty_cache()
        # Collect garbage
        torch.cuda.ipc_collect()


def initialize_parser():
    """
    Initializes the argument parser. 
    """

    # Initializing the parser 
    parser = argparse.ArgumentParser(description="Test script for generating images.")

    # Add arguments 
    parser.add_argument("--model", choices=["all", 
                                            "firelfy", 
                                            "dalle", 
                                            "kandinsky", 
                                            "runway_ml", 
                                            "stable_cascade", 
                                            "stable_diffusion", 
                                            "z_image_turbo", 
                                            "qwen_image", 
                                            "flux"], default="z_image_turbo")
    
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=42, help="The generator seed.")
    parser.add_argument("--object_detection", type=str, choices=["yolo26n", "yolo26s", "yolo26m",
                                                               "yolo26l", "yolo26x", "detectron"], default="yolo26x", help="The method used for object detection.")
    parser.add_argument("--depth_estimation", type=str, choices=["yolo26n", "yolo26s", "yolo26m",
                                                                 "yolo26l", "yolo26x", "detectron"], default="yolo26x", help="The method used for object detection.")
    parser.add_argument("--text_extraction", choices=["paddleocr", "qwen-vl"], type=str, default="qwen-vl", help="The method used for text extraction.")
    parser.add_argument("--emotion_extraction", choices=["small", "large"], type=str, default="large", help="The VLM model used for emotion extraction.")
    parser.add_argument("--annotations_file", default="./data/to_annotate/z_image_turbo_annotations.xlsx", type=str, help="Path to the file to annotate.")
    parser.add_argument("--images_folder", default="./data/to_annotate/z_image_turbo", type=str, help="The folder that contains images to annotate.")
    parser.add_argument("--verbose", default=1, choices=[0,1], type=int, help="The verbosity controls if the extraction results are shown or not.")


    return parser


def main():
    """
    Main script
    """

    # Free memory
    free_memory()

    # Parse arguments 
    parser = initialize_parser()
    # Set the model 
    MODEL = parser.parse_args().model
    # Set the device
    DEVICE = parser.parse_args().device
    # Set the seed 
    SEED = parser.parse_args().seed
    # Get the text extraction method
    text_extraction_method = parser.parse_args().text_extraction
    # Get the file to annotate
    annotations_file = parser.parse_args().annotations_file
    # Set the output file
    output_file = annotations_file
    annotations_file = pd.read_excel(annotations_file)
    # Process the annotations file 
    annotations_file = utils.process_annotations_file(annotations_file)
    # Get the images folder
    images_folder = parser.parse_args().images_folder
    # Get the verbosity
    VERBOSE = parser.parse_args().verbose
    # Get the object detection model
    object_detection_method = parser.parse_args().object_detection
    # Get the depth estimation model
    depth_estimation = parser.parse_args().depth_estimation
    # Load logs 
    LOGS = json.loads(utils.load_logs(f"./evaluation/llm_judge/judge_{MODEL}_seed_{SEED}_logs.json"))
    # Load the code file
    code_file = pd.read_csv("./skills_code.csv", encoding="utf-8")
    # Load the access token
    access_token, QWEN_VL_FP8_PATH = utils.load_environment()
    # Set random seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # Extract the texts
    annotations_file = extract_texts(
        annotations_file=annotations_file, 
        images_folder=images_folder,
        device=DEVICE,
        output_file=output_file,
        text_extraction=text_extraction_method, 
        access_token=access_token
    )

    # Free memory
    free_memory()

    # Extract the objects and their bounding boxes. Return the persons and their bounding boxes ?
    annotations_file = extract_objects_yolo(
        annotations_file=annotations_file,
        code_file=code_file,
        images_folder=images_folder, 
        detection_model=object_detection_method, 
        output_file=output_file,
    )

    # Free memory
    free_memory()

    # Extract the size relationships
    annotations_file = extract_sizes(
        annotations_file=annotations_file,
        code_file=code_file,
        images_folder=images_folder, 
        detection_model=object_detection_method, 
        output_file=output_file,
    )

    # Free memory
    free_memory()

    # Extract the spatial relationships
    annotations_file = extract_spatial_yolo(
        annotations_file=annotations_file,
        code_file=code_file,
        images_folder=images_folder, 
        object_detection=object_detection_method, 
        depth_estimation=depth_estimation, 
        output_file=output_file,
    )

    # Free memory
    free_memory()

    # Loading the processor
    processor = AutoProcessor.from_pretrained(
        QWEN_VL_FP8_PATH,
        trust_remote_code=True,
        token=access_token
    )

    # Setup the vlm
    llm = Engine(
        model_path=QWEN_VL_FP8_PATH,
        enable_multimodal=True,
        mem_fraction_static=0.9,
        tp_size=torch.cuda.device_count(),
        #attention_backend="fa2"
    )

    # Extract the emotions
    annotations_file = extract_emotions(
        annotations_file=annotations_file, 
        code_file=code_file, 
        images_folder=images_folder, 
        access_token=access_token, 
        qwen_path=QWEN_VL_FP8_PATH, 
        detection_model=object_detection_method, 
        output_file=output_file,
        processor=processor, 
        llm=llm
    )

    # Free memory
    free_memory()

    # Extract the colors
    annotations_file = extract_colors(
        annotations_file=annotations_file, 
        code_file=code_file, 
        images_folder=images_folder, 
        device=DEVICE, 
        access_token=access_token, 
        qwen_path=QWEN_VL_FP8_PATH,
        segmentation_model=object_detection_method, 
        output_file=output_file,
        processor=processor, 
        llm=llm
    )

    # Free memory
    free_memory()

    # Extract cohesiveness
    annotations_file = extract_cohesiveness(
        annotations_file=annotations_file, 
        images_folder=images_folder, 
        access_token=access_token, 
        qwen_path=QWEN_VL_FP8_PATH, 
        output_file=output_file,
        processor=processor, 
        llm=llm
    )

    #del llm, processor

    # Free memory
    free_memory()


if __name__=="__main__":
    main()