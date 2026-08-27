import numpy as np

def process_text(text:str):
    """Removes extra spaces and line breaks from text"""

    if isinstance(text,str):
        return text.replace("\n","").strip()
    
    return text


def process_division(text:str):
    """Converts a"""

    if isinstance(text,str):
        # Remove extra spaces and line breaks
        text = text.replace("\n", "").strip()
        # Split the data
        text_split = text.split("/")
        # Check if the format is good
        if len(text_split)==2:
            # Return the division
            part_1 = int(text_split[0])
            part_2 = int(text_split[-1])
            # Compute the value
            text = part_1/part_2
        else:
            return text
    
    return text

def process_decimals(text:str):
    """
    """

    if isinstance(text,str):
        try:
            # Remove extra spaces and line breaks
            text = text.replace("\n", "").strip()
            # Split the data
            text_split = text.split(",")
            # Check if the format is good
            if len(text_split)==2:
                # Return the division
                part_1 = int(text_split[0])
                part_2 = int(text_split[-1])
                # Compute the text
                text = str(part_1)+"."+str(part_2)
                text = float(text)
            else:
                return text
        except:
            return np.nan
    
    return text


def process_counting(text:str):
    """
    """


    if isinstance(text,str):
        # Remove extra spaces and line breaks
        text = text.replace("\n", "").strip()
        # Split the data
        text_split = text.split(",")
        # Check if the format is good
        if len(text_split)>1:
            if text[-1]==";":
                text = text[0:-2]
        else:
            return text
    
    return text