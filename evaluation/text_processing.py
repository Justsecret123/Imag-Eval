import numpy as np

def process_text(text: str):
    """
    Normalize textual content by removing line breaks and trimming
    leading and trailing whitespace.

    This utility is primarily used during annotation post-processing to
    ensure a consistent text representation before evaluation.

    Args:
        text (str): Text to normalize.

    Returns:
        str: Cleaned text with line breaks removed and surrounding
        whitespace stripped. If the input is not a string, it is returned
        unchanged.
    """

    if isinstance(text,str):
        return text.replace("\n","").strip()
    
    return text


def process_division(text: str):
    """
    Convert a string representation of a division into its numerical value.

    The function expects values formatted as `"numerator/denominator"`
    (e.g., `"3/5"`) and returns the corresponding floating-point result.
    Inputs that do not follow this format are returned unchanged.

    Args:
        text (str): Input value to process.

    Returns:
        float | str: Computed division result when the input follows the
        expected format, otherwise the original value.
    """

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

def process_decimals(text: str):
    """
    Convert comma-separated numeric values into floating-point numbers.

    The function expects inputs formatted as `"integer,decimal"`
    (e.g., `"3,14"`) and converts them to their floating-point
    representation (`3.14`). Inputs that do not follow the expected
    format are returned unchanged.

    Args:
        text (str): Input value to process.

    Returns:
        float | str | numpy.nan: Converted floating-point value when the
        input matches the expected format, the original value when no
        conversion is applicable, or `numpy.nan` if the conversion fails.
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


def process_counting(text: str):
    """
    Normalize counting annotations by removing formatting artifacts.

    The function cleans counting annotations generated during the
    evaluation process by removing line breaks, trimming surrounding
    whitespace, and discarding any trailing separator characters.

    Args:
        text (str): Counting annotation string, typically formatted as
            one or more `generated,required` pairs separated by
            semicolons.

    Returns:
        str: Normalized counting annotation. If the input does not match
        the expected format or is not a string, the original value is
        returned unchanged.
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