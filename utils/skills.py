class Skills:
    """
    Enumeration of evaluation skills supported by the IMAG-EVAL benchmark.

    These identifiers are used throughout the benchmark generation,
    annotation, and evaluation pipelines to represent individual
    instruction-following capabilities.

    Attributes:
        COUNTING (str): Object counting evaluation.
        COLOR (str): Color attribution evaluation.
        SPATIAL (str): Spatial relationship evaluation.
        SIZE (str): Relative size relationship evaluation.
        EMOTION (str): Emotion attribution evaluation.
        TEXT (str): Text rendering evaluation.
    """

    COUNTING = "counting"
    COLOR = "color"
    SPATIAL = "spatial"
    SIZE = "size"
    EMOTION = "emotion"
    TEXT = "text"


class RobustnessTests:
    """
    Enumeration of robustness evaluation settings supported by IMAG-EVAL.

    These identifiers define perturbation strategies used to assess the
    robustness of Text-to-Image models under modified instructions.

    Attributes:
        TYPOS (str): Robustness evaluation using spelling perturbations.
        CONSISTENCY (str): Robustness evaluation using semantic or lexical
            consistency checks.
    """

    TYPOS = "typos"
    CONSISTENCY = "consistency"