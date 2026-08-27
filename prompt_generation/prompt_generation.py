import random
import numpy as np
from utils.skills import Skills,RobustnessTests

class MetaPromptGeneration:
    def __init__(self, metadata):
        self.levels = ["easy", "medium", "hard"]
        self.data_type = "coco"
        self.metadata = metadata
        self.scene = {}

    def _setup_scene(self, skills, level):
        scene = {}
        if Skills.EMOTION in skills:
            scene = self.apply_emotion(level, scene)
            if skills==[Skills.EMOTION]:
                self.scene = scene
                return scene

        # Every skill but emotion requires objects
        k=0
        match level:
            case "easy": k=2
            case "medium": k=3
            case "hard": k=4
        scene["objects"] = [{"object":x} for x in self.metadata.get_rnd_objects(k=k)]

        if Skills.COUNTING in skills:
            scene = self.apply_counting(level, scene)
        if Skills.COLOR in skills:
            scene = self.apply_color(level, scene)
        if Skills.SPATIAL in skills:
            scene = self.apply_spatial(level, scene)
        if Skills.SIZE in skills:
            scene = self.apply_size(level, scene)
        if Skills.TEXT in skills:
            scene = self.apply_text(level, scene)

        self.scene = scene

        return scene

    def generate_meta_prompt(self, skills, level):

        scene = self._setup_scene(skills, level)

        template = "Create a natural language text description for an image"

        if scene and "objects" in scene:
            template += " that contains the following elements"

            if Skills.COLOR in skills:
                template += " with specified colors"
            if Skills.COUNTING in skills:
                template += " and quantities"

            template += ": "
            for obj in scene["objects"]:
                template += f"{obj}, "
            template = template[:-2] + "."

            if Skills.SPATIAL in skills:
                for o1, relationship, o2 in scene['spatial_relations']:
                    template += f" Position the {o1}(s) {relationship} the {o2}(s)."

            if Skills.SIZE in skills:
                for o1, relationship, o2 in scene['size_relations']:
                    template += f" Make the {o1}(s) {relationship} than the {o2}(s)."

            if Skills.TEXT in skills:
                word_count = scene['text']['word_count']
                objects = ", ".join(scene['text']['objects'])
                template += f" The image must include visible text (on a sign, note, paper, label, or screen) with exactly {word_count} words written about {objects}."
        else:
            template +="."

        if Skills.EMOTION in skills:
            for emotion in scene['emotion']:
                template += f" Include a person showing {emotion}."
        
        # Set a negative prompt
        template += " Output ONLY prompt, no comment, no explanations."

        print("Generated Scene:", scene)
        return template

    def apply_counting(self, level, scene):
        low, high = 0, 0
        k = len(scene['objects'])
        match level:
            case "easy": low = 1; high = 2
            case "medium": low = 2; high = 3
            case "hard": low = 4; high = 5

        # Note that randint high is exclusive
        for i in range(k):
            scene['objects'][i]['count'] = np.random.randint(low=low, high=high+1)
        return scene

    def apply_color(self, _, scene):
        k = len(scene['objects'])
        colors = self.metadata.get_rnd_colors(k=k)
        for i in range(k):
            scene['objects'][i]['color'] = colors[i]
        return scene

    def apply_spatial(self, level, scene):
        # Note that transitivity problems are avoided via the choice of objects.
        if level == "easy":
            # One binary relation
            rel = self.metadata.get_rnd_spatial_relations(1)
            objs = random.sample([obj['object'] for obj in scene['objects']], k=2)
            scene['spatial_relations'] = [(objs[0], rel[0], objs[1])]
            return scene
        elif level == "medium":
            # Two binary relations in 3 objects
            rel = self.metadata.get_rnd_spatial_relations(2)
            objs = random.sample([obj['object'] for obj in scene['objects']], k=3)
            # O0 ~ r1 O1 and O0 ~ r2 O2 to mimic previous code structure
            scene['spatial_relations'] = [(objs[0], rel[0], objs[1]), (objs[0], rel[1], objs[2])]
            return scene
        elif level == "hard":
            # Two binary relations in 4 objects
            rel = self.metadata.get_rnd_spatial_relations(2)
            objs = random.sample([obj['object'] for obj in scene['objects']], k=4)
            # O0 ~ r1 O1 and O2 ~ r2 O3 to mimic previous code structure.
            scene['spatial_relations'] = [(objs[0], rel[0], objs[1]), (objs[2], rel[1], objs[3])]
            return scene

    def apply_size(self, level, scene):
        if level == "easy":
            # One binary relation
            rel = self.metadata.get_rnd_size_relations()
            objs = random.sample([obj['object'] for obj in scene['objects']], k=2)
            scene['size_relations'] = [(objs[0], rel[0], objs[1])]
            return scene
        elif level == "medium":
            # Two binary relations in 3 objects
            rel = self.metadata.get_rnd_size_relations(2)
            objs = random.sample([obj['object'] for obj in scene['objects']], k=3)
            # O0 ~ r1 O1 and O0 ~ r2 O2 to mimic previous code structure
            scene['size_relations'] = [(objs[0], rel[0], objs[1]), (objs[0], rel[1], objs[2])]
            return scene
        elif level == "hard":
            # Two binary relations in 4 objects
            rel = self.metadata.get_rnd_size_relations(2)
            objs = random.sample([obj['object'] for obj in scene['objects']], k=4)
            # O0 ~ r1 O1 and O2 ~ r2 O3 to mimic previous code structure.
            scene['size_relations'] = [(objs[0], rel[0], objs[1]), (objs[2], rel[1], objs[3])]
            return scene

    def apply_emotion(self, level, scene):
        k=0
        match level:
            case "easy": k=1
            case "medium": k=2
            case "hard": k=3

        emotions = self.metadata.get_rnd_emotions(k=k)
        scene['emotion'] = emotions
        return scene

    def apply_text(self,level, scene):
        low, high,k = 0,0,0
        match level:
            case "easy": low = 1; high = 3; k=1
            case "medium": low = 4; high = 6; k=2
            case "hard": low = 6; high = 8; k=2

        word_count = np.random.randint(low=low, high=high+1)
        objects = random.sample([obj['object'] for obj in scene['objects']], k=k)
        scene["text"] = {"word_count": word_count, "objects": objects}

        return scene

class PromptGeneration:
    def __init__(self,llm_interface):
        self.llm_interface = llm_interface

    def generate_prompt(self, meta_prompt, robustness_test=[], k=1):
        """
        Generate prompts using the LLM interface with optional robustness tests.
        Args:
            meta_prompt (str): The meta prompt to be used for generation.
            robustness_test (list): List of robustness tests to apply.
            k (int): Number of prompts to generate.
        Returns:
            list: Generated prompts.
            In the case of robustness, the first prompt is the original, and the rest are modified versions.
        """
        if not robustness_test:
            return [self.llm_interface.generate(meta_prompt) for _ in range(k)]

        elif k==1:
            raise ValueError("k must be greater than 1 when robustness tests are applied.")

        prompt = self.llm_interface.generate(meta_prompt)
        prompts = [prompt]

        template = ""
        if RobustnessTests.TYPOS in robustness_test and RobustnessTests.CONSISTENCY in robustness_test:
            template = "Rewrite the following image prompt in a different way while adding a few small spelling mistakes."

        elif RobustnessTests.TYPOS in robustness_test:
            template = "Add a few small spelling mistakes to the following image prompt."

        elif RobustnessTests.CONSISTENCY in robustness_test:
            template = "Rewrite the following image prompt using different words but keeping exactly the same meaning."

        template = template + " " + " Output ONLY the modified prompt, no explanations:"

        if template:
            prompts.extend([self.llm_interface.generate(template + prompt) for _ in range(k-1)])

        return prompts
