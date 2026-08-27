from .object_detector import Detectron2ObjectDetector
from evaluation.base_evaluator import BaseEvaluator
import os
from utils.skills import Skills
import json

class DetectronEvaluator(BaseEvaluator):
    def __init__(self, cache_dir=os.path.join(os.path.dirname(__file__), "cache")):
        self.object_detector = Detectron2ObjectDetector()
        self.cache_dir = cache_dir

    def evaluate(self, images_path, synthetic_prompts, images_metadata):
        self.object_detector.run_on_images([x for x in os.listdir(images_path)], out_dir=self.cache_dir, save=False)
        return super().evaluate(images_path, synthetic_prompts, images_metadata)

    def run_detectron(self, image_paths):
        results = self.object_detector.run_on_images(image_paths, out_dir=self.cache_dir, save=False)
        output_file = os.path.join(self.cache_dir, "detectron_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        return results

    def evaluate_image(self, image_path, synthetic_prompt, image_metadata):
        skill_set = synthetic_prompt['skill']
        result = []
        if Skills.COUNTING in skill_set:
            result.append(self.evaluate_counting(image_path, image_metadata))
        return result

    def evaluate_counting(self, image_path, image_metadata):
        pass

if __name__ == "__main__":
    # Test run_detectron method
    evaluator = DetectronEvaluator()

    # Example image paths - you can modify these as needed
    test_image_paths = ["data/images/image_1.png"]

    # Run detectron on test images
    results = evaluator.run_detectron(test_image_paths)
    print(f"Detectron results: {results}")
