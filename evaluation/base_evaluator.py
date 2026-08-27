from abc import ABC, abstractmethod
import os

class BaseEvaluator(ABC):

    def evaluate(self, images_path, synthetic_prompts, images_metadata):
        results = []
        for img_filename in os.listdir(images_path):
            img_path = os.path.join(images_path, img_filename)
            synthetic_prompt = synthetic_prompts.get(img_filename, None)
            image_metadata = images_metadata.get(img_filename, None)

            results.append(self.evaluate_image(img_path, synthetic_prompt, image_metadata))
        return results

    @abstractmethod
    def evaluate_image(self, image_path, synthetic_prompt, image_metadata) -> list:
        pass
