import random
from .base_metadata import BaseMetadata

class CocoMetadata(BaseMetadata):
    def __init__(self):
        self.spatial_relations = [
            "above", "below", "next to", "to the left of", "to the right of",
            "in front of", "behind", "inside", "on top of", "under"
        ]
        self.size_relations = [
            "smaller", "larger"
        ]
        self.colors = [
            "red", "blue", "green", "yellow", "black", "white",
            "orange", "purple", "pink", "brown", "gray"
        ]
        self.emotions = [
            "joy", "sadness", "anger", "fear", "disgust", "surprise", "trust", "anticipation"
        ]
        # Load objects from file, excluding "knife". "knife" triggers content filters.
        self.objects = [obj for obj in self._load_from_file("data/COCO/objects.txt") if obj.lower() != "knife"]

    def get_rnd_objects(self, k=1):
        return random.sample(self.objects, k) if self.objects else None

    def get_rnd_colors(self, k=1):
        return random.sample(self.colors, k) if self.colors else None

    def get_rnd_spatial_relations(self, k=1):
        return random.sample(self.spatial_relations, k) if self.spatial_relations else None

    def get_rnd_size_relations(self, k=1):
        return random.sample(self.size_relations, k) if self.size_relations else None

    def get_rnd_emotions(self, k=1):
        return random.sample(self.emotions, k) if self.emotions else None
