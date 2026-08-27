from abc import ABC

class BaseMetadata(ABC):
    def _load_from_file(self, filepath):
        """Load items from a text file, one per line."""
        try:
            with open(filepath, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Warning: {filepath} not found. Using empty list.")
            return []

    def get_rnd_objects(self, k=1):
        raise NotImplementedError("Subclasses must implement get_rnd_objects")

    def get_rnd_colors(self, k=1):
        raise NotImplementedError("Subclasses must implement get_rnd_colors")

    def get_rnd_spatial_relations(self, k=1):
        raise NotImplementedError("Subclasses must implement get_rnd_spatial_relations")

    def get_rnd_size_relations(self, k=1):
        raise NotImplementedError("Subclasses must implement get_rnd_size_relations")

    def get_rnd_emotions(self, k=1):
        raise NotImplementedError("Subclasses must implement get_rnd_emotions")

# TODO
# pascal
# imagenet **
# cifar
