import cv2
import os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

class Detectron2ObjectDetector:
    def __init__(self,
                 model_cfg_path="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
                 score_thresh=0.5,
                 device="cuda"):

        self.cfg_path = model_cfg_path
        self.score_thresh = score_thresh
        self.device = device

        self.predictor, self.metadata = self.build_predictor()

    def build_predictor(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.cfg_path))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.score_thresh
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.cfg_path)
        cfg.MODEL.DEVICE = self.device
        predictor = DefaultPredictor(cfg)
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        return predictor, metadata

    def run_on_image(self, image_path, out_path="output.png", save=False):
        im = cv2.imread(image_path)
        outputs = self.predictor(im)
        instances = outputs["instances"].to("cpu")

        if save:
            v = Visualizer(
                im[:, :, ::-1],
                metadata=self.metadata,
                instance_mode=ColorMode.IMAGE
            )
            vis = v.draw_instance_predictions(instances)
            cv2.imwrite(out_path, vis.get_image()[:, :, ::-1])
            print(f"saved: {out_path}")

        boxes = instances.pred_boxes.tensor.numpy()
        classes = instances.pred_classes.numpy()
        scores = instances.scores.numpy()
        masks = instances.pred_masks.numpy() if instances.has("pred_masks") else None  # already on CPU

        detections = []
        for i, (cls, score, box) in enumerate(zip(classes, scores, boxes)):
            label = self.metadata.thing_classes[cls]

            mask_avg = None
            if masks is not None and i < len(masks):
                mask_avg = float(masks[i].mean())

            detections.append({
                "class": label,
                "score": float(score),
                "bbox": [float(x) for x in box.tolist()],
                "area": mask_avg,  # fraction of image area covered (0..1)
            })

        return {
            "image": os.path.basename(image_path),
            "detections": detections
        }


    def run_on_images(self, image_paths, out_dir="outputs", save=False):
        os.makedirs(out_dir, exist_ok=True)
        results = []  # list to store dicts of detections

        for p in image_paths:
            out_path = os.path.join(out_dir, os.path.basename(p))
            result = self.run_on_image(p, out_path=out_path, save=save)
            results.append(result)
        return results
