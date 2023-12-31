from src.crops import CropsProcessor
from PIL import Image
from mmdet.apis import DetInferencer


class Detector(object):
    def __init__(self, crops_config_path, model_config_path, model_weights_path, device):
        self.inferencer = DetInferencer(
            model=model_config_path, weights=model_weights_path)
        self.crops_processor = CropsProcessor(crops_config_path)

    def get_raw_detections(self, image_path):

        image = Image.open(image_path)

        detections = {}

        for crop_name in self.crops.names:
            crop = self.crops_processor.crop_image(image, crop_name)
            detections[crop_name] = self.inferencer(crop)

        return detections

    def combine_raw_detections(self, detections):

        combined_detections = None

        return combined_detections

    def __run__(self, image_path):

        raw_detections = self.get_raw_detections(image_path)

        detections = self.combine_raw_detections(raw_detections)

        return detections
