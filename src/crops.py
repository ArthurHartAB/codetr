import numpy as np
import cv2
import yaml


class CropsProcessor(object):
    def __init__(self, config_path):
        self.parse_config(config_path)

    def parse_config(self, config_path):

        with open(config_path, "r") as file:
            data = yaml.load(file, Loader=yaml.FullLoader)

        self.crops = data['crops']

        self.names = list(self.crops.keys())

        self.image_height = data['image_height']
        self.image_width = data['image_width']

        self.intersection_delta = data['intersection_delta']

    def crop_image(self, image, crop_name):

        image = np.array(image)
        x, y, h, w = self.crops[crop_name]['x'], \
            self.crops[crop_name]['y'], \
            self.crops[crop_name]['h'], \
            self.crops[crop_name]['w']

        x1, y1, x2, y2 = int(x - w), int(y - h), \
            int(x + w), int(y + h)

        crop = image[x1:x2, y1:y2]

        return crop

    def draw_crop_map(self):

        backgound = np.ones([self.image_height, self.image_width, 3])

        image = backgound

        for crop_name in self.names:
            x, y, h, w = self.crops[crop_name]['x'], \
                self.crops[crop_name]['y'], \
                self.crops[crop_name]['h'], \
                self.crops[crop_name]['w']

            x1, y1, x2, y2 = int(x - w), int(y - h), \
                int(x + w), int(y + h)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 3)

        return image

    def crop_to_bbox_distance(self, crop_name, bbox):

        return 0
