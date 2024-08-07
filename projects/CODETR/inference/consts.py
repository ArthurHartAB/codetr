# From https://github.com/cortica-iei/AB_AutoTagging_Inference

CROP_H_POS = 1920 / 2  # + 300  # + 100
CROP_H = 600

BORDER_DELTA = 10
INTERSECTION_DELTA = 10

# LABELS_MAP = {0: "2w", 1: "4w", 2: "ped", 3: "rider"}
LABELS_MAP = {0: "2w", 1: "4w", 2: "ped", 3: "rider",
              4: "4w", 5: "4w"}  # last two : "van", "truck"

COCO_LABELS_MAP = {0: "BICYCLE", 1: "CAR", 2: "PEDESTRIAN", 3: "RIDER"}

OUTPUT_COLUMNS = ['name', 'label', 'score', 'coco_label', 'x_center', 'y_center',
                  'width', 'height', 'is_occluded', 'is_truncated',
                  'd3_separation', 'l_label', 'r_label', 'is_rider_on_2_wheels',
                  'box_correction']
