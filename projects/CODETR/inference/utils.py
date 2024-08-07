# From https://github.com/cortica-iei/AB_AutoTagging_Inference

import pandas as pd

from projects.CODETR.inference.consts import LABELS_MAP, COCO_LABELS_MAP, OUTPUT_COLUMNS


def parse_bbox(x1, y1, x2, y2):
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = (x2 - x1)
    h = (y2 - y1)

    return x, y, w, h


def parse_results(image_id, results_dict):
    results_df = pd.DataFrame(results_dict)

    results_df['name'] = image_id
    results_df['score'] = (results_df['scores'] * 100).astype(int)
    results_df[['x_center', 'y_center', 'width', 'height']] = results_df[
        'bboxes'].apply(lambda bbox_list: pd.Series(parse_bbox(*bbox_list)))
    results_df['label'] = results_df['labels'].apply(
        lambda x: LABELS_MAP.get(int(x)))
    results_df['coco_label'] = results_df['labels'].apply(
        lambda x: COCO_LABELS_MAP.get(int(x)))

    results_df['threshold'] = 0
    results_df['is_occluded'] = 0
    results_df['is_truncated'] = 0
    results_df['d3_separation'] = 0
    results_df['is_rider_on_2_wheels'] = 0

    results_df['l_label'] = None
    results_df['r_label'] = None

    results_df['box_correction'] = None
    results_df['source_res'] = None

    return results_df[OUTPUT_COLUMNS]


def chunks(lst, n=1):
    # Calculate the target size for each part
    target_size = len(lst) // n
    remainder = len(lst) % n

    # Initialize variables
    start_index = 0
    end_index = 0

    # Split the list into parts
    result = []
    for i in range(n):
        # Adjust the end index to account for the remainder
        end_index += target_size + (1 if i < remainder else 0)

        # Extract the part from the list
        part = lst[start_index:end_index]

        # Add the part to the result
        result.append(part)

        # Update the start index for the next part
        start_index = end_index

    return result
