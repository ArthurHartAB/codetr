# From https://github.com/cortica-iei/AB_AutoTagging_Inference

import pandas as pd

from projects.CODETR.inference.consts import (LABELS_MAP, OUTPUT_COLUMNS, SCORE_COLUMN_NAME, LABEL_COLUMN_NAME,
                                              HANDLING_HINT_COLUMN_NAME, IMAGE_SHAPE)


def parse_bbox(x1, y1, x2, y2):
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = (x2 - x1)
    h = (y2 - y1)

    return x, y, w, h


def parse_results(image_id, results_dict):
    results_df = pd.DataFrame(results_dict)

    results_df['name'] = image_id
    results_df[SCORE_COLUMN_NAME] = (results_df['scores'] * 100).astype(int)
    results_df[['x_center', 'y_center', 'width', 'height']] = results_df[
        'bboxes'].apply(lambda bbox_list: pd.Series(parse_bbox(*bbox_list)))
    results_df[LABEL_COLUMN_NAME] = results_df['labels'].apply(
        lambda x: LABELS_MAP.get(int(x)))
    results_df[HANDLING_HINT_COLUMN_NAME] = None

    return results_df[list(set(results_df.columns) & set(OUTPUT_COLUMNS))]


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


def output_to_tsv(output_path, results_df, columns=None):
    results_df.to_csv(str(output_path), index=False, sep="\t", columns=columns)


def add_bbox_columns(detections):
    detections['x1'] = detections['x_center'] - detections['width'] / 2
    detections['y1'] = detections['y_center'] - detections['height'] / 2
    detections['x2'] = detections['x_center'] + detections['width'] / 2
    detections['y2'] = detections['y_center'] + detections['height'] / 2


def xyhw_from_xyxy(detections):
    detections['x_center'] = (detections['x1'] + detections['x2'])/2
    detections['y_center'] = (detections['y1'] + detections['y2'])/2
    detections['width'] = (detections['x2'] - detections['x1'])
    detections['height'] = (detections['y2'] - detections['y1'])


def remove_bbox_columns(detections):
    detections.drop(columns=['x1', 'y1', 'x2', 'y2'], inplace=True)


def clip_bboxes(df):

    add_bbox_columns(df)

    clip_condition = (df.x1 < 0) | (df.y1 < 0) | (
        df.x2 > IMAGE_SHAPE[1]) | (df.y2 > IMAGE_SHAPE[0])
    df.loc[df.x1 < 0, 'x1'] = 0
    df.loc[df.y1 < 0, 'y1'] = 0
    df.loc[df.x2 > IMAGE_SHAPE[1], 'x2'] = IMAGE_SHAPE[1]
    df.loc[df.y2 > IMAGE_SHAPE[0], 'y2'] = IMAGE_SHAPE[0]

    df.loc[clip_condition, 'x_center'] = (
        df[clip_condition]['x2'] + df[clip_condition]['x1'])/2
    df.loc[clip_condition, 'y_center'] = (
        df[clip_condition]['y2'] + df[clip_condition]['y1'])/2
    df.loc[clip_condition, 'x_center'] = df[clip_condition]['x2'] - \
        df[clip_condition]['x1']
    df.loc[clip_condition, 'y_center'] = df[clip_condition]['y2'] - \
        df[clip_condition]['y1']

    remove_bbox_columns(df)

    return df


def add_bbox_columns(detections):
    detections.loc[:, 'x1'] = detections['x_center'] - detections['width'] / 2
    detections.loc[:, 'y1'] = detections['y_center'] - detections['height'] / 2
    detections.loc[:, 'x2'] = detections['x_center'] + detections['width'] / 2
    detections.loc[:, 'y2'] = detections['y_center'] + detections['height'] / 2


def clean_bbox_columns(detections):
    return detections.drop(columns=['x1', 'y1', 'x2', 'y2'])
