import matplotlib.pyplot as plt
import re
from scipy import interpolate
import os
import pandas as pd
import numpy as np


class OLDLogParser():
    def __init__(self, folder, model_name, dataset_name, eval_type, precision_val=60):
        self.log_file_paths = {cls: os.path.join(
            folder, f"eval_{model_name}_{dataset_name}_{eval_type}/output/{cls}/{cls}_log.log") for cls in ["4w", "2w", "peds"]}
        self.precision_val = precision_val

        self.data_pattern = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} INFO\s+'
                                       r'\d+\s+recall: \d+/\d+ = (?P<recall>\d+\.\d+)\s+'
                                       r'precision_loose: (?P<precision_l>\d+\.\d+)\s+'
                                       r'precision_strict: (?P<precision_s>\d+\.\d+)')

        self.delimiter_pattern = re.compile(
            r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} INFO\s+Filtering dets.')

        self.bins = {
            "4w": ["?", "??", "11-14", "14-18", "18-22", "22-27", "27-30", "30-34", "34-39", "39-46", "46-55", "55-68", "68-91", "91-137", "137-273", "273-1920"],
            "2w": ["?", "??", "18-20", "20-23", "23-26", "26-31", "31-37", "37-45", "45-61", "61-92", "92-192", "192-1920"],
            "peds": ["?", "??", "33-37", "37-41", "41-47", "47-55", "55-66", "66-82", "82-110", "110-165", "165-327", "327-1920"],
            # "rider": ["?", "??", "33-37", "37-41", "41-47", "47-55", "55-66", "66-82", "82-110", "110-165", "165-327", "327-1920"]
        }

        self.precision_recall_dict = self.parse_logs()
        self.recalls_dict = self.interpolate_recall()

    def parse_logs(self):
        precision_recall_dict = {}
        for cls, log_file_path in self.log_file_paths.items():
            current_precisions = []
            current_recalls = []
            curve_id = 0
            with open(log_file_path, 'r') as file:
                for line in file:
                    if self.delimiter_pattern.match(line.strip()):
                        if current_recalls and current_precisions:
                            if cls not in precision_recall_dict:
                                precision_recall_dict[cls] = {}
                            if self.bins[cls][curve_id] not in precision_recall_dict[cls]:
                                precision_recall_dict[cls][self.bins[cls][curve_id]] = {
                                    'recalls': current_recalls,
                                    'precisions': current_precisions
                                }

                                current_precisions = []
                                current_recalls = []
                                curve_id += 1

                    else:
                        match = self.data_pattern.search(line)
                        if match:
                            recall = float(match.group('recall'))
                            precision = float(match.group('precision_l'))
                            current_recalls.append(recall)
                            current_precisions.append(precision)
                            if precision == 0.0 and recall == 0.0:
                                current_precisions[-1] = 100

                if current_recalls and current_precisions:
                    precision_recall_dict[cls][self.bins[cls][curve_id]] = {
                        'recalls': current_recalls,
                        'precisions': current_precisions
                    }

        return precision_recall_dict

    def add_pr_curve(self, recalls, precisions, label, color):
        plt.plot(precisions, recalls, label=label, color=color)
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.legend()
        plt.grid(True)

    def interpolate_recall(self):
        recalls_dict = {}
        for cls, bins_data in self.precision_recall_dict.items():
            for bin, data in bins_data.items():
                precision_vals = data['precisions']
                recall_vals = data['recalls']
                if not precision_vals or not recall_vals:
                    continue
                if precision_vals[0] < precision_vals[-1]:
                    precision_vals = precision_vals[::-1]
                    recall_vals = recall_vals[::-1]

                if len(precision_vals) > 1:

                    interp_func = interpolate.interp1d(
                        precision_vals[10:], recall_vals[10:], bounds_error=False, fill_value=(recall_vals[-1], recall_vals[0]))
                    interpolated_recall = interp_func(self.precision_val)
                else:
                    interpolated_recall = 0

                if cls not in recalls_dict:
                    recalls_dict[cls] = {}
                recalls_dict[cls][bin] = interpolated_recall
        return recalls_dict


class NEWLogParser(OLDLogParser):
    def __init__(self, eval_folder, precision_val=50):

        self.log_folders_paths = {cls: os.path.join(
            eval_folder, f"/outputs/{cls}") for cls in ["4w", "2w", "ped"]}

        self.precision_val = precision_val

        self.bins = {
            "4w": ["11-14", "14-18", "18-22", "22-27", "27-30", "30-34", "34-39", "39-46", "46-55", "55-68", "68-91", "91-137", "137-273", "273-1920"],
            "2w": ["18-20", "20-23", "23-26", "26-31", "31-37", "37-45", "45-61", "61-92", "92-192", "192-1920"],
            "ped": ["33-37", "37-41", "41-47", "47-55", "55-66", "66-82", "82-110", "110-165", "165-327", "327-1920"],
            # "rider": ["?", "??", "33-37", "37-41", "41-47", "47-55", "55-66", "66-82", "82-110", "110-165", "165-327", "327-1920"]
        }

        self.precision_recall_dict = self.parse_logs()
        self.recalls_dict = self.interpolate_recall()

    def parse_logs(self):
        precision_recall_dict = {}
        for cls, log_folder_path in self.log_folders_paths.items():

            curve_id = 0

            threshs = np.arange(0, 100, 1)

            for bin in self.bins[cls]:
                fn_bin = pd.read_csv(
                    f"{log_folder_path}/FN/hbin_{bin}.tsv", sep='\t')
                fp_loose_bin = pd.read_csv(
                    f"{log_folder_path}/FP_LOOSE/hbin_{bin}.tsv", sep='\t')
                fp_strict_bin = pd.read_csv(
                    f"{log_folder_path}/FP_STRICT/hbin_{bin}.tsv", sep='\t')
                tp_loose_bin = pd.read_csv(
                    f"{log_folder_path}/TP_LOOSE/hbin_{bin}.tsv", sep='\t')
                tp_strict_bin = pd.read_csv(
                    f"{log_folder_path}/TP_STRICT/hbin_{bin}.tsv", sep='\t')

                prec_arr, rec_arr = [], []

                for thresh in threshs:
                    tp_loose = len(
                        tp_loose_bin[tp_loose_bin["det_score"] > thresh])
                    tp_strict = len(
                        tp_strict_bin[tp_strict_bin["det_score"] > thresh])

                    fp_strict = len(
                        fp_strict_bin[fp_strict_bin["det_score"] > thresh])
                    fp_loose = len(
                        fp_loose_bin[fp_loose_bin["det_score"] > thresh])
                    fn = len(fn_bin) + \
                        len(tp_strict_bin[tp_strict_bin["det_score"] <= thresh])
                    prec = (tp_strict + 1e-5) / (tp_strict + fp_loose + 1e-5)
                    rec = tp_strict / (len(tp_strict_bin) + fn)
                    prec_arr.append(prec*100)
                    rec_arr.append(rec*100)

                if cls not in precision_recall_dict:
                    precision_recall_dict[cls] = {}
                precision_recall_dict[cls][bin] = {
                    'recalls': rec_arr,
                    'precisions': prec_arr
                }

        return precision_recall_dict
