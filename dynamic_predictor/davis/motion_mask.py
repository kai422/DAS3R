#!/usr/bin/env python
import os
import sys
from time import time
import argparse

import numpy as np
import pandas as pd
from davis2017.evaluation import MaskEvaluation
'''
python motion_mask.py --label_path /home/remote/data/sintel/training/dynamic_label_perfect --results_path /home/remote/project/DyGS/InstantSplat/baselines/3dgs/sintel

python motion_mask.py --label_path /home/remote/data/sintel/training/dynamic_label_perfect --results_path /home/remote/project/DyGS/InstantSplat/data/sintel_pose_dec8_baseline

'''
'''

python /home/kai/monster/monst3r/main/evaluation/sintel/motion_mask.py --label_path /home/kai/monster/monst3r/main/data/sintel/training/dynamic_label_perfect --results_path /home/kai/monster/github/monst3r/results/sintel_pose

 J&F-Mean   J-Mean  J-Recall  J-Decay   F-Mean  F-Recall  F-Decay
 0.414534 0.371022  0.337043      0.0 0.458046  0.437202      0.0

 python /home/kai/monster/monst3r/main/evaluation/sintel/motion_mask.py --label_path /home/kai/monster/monst3r/main/data/sintel/training/dynamic_label_perfect --results_path /home/kai/monster/monst3r/monst3r_assets/results/EqMSeg_fix_monst3r/pred_mask_avg_pred1
 J&F-Mean   J-Mean  J-Recall  J-Decay  F-Mean  F-Recall  F-Decay
 0.558321 0.593482  0.689984      0.0 0.52316  0.529412      0.0

python /home/kai/monster/monst3r/main/evaluation/sintel/motion_mask.py --label_path /home/kai/monster/monst3r/main/data/sintel/training/dynamic_label_perfect --results_path /home/kai/monster/monst3r/monst3r_assets/results/EqMSeg_fix_encoder/50
 J&F-Mean   J-Mean  J-Recall  J-Decay   F-Mean  F-Recall  F-Decay
 0.570488 0.594178  0.629571      0.0 0.546798   0.54213      0.0



 
python evaluation/sintel/motion_mask.py --label_path /home/kai/monster/monst3r/data/sintel/training/dynamic_label_perfect --results_path results/sintel_pose_123456789_from_monst3r_lrx0.2/dynamic_mask_nn

 J&F-Mean   J-Mean  J-Recall  J-Decay   F-Mean  F-Recall  F-Decay
 0.352903 0.378218  0.443561      0.0 0.327589  0.340223      0.0

python evaluation/sintel/motion_mask.py --label_path /home/kai/monster/monst3r/data/sintel/training/dynamic_label_perfect --results_path results/sintel_pose_123456789_from_monst3r_lrx0.2/dynamic_mask_raft

--------------------------- Global results ---------------------------
 J&F-Mean   J-Mean  J-Recall  J-Decay   F-Mean  F-Recall  F-Decay
 0.362136 0.309305  0.252782      0.0 0.414966  0.424483      0.0

'''


seq_list = ["alley_2", "ambush_4", "ambush_5", "ambush_6", "cave_2", "cave_4", "market_2",
            "market_5", "market_6", "shaman_3", "sleeping_1", "sleeping_2", "temple_2", "temple_3"]

time_start = time()
parser = argparse.ArgumentParser()
parser.add_argument('--label_path', type=str, help='Subset to evaluate the results', default='all')
parser.add_argument('--results_path', type=str, help='Subset to evaluate the results', default='all')
args, _ = parser.parse_known_args()


csv_name_global = f'global_results.csv'
csv_name_per_sequence = f'per-sequence_results.csv'

# Check if the method has been evaluated before, if so read the results, otherwise compute the results
csv_name_global_path = os.path.join(args.results_path, csv_name_global)
csv_name_per_sequence_path = os.path.join(args.results_path, csv_name_per_sequence)

print(f'Evaluating sequences...')
# Create dataset and evaluate
dataset_eval = MaskEvaluation(root=args.label_path, sequences=seq_list)
metrics_res = dataset_eval.evaluate(args.results_path)
J, F = metrics_res['J'], metrics_res['F']

# Generate dataframe for the general results
g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
g_res = np.array([final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
                    np.mean(F["D"])])
g_res = np.reshape(g_res, [1, len(g_res)])
table_g = pd.DataFrame(data=g_res, columns=g_measures)
with open(csv_name_global_path, 'w') as f:
    table_g.to_csv(f, index=False, float_format="%.3f")
print(f'Global results saved in {csv_name_global_path}')

# Generate a dataframe for the per sequence results
seq_names = list(J['M_per_object'].keys())
seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
J_per_object = [J['M_per_object'][x] for x in seq_names]
F_per_object = [F['M_per_object'][x] for x in seq_names]
table_seq = pd.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object)), columns=seq_measures)
with open(csv_name_per_sequence_path, 'w') as f:
    table_seq.to_csv(f, index=False, float_format="%.3f")
print(f'Per-sequence results saved in {csv_name_per_sequence_path}')

# Print the results
sys.stdout.write(f"--------------------------- Global results ---------------------------\n")
print(table_g.to_string(index=False))
# sys.stdout.write(f"\n---------- Per sequence results ----------\n")
# print(table_seq.to_string(index=False))
total_time = time() - time_start
sys.stdout.write('\nTotal time:' + str(total_time))
