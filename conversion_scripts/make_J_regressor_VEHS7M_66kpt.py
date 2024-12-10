import numpy as np
import os
import argparse
import yaml

# J_regressor for SMPL mesh

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skeleton_file', type=str, default=r'config/VEHS_ErgoSkeleton_info/Ergo-Skeleton-66.yaml')
    parser.add_argument('--output_npy', type=str, default=r'J_regressor_VEHS7M_66kpt.npy')

    args = parser.parse_args()
    with open(args.skeleton_file, 'r') as stream:
        data = yaml.safe_load(stream)
        args.marker_vids = data['marker_vids']
    return args

if __name__ == '__main__':
    args = parse_args()
    smpl_vids = args.marker_vids['smpl']


    paper_custom_6D_joint_names = ['RPSIS', 'RASIS', 'LPSIS', 'LASIS', 'C7_d', 'SS', 'T8', 'XP', 'C7', 'HDTP', 'MDFH', 'REAR', 'LEAR', 'RAP', 'RAP_f', 'RLE', 'RAP_b', 'RME', 'LAP', 'LAP_f', 'LLE', 'LAP_b', 'LME',
                                'LUS', 'LRS', 'RUS', 'RRS', 'RMCP5', 'RMCP2', 'LMCP5', 'LMCP2', 'LGT', 'LMFC', 'LLFC', 'RGT', 'RMFC', 'RLFC', 'RMM', 'RLM', 'LMM', 'LLM', 'LMTP1', 'LMTP5', 'LHEEL',
                                'RMTP1', 'RMTP5', 'RHEEL',
                                   'HEAD', 'RSHOULDER', 'LSHOULDER', 'THORAX', 'LELBOW', 'RELBOW', 'RWRIST', 'LWRIST', 'RHAND', 'LHAND', 'PELVIS', 'RHIP', 'RKNEE',
                                    'RANKLE', 'RFOOT', 'LHIP', 'LKNEE', 'LANKLE', 'LFOOT']  # 66: drop c7_m, add MDFH, V2

    kpt_sequence = paper_custom_6D_joint_names

    joint_calculation = {'center':{'HEAD': ('LEAR', 'REAR'),
                                    'RSHOULDER': ('RAP_f', 'RAP_b'), 'LSHOULDER': ('LAP_f', 'LAP_b'),
                                    'RELBOW': ('RME', 'RLE'), 'LELBOW': ('LME', 'LLE'),
                                    'RWRIST': ('RRS', 'RUS'), 'LWRIST': ('LRS', 'LUS'),
                                    'RHAND': ('RMCP2', 'RMCP5'), 'LHAND': ('LMCP2', 'LMCP5'),
                                    'PELVIS': ('RASIS', 'LASIS', 'RPSIS', 'LPSIS'),
                                    'RKNEE': ('RLFC', 'RMFC'), 'LKNEE': ('LLFC', 'LMFC'),
                                    'RANKLE': ('RLM', 'RMM'), 'LANKLE': ('LLM', 'LMM'),
                                    'RFOOT': ('RMTP1', 'RMTP5'), 'LFOOT': ('LMTP1', 'LMTP5'),

                                   },
                         'translate': {'THORAX': (('SS', 'C7_d'), 9/150),  # offset 9 mm
                                       'RHIP': (('RGT', 'LGT'), 25.4/380),  # offset 1 inch
                                       'LHIP': (('LGT', 'RGT'), 25.4/380),  # offset 1 inch
                                       }
                         }



    J_regressor = np.zeros((len(kpt_sequence), 6890), dtype=np.float32)
    for j, joint_name in enumerate(kpt_sequence):
        if joint_name in smpl_vids:  # surface kpts
            J_regressor[j, smpl_vids[joint_name]] = 1
        elif joint_name in joint_calculation['center']:  # joint center from 2 or more surface kpts
            surface_kpt_number = len(joint_calculation['center'][joint_name])
            for surface_kpt_name in joint_calculation['center'][joint_name]:
                J_regressor[j, smpl_vids[surface_kpt_name]] = 1 / surface_kpt_number
        elif joint_name in joint_calculation['translate']:  # translate kpts from 2 surface kpts
            surface_kpt_names, ratio = joint_calculation['translate'][joint_name]
            J_regressor[j, smpl_vids[surface_kpt_names[0]]] = 1-ratio
            J_regressor[j, smpl_vids[surface_kpt_names[1]]] = ratio
    np.save(args.output_npy, J_regressor)

