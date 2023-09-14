import SMPL
import numpy as np

smpl_object = SMPL(model_dir=r'C:\Users\Public\Documents\Vicon\vicon_coding_projects\MotionBERT\data\mesh\SMPL_NEUTRAL.pkl')
smpl_output = smpl_object.forward(beta=np.zeros(10), theta=np.zeros(72))
joints = smpl_output.joints
vertices = smpl_output.vertices



