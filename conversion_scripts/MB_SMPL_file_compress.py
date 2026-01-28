import pickle
import numpy as np
from numpy.lib.format import open_memmap
import gc

PKL = "/Volumes/Z/RTMPose/37kpts_rtmw_v5/20fps/mesh_compare/SMPL_RTM17kpts_V1/h36m_results.pkl"
OUT = "/Volumes/Z/RTMPose/37kpts_rtmw_v5/20fps/mesh_compare/SMPL_RTM17kpts_V1/h36m_results_markers.npy"

# 49 marker vertex indices (order matches name_list)
idx_list = np.array(
    [411, 3941, 517, 335, 829, 1305, 3171, 3024, 3077, 5246, 6573, 1781, 3156,
     4721, 5283, 5354, 1238, 2888, 1317, 5090, 5202, 1621, 1732, 5573, 5568,
     2112, 2108, 5595, 5636, 2135, 2290, 5257, 4927, 1794, 1454, 4842, 4540,
     1369, 1054, 6832, 6728, 3432, 3327, 6739, 6745, 3337, 3344, 6786, 3387],
    dtype=np.int64
)

CHUNK = 2000  # frames per chunk; adjust (1000-5000) based on RAM / speed

with open(PKL, "rb") as f:
    d = pickle.load(f)  # unavoidable

verts = np.asarray(d["verts"]).reshape(-1, 6890, 3)
del d
gc.collect()

T = verts.shape[0]
P = idx_list.shape[0]

out = open_memmap(OUT, mode="w+", dtype=np.float32, shape=(T, P, 3))

for s in range(0, T, CHUNK):
    e = min(T, s + CHUNK)
    out[s:e] = verts[s:e][:, idx_list, :].astype(np.float32, copy=False)
    if (s // CHUNK) % 20 == 0:
        print(f"wrote {s}:{e} / {T}")

out.flush()

del out, verts
gc.collect()

print("Saved:", OUT)
print("Load with: np.load(OUT, mmap_mode='r')")