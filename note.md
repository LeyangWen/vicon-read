# Note


## Mesh compare evaluation
2024-12-17

### Working in [this](conversion_scripts/MB_np_smpl_to_angles_3DSSPP.py)
- Frame number is not lining up
  - GT 6D mesh file: 287072 test, 277948 validate
    - After clip by source name: 277992 test
  - 6D mesh result: 286528 test
- Also, 3D pose is in 50 fps, while 6D mesh is 20 fps

## RTMPose 2D Evaluate
2025-01-09

### Selected metric: normalized_MPJPE, PCK (circle)

