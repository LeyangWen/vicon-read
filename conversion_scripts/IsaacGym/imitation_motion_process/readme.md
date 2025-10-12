

# Imitation Motion Processing Pipeline

This document describes the full workflow for processing imitation motion data — from raw videos to Isaac Gym–ready motion datasets.

---

## 1. Capture Videos

Record videos of the motions you want to process.
You can use **single-camera** or **multi-camera synchronized** setups.

---

## 2. Create Clips (Trim to Useful Segments)

### **Single-Camera**

If only one camera is used:

* Simply trim the raw video into meaningful segments (each containing a single task or motion).

### **Multi-Camera (Synced)**

If multiple cameras are used and each video contains multiple motion clips:

1. Import the videos into **Adobe Premiere**.
2. Add **markers** to indicate the start and end of each useful clip.
3. Export the marker list as a `.csv` file.
4. Run the following script to automatically split the video into clips:

   ```bash
   python conversion_scripts/IsaacGym/imitation_motion_process/adobe_marker_csv_to_clips.py
   ```
5. If any clips need rotation (e.g., vertical video):

   ```bash
   sh conversion_scripts/IsaacGym/imitation_motion_process/rotate90.sh
   ```

---

## 3. Extract Human Motion Using SMPLest-X

1. Place your trimmed clips inside:

   ```
   demo/imitation_motions/
   ```
2. Run batch inference to extract SMPLest-X poses:

   ```bash
   sh scripts/batch_inference_each.sh
   ```
3. The results will be saved to:

   ```
   demo/result_imitation_motions/
   ```

   with visualization frames stored in:

   ```
   demo/output_frames/
   ```

---

## 4. Convert to Isaac Gym Format (TokenHSI Project)

1. Go to the **TokenHSI** project.

2. Run the preprocessing script to convert SMPLest-X results into Isaac Gym–compatible motion files:

   ```bash
   python tokenhsi/data/dataset_carry/preprocess_smplest.py
   ```

3. Remember to modify the dataset configuration file before running:

   ```
   tokenhsi/data/dataset_carry/config/dataset_cfg_MMH.yaml
   ```

4. The processed human motion files will be saved under:

   ```
   tokenhsi/data/dataset_carry/motions/MMH/
   ```

---

## 5. (Optional) Annotate or Modify Motions in Blender

If you need to:

* Annotate **box locations**
* Adjust **motion trajectories** or **timing**

You can open the generated motion files in **Blender** and follow:

```
tokenhsi/data/dataset_carry/blender/Blender_processing_guide.md
```

---

### Summary of Key Paths

| Stage                 | Directory                                                 | Description                     |
| --------------------- | --------------------------------------------------------- | ------------------------------- |
| Raw videos            | `demo/imitation_motions/`                                 | Input clips for SMPLest-X       |
| Extracted motions     | `demo/result_imitation_motions/`                          | SMPLest-X motion data           |
| Visualization frames  | `demo/output_frames/`                                     | Extracted pose images           |
| Isaac Gym motion data | `tokenhsi/data/dataset_carry/motions/MMH/`                | Final motion data for training  |
| Config file           | `tokenhsi/data/dataset_carry/config/dataset_cfg_MMH.yaml` | Controls preprocessing behavior |

