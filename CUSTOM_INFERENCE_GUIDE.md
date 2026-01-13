# Custom Inference Guide

This guide shows you how to use your own point cloud data and custom language instructions for grasp generation.

## Quick Start

### 1. Basic Usage

**Simplest way (use latest checkpoints automatically):**
```bash
python infer_custom.py \
  --mesh_path "data/oakink/shape/OakInkObjectsV2/025_mug/align/textured_simple.obj" \
  --guidance "grasp the mug handle, one finger one the top" \
  --cate_id "mug" \
  --action_id "0001"
```

### 2. Parameters

- `--mesh_path`: Path to your 3D mesh file (.obj or .ply)
  - The script will automatically sample a point cloud from the mesh for inference
  - The original mesh will be used for visualization
  - Example: `data/oakink/shape/OakInkObjectsV2/025_mug/align/textured_simple.obj`
- `--guidance`: Natural language instruction (e.g., "grasp the handle to use it")
- `--cate_id`: Object category from the following list:
  ```
  apple, banana, binoculars, bottle, bowl, cameras, can, cup,
  cylinder_bottle, donut, eyeglasses, flashlight, fryingpan, gamecontroller,
  hammer, headphones, knife, lightbulb, lotion_pump, mouse, mug, pen,
  phone, pincer, power_drill, scissors, screwdriver, squeezable, stapler,
  teapot, toothbrush, trigger_sprayer, wineglass, wrench
  ```
- `--action_id`: Intention type
  - `0001`: use (functional grasp for using the object)
  - `0002`: hold (stable grasp for holding)
  - `0003`: lift (grasp for lifting)
  - `0004`: hand_over (grasp for handing over to another person)
- `--idgc_checkpoint`: IDGC checkpoint (optional, defaults to "latest")
  - `"latest"`: automatically finds the latest checkpoint (default)
  - `"200"`: finds epoch 200 checkpoint
  - `"./Experiments/idgc/"`: searches in this directory for latest
  - Full path: `"./Experiments/idgc/epoch200_minus_loss_-2.5936_latest.pth"`
- `--qgc_checkpoint`: QGC checkpoint (optional, defaults to "latest")
  - Same options as idgc_checkpoint
- `--save_dir`: Output directory for results

### 3. Output Structure

After inference, you'll find:

```
custom_inference_results/
├── idgc_grasp_1.obj          # IDGC result 1 (coarse)
├── idgc_grasp_2.obj          # IDGC result 2 (coarse)
├── ...
├── qgc_refined_grasp_1.obj   # QGC refined result 1 (high quality)
├── qgc_refined_grasp_2.obj   # QGC refined result 2 (high quality)
├── ...
└── inference_results.json    # Numerical results (28-DOF poses)
```


