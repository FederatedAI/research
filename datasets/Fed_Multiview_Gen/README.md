# Fed_Multiview_Gen
This repo contains code for generating multiview images from 3D CAD models for federated learning research.

Main contributions of this repo:
1. Modified phong.blend for better image quality
2. Scripts for automating the process of the dataset generation
3. Scripts for post-processing images 



## Requirements
- CAD models can be found here: https://github.com/lmb-freiburg/orion
- Convert CAD model to images, Windows version: https://github.com/zeaggler/ModelNet_Blender_OFF2Multiview
- Convert CAD model to images, Linux version: https://github.com/WeiTang114/BlenderPhong

Please refer to above github repos for the installation.

## Usage
There are two steps for generating the dataset:
1. Generate png images from 3D CAD model using Blender;
2. Post-process images from step 1. to adjust object to image ratio.

Firstly, specify the BLENDER_PATH in main.py：

```python
BLENDER_PATH = "D:/Program Files/blender-2.79b-windows64/blender.exe"
```

Then generate the dataset with command:
```bash
python --model_dir ./dataset_samples --target_dir ./dataset_images --action all
```
  
