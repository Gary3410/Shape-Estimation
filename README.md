## Installation

1\) Environment requirements

* Python 3.x
* Pytorch 1.11
* CUDA 9.2 or higher

The following installation guild suppose ``python=3.7`` ``pytorch=1.11`` and ``cuda=10.2``. You may change them according to your system.

Create a conda virtual environment and activate it.
```
conda create -n softgroup python=3.7
conda activate softgroup
```


2\) Clone the following two repositories.
```
git clone https://github.com/thangvubk/SoftGroup.git
git clone https://github.com/dbolya/yolact.git
```
note: Please put the above two projects in the same directory.


3\) Install the dependencies.
```
conda install pytorch cudatoolkit=10.2 -c pytorch
pip install spconv-cu102
pip install -r requirements.txt
pip install opencv-python
pip install pycocotools
pip install PyQt5
pip install opencv-contrib-python==4.5.2.52
pip install pybullet
pip install open3D==0.8.0
```

4\) Install build requirement.

```
sudo apt-get install libsparsehash-dev
```

5\) Setup
```
python setup.py build_ext develop
```
6\) Replace files

The files that need to be replaced are as follows.
```
SoftGroup
├── configs
│   ├── softgroup_s3dis_backbone_fold5.yaml
│   ├── softgroup_s3dis_fold5.yaml
├── data
│   ├── coco.py
│   ├── config.py
├── dataset
│   ├── s3dis
│       ├── downsample.py
│       ├── prepare_data.sh
│       ├── prepare_data_inst.py
│       ├── prepare_data_inst_gttxt.py
├── train_softgroup.py
├── train.py
├── yolact.py

```
## Prepare Data
Prepare your own 3D model files (urdf format)

1\) Generate dataset

Please creates folders as follows.
```
dataset
├── sense_data
│   ├── depth
│   ├── ints_img
│   ├── label
│   ├── label_img
│   ├── rgb_img
│   ├── Stanford3dDataset_v1.2
│       ├── Area_1
│       ├── Area_2
│       ├── Area_3
│       ├── Area_4
│       ├── Area_5
│       ├── Area_6
│   ├── data
│       ├── banana
│       ├── bowl
│       ├── ...


```
Then run:
```
conda activate softgroup
cd create_data
python create_val_dataset.py
```
note: The 3D model path needs to be modified in `create_val_dataset.py`

2\) Prepare 2D instance segmentation label
```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
pip install git+git://github.com/waspinator/coco.git@2.1.0
```
Put `./create_data/create_json.py` to `./cocoapi/PythonAPI`

```
python create_name_list.py
python create_json.py
```
After running the script, you should get two files `instances_train2017.json` and `instances_val2017.json`.

Put the above two files to `./SoftGroup/data/coco/annotations`.

Put the images under the `./dataset/sense_data/rbg_img` folder into `./SoftGroup/data/coco/images`.

3\) Prepare 3D instance segmentation label

Put the ``./dataset/sense_data/Stanford3dDataset_v1.2`` to ``./SoftGroup/dataset/s3dis/`` folder.

Preprocess data
```
cd SoftGroup/dataset/s3dis
bash prepare_data.sh
```
4\) Prepare shape estimation dataset

The generated dataset is stored in the ``./dataset/sense_data/data`` directory
```
cd create_data
python create_scale_obj_dataset.py
```

## Train and Test
1\) Train 3D instance segmentation model
```
python train_softgroup.py --config=softgroup_s3dis_backbone_fold5.yaml
python train_softgroup.py --config=softgroup_s3dis_fold5.yaml
```

2\) Train 2D instance segmentation model

```
python train.py --config=yolact_resnet50_config
```

3\) Train 2D+3D instance segmentation model

First, Replace the following files with the files in the ``replace_files`` directory
```
SoftGroup
├── softgroup
│   ├── data
│       ├── custom.py
│       ├── s3dis.py
│       ├── __init__.py
│   ├── model
│       ├── softgroup.py
│   ├── util
│       ├── optim.py
```
Then run
```
python train_2D+3D.py
```

4\) Train shape estimation model

```
python train_test_cp.py
```

5\) Eval 2D+3D instance segmentation model
```
python eval_test.py
```
6\) Eval overall_pipline
```
python eval_test_scale.py
```
7\) Eval only 2D+shape-estimation
```
python eval_ints_scale.py
```
8\) Eval only 3D+shape-estimation
```
python test_scale.py
```