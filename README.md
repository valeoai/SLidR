# Image-to-Lidar Self-Supervised Distillation for Autonomous Driving Data

Official PyTorch implementation of the method **SLidR**. More details can be found in the paper:

**Image-to-Lidar Self-Supervised Distillation for Autonomous Driving Data**, CVPR 2022 [[arXiv](https://arxiv.org/)]
by *Corentin Sautier, Gilles Puy, Spyros Gidaris, Alexandre Boulch, Andrei Bursuc, and Renaud Marlet*

![Overview of the method](./assets/method.png)

If you use SLidR in your research, please consider citing:
```
@inproceedings{SLidR,
   title = {Image-to-Lidar Self-Supervised Distillation for Autonomous Driving Data},
   author = {Corentin Sautier and Gilles Puy and Spyros Gidaris and Alexandre Boulch and Andrei Bursuc and Renaud Marlet},
   journal = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
   year = {2022}
}
```

## Dependencies

Please install the required required packages. Some libraries used in this project, including MinkowskiEngine and Pytorch-lightning are known to have a different behavior when using a different version; please use the exact versions specified in `requirements.txt`.

## Datasets

The code provided is compatible with [nuScenes](https://www.nuscenes.org/lidar-segmentation) and [semantic KITTI](http://www.semantic-kitti.org/tasks.html#semseg). Put the datasets you intend to use in the datasets folder (a symbolic link is accepted).

## Results

### Few-shot semantic segmentation

Results on the validation set using Minkowski SR-Unet:

Method          |nuScenes<br />lin. probing|nuScenes<br />Finetuning with 1% data|KITTI<br />Finetuning with 1% data
---             |:-:                  |:-:                             |:-:
Random init.    |8.1                  |30.3                            |39.5
PointContrast   |21.9                 |32.5                            |41.1
DepthContrast   |22.1                 |31.7                            |41.5
PPKT            |36.4                 |37.8                            |43.9
SLidR           |**38.8**             |**38.3**                        |**44.6**

### Semantic Segmentation on nuScenes

Results on the validation set using Minkowski SR-Unet:

Method          |1%      |5%      |10%     |25%     |100%
---             |:-:     |:-:     |:-:     |:-:     |:-:
Random init.    |30.3    |47.7    |56.6    |64.8    |74.2
SLidR           |**39.0**|**52.2**|**58.8**|**66.2**|**74.6**

### Object detection on KITTI

Results on the validation set using Minkowski SR-Unet:

Method          |5%      |10%     |20%     
---             |:-:     |:-:     |:-:
Random init.    |56.1    |59.1    |61.6
PPKT            |**57.8**|60.1    |61.2
SLidR           |**57.8**|**61.4**|**62.4**

## Reproducing the results

### Pre-computing the superpixels (required)

Before launching the pre-training, you first need to compute all superpixels on nuScenes, this can take several hours:

```python superpixel_segmenter.py```

### Pre-training a 3D backbone

To launch a pre-training of the Minkowski SR-UNet on nuScenes:

```python pretrain.py```

Weights of the pre-training can be found in the output folder, and can be re-used during a downstream task.

### Semantic segmentation

To launch a semantic segmentation, use the following command:

```python downstream.py --cfg_file="config/semseg_nuscenes.yaml" --pretraining_path="output/pretrain/[...]/model.pt"```

with the previously obtained weights, and any config file. The default config will perform a finetuning on 1% of nuScenes' training set.

To re-evaluate the score of any downstream network, run:

```python evaluate.py --resume_path="output/downstream/[...]/model.pt" --dataset="nuscenes"```

## Acknowledgment

Part of the codebase has been adapted from [PointContrast](https://github.com/facebookresearch/PointContrast).
Computation of the lovasz loss used in semantic segmentation follows the code of [PolarNet](https://github.com/edwardzhou130/PolarSeg).


## License
SLidR is released under the [Apache 2.0 license](./LICENSE).