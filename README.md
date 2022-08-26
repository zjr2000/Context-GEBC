# Context-GEBC
Code for [LOVEU Challenge 2022](https://sites.google.com/view/loveucvpr22/home) (Track 2 Generic Event Boundary Captioning Challenge). Our model directly takes the whole video clip as input and generates a caption for each time boundary parallelly. With this design, the model could learn the context information of each time boundary, thus, the potential boundary-boundary interaction could be modeled. 


Our method achieves a 72.84 score on the test set, and we reach the $2^{nd}$ place in this challenge. The technical report is available [here](https://arxiv.org/abs/2207.01050v1).

## Environment
Our code is adapted from the official implementation of PDVC, please see the original [repo](https://github.com/ttengwang/PDVC) for the environment preparation.

## Data
Using [CLIP](https://github.com/openai/CLIP) to extract frame-level features and [Omnivore](https://github.com/facebookresearch/omnivore) to extract clip-level features. We use [this](https://github.com/zjr2000/Untrimmed-Video-Feature-Extractor) pipeline to extract features. 

Then, put the extracted features under these two folders:
```
data/gebc/features/clip_gebc,
data/gebc/omni_gebc
``` 

You can also directly download the official provided features [here](https://sites.google.com/view/loveucvpr22/home). But, remember to change the ```visual_feature_folder``` and ```feature_dim``` in the config file.


Using [VinVL](https://github.com/microsoft/scene_graph_benchmark) to extract region-level features. The region feature of a video is saved to multiple ```.npy``` files, where each single file contains the region feature of a sampled frame. Merge the feature file paths into  ```video_to_frame_index.json``` in the following format:
```
{
    "video_id": [
        "frame_1_feat.npy",
        "frame_2_feat.npy",
        ...     
    ],
    ...
}
``` 
Then put this file under ```data/gebc/```.

## Usage
### Train
```sh
python train.py --cfg_path ${CONFIG_PATH} --gpu_id ${GPU_ID}
```

### Evaluation
```sh
python eval.py --eval_folder ${EVAL_FOLDER} \
 --gpu_id=${GPU_ID} \
 --eval_caption_file=${VAL_ANNO_FILE} \
 --eval_model_path=save/${eval_folder}/model-best-dvc.pth \
 --eval_transformer_input_type gt_proposals \
 --eval_tool_version 2018_cider \
 --eval_batch_size ${EVAL_BATCHSIZE}
```
We train three models to predict subject, before and after, the corresponding config file and validation file are listed below:

| Type | CONFIG_PATH | VAL_ANNO_FILE|
| :----: | :----: | :----: |
| Subject | cfgs/gebc/gebc_clip_omni_5e5_objq50_subject.yml | data/gebc/valset_highest_f1_subject.json|
| Before | cfgs/gebc/gebc_clip_omni_5e5_objq50_before.yml | data/gebc/valset_highest_f1_before.json|
| After | cfgs/gebc/gebc_clip_omni_5e5_objq50_after.yml | data/gebc/valset_highest_f1_after.json|

## Acknowledgement
This repo is mainly based on [PDVC](https://github.com/ttengwang/PDVC). We thank the authors for their efforts.
