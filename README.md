# yolov7_traffic_light_detect

## Installation
Create new environment in anconda: 

```anconda
 conda create -n name python=version 
 conda activate name 
 pip install -r requirements.txt 
```


## Testing 

We have three weights in first stage and one weight in second stage.<br>
[Download weights!](https://drive.google.com/drive/folders/1T3CyOxqVJ_0Ip9x9dUU6hf9qo6b485uD?usp=drive_link)

```python
## First Stage
# Use a weight:
python test_TL.py --data data/TL_3cls.yaml --weights (your_weights_path)/FristStage_model1_3cls.pt --img-size 1280 --batch-size 8 --task test  --name (dir_name)

# Use three weights(ensemble learning):<br>
python test_TL.py --data data/TL_3cls.yaml --weights (your_weights_path)/FristStage_model1_3cls.pt (your_weights_path)/FristStage_model2_3cls.pt (your_weights_path)/FristStage_model3_3cls.pt --img-size 1280 --batch-size 8 --task test --no-trace --name (dir_name)<br>

## Second Stage
# You need to crop the TL from First Stage result and label it as 7 cls.
python test_TL.py --data data/TL_7cls.yaml --weights (your_weights_path)/SecondStage_7cls.pt --img-size 640 --batch-size 8 --task test  --name (dir_name)<br>
```
## Inference 
```python
python detect_twonet.py --site-weights (your_weights_path)/FristStage_model1_3cls.pt (your_weights_path)/FristStage_model2_3cls.pt (your_weights_path)/FristStage_model3_3cls.pt --state-weights (your_weights_path)/SecondStage_7cls.pt --source (your_test_img_path)/images --img-size 1280 --name (dir_name) --no-trace
```
## Training


```python
python train.py --data data/TL_3cls.yaml --batch-size 8 --epoch 400 ---weights '' --cfg cfg/training/yolov7.yaml --name (dir_name)
```
