# SMPT -- Pre-Training Molecular Representation Model with Spatial Geometry for Property Prediction

## Description 
SMPT a molecular representation model, combining a spatial information based three-level network and self- supervised learning.

## Dependencies
To install the necessary dependencies for the SMPT model, you will need to have 'paddle' and 'pahelix' packages. You can install these  packages using the following commands:
```bash
pip install paddle
pip install pahelix
```
## Pre-training the Model
The pre-training instructions for the SMPT are provided in the 'pretrain.sh' script. To start the pre-training process, you can use the command:
```bash
bash pretrain.sh
```

## Fine-tuning the Model
For the fine-tuning the model on the downstream tasks, refer to the instructions in the 'finetune_class.sh', you can use the command:
```bash
bash finetune_class.sh
```
###  Data Formating for downstream tasks
Before starting training for downstream tasks, ensure to format the data correctly. Set the 'task' parameter in sh script to 'data' to save the cached data for training.
### Training for donwnstream tasks
Once the cached data saved, set the 'task' parameter to 'train' in 'finetune_class.sh'.
