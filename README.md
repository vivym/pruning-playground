# Pruning Playground

## Usage

### 1) Install

```bash
conda create -n pruning python=3.9
conda activate pruning

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

git clone https://github.com/VainF/Torch-Pruning.git
pip install ./Torch-Pruning

git clone https://github.com/vivym/pruning-playground.git
cd pruning-playground
pip install -e .
```

### 2) Generate Pruning Indices

```bash
python tools/train.py validate --config configs/datasets/imagenet.yaml --config configs/trainers/pruning.yaml --config configs/models/resnet50.yaml --model.pruning_stage True

# show FLOPS and #Params
python tools/count_flops.py --config configs/models/resnet50.yaml
python tools/count_flops.py --config configs/models/resnet50.yaml --pruning-strategy CustomIndices
```

Try more configs

```bash
python tools/train.py validate --config configs/datasets/imagenet.yaml --config configs/trainers/pruning.yaml --config configs/models/resnet50.yaml --model.pruning_stage True --model.pruning_ratio 0.5 --model.pruning_indices_path datasets/pruning_indices_resnet50_0.5.pth

# show FLOPS and #Params
python tools/count_flops.py --config configs/models/resnet50.yaml
python tools/count_flops.py --config configs/models/resnet50.yaml --pruning-strategy CustomIndices --pruning-ratio 0.5 --pruning-indices-path datasets/pruning_indices_resnet50_0.5.pth

# Random Pruning
python tools/train.py validate --config configs/datasets/imagenet.yaml --config configs/trainers/pruning.yaml --config configs/models/resnet50.yaml --model.pruning_stage True --model.pruning_strategy Random --model.pruning_ratio 0.4 --model.pruning_indices_path datasets/pruning_indices_resnet50_0.5.pth
```

### 3) Finetune models

```bash
python tools/train.py fit --config configs/datasets/imagenet.yaml --config configs/trainers/finetuning.yaml --config configs/models/resnet50.yaml --model.pruning_strategy CustomIndices

# Mixed Precision (16-bit) Training
python tools/train.py fit --config configs/datasets/imagenet.yaml --config configs/trainers/finetuning.yaml --config configs/models/resnet50.yaml --model.pruning_strategy CustomIndices --trainer.precision 16
```

### 4) Online Training Monitor
[Link](https://wandb.ai/viv/pruning-playground).
