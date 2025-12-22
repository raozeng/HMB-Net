# HMB-Net

## Introduction
HMB-Net: Synergizing Global Genomic Context and Local Sequence Motifs for Robust RNA Methylation Prediction.

## Requirements
To install the required dependencies, please run the following command:

    pip install -r requirements.txt

## Usage
To train and evaluate the model using ResNet on the RMBase dataset, run:

    python HybridMamba/run.py --mode all --model_type resnet \
    --train_data zr/data/train/benchmark/RMBase_m_m6A.fa \
    --test_data zr/data/test/independent/RMBase_m_m6A_Test.fa \
    --model_dir saved_models/resnet/RMBase_m_m6A \
    --result_dir results/resnet/RMBase_m_m6A \
    --epochs 30

## Citation
Citation will be updated upon publication.