# Propagation and Attribution of Uncertainty in Medical Imaging Pipelines

This is the repository to the paper accepted at UNSURE at MICCAI 2023.

The paper is available on Arxiv and the UNSURE proceedings:
https://arxiv.org/abs/2309.16831
https://link.springer.com/chapter/10.1007/978-3-031-44336-7_1


The upstream model is written in PyTorch and in the following repository:
https://github.com/LeonhardFeiner/uncertainty_propagation_upstream

The downstream model is written in JAX stored in this repository:

# Usage Overview
To use the pipeline, you have to 
1. Train the upstream model
2. Predict the downstream training and validation and testset using the upstream model
3. Train the downstream model using the stored predictions from the upstream model
4. Predict the results on validation or testset using the downstream model


# Setup
you have to set up 2 environments, one for the upstream model and one for the downstream model
To set up this environment use:

`conda env create -f environment.yml`

## Train
Training the downstream model requires predictions of the upstream model to be stored. These predictions have to be started separately for every accelleration factor the resulting folder has to be renamed by extending the accelleration factor at the end of the name.
Thereafter, the accelleration factors given after --mri_accellerations have to be postfixed to the folder name of path/of/train/subset so it will look like e.g. path/of/train/subset04
these folders must contain the predictions of the MRI reconstructions.

```
python main.py --train --csv_path ~/path/to/fastmri/csv/dir --train_prediction_path path/of/train/subset predictions> --val_prediction_path path/of/val/subset --test_prediction_path path/of/test/subset --mri_accelerations 04,08,16,32,64 --dataset knee --run_name knee_side_mc_mvnd --classifier --model resnet --batch_size 16 --input_distribution mvnd --propagator mc --original_data_path ~/datasets --num_workers 8 --sampling_loss --log_path ~/logs/uncertainty_propagator
```

## Predict

```
python main.py --eval --csv_path ~/path/to/fastmri/csv/dir --train_prediction_path path/of/train/subset predictions> --val_prediction_path path/of/val/subset --test_prediction_path path/of/test/subset --mri_accelerations 04,08,16,32,64 --slice_list 0.2,0.4,0.6,0.8 --dataset knee --subset val1 --load_run_parent_path ~/logs/uncertainty_propagator/knee/ --load_run_name path/to/logs/train/log/folder --multi_augmentation_batch --propagator mc --num_samples 256 --run_name knee_side_recon_sample_mc256_mvnd --classifier --model resnet --input_distribution mvnd --stage_aleatoric --num_workers 8 --log_path ~/logs/uncertainty_propagation
```
 

