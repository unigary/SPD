# Self-Predictive Dynamics for Generalization of Vision-based Reinforcement Learning

## Installation
We assume you have access to a gpu that can run CUDA 11.0.
All of the dependencies are in the `spd.yaml` file. They can be installed manually or with the following command:
```
conda env create -f spd.yaml
```

And then you can activate your environment with
```
conda activate spd
```

## Running the IdealGas, Natural Video setting
#### Simple distractor (IdealGas) setting
You can create a simple distractor (IdealGas) video with: https://github.com/unigary/SPD/blob/master/distractor/render_n_body_problem_envs.py

#### Natrual Video setting
You can download the Kinetics 400 dataset and grab the driving_car label from the train dataset to replicate our setup. Some instructions for downloading the dataset can be found here: https://github.com/Showmax/kinetics-downloader.

After downloading each video, modify `config_spd.yaml` based on the path of the downloaded videos.

## Instructions
To train a SPD agent on the `walker walk` task from image-based observation
```
python train.py env=walker_walk
```

To reproduce the results from the paper run
```
python train.py env=walker_walk batch_size=128 action_repeat=2
```

## Logging
This will produce the `runs`folder, where all the outputs are going to be stored including train/eval logs


### Train
In your console, you should see printouts that look like this:
```
| train | E: 1 | S: 500 | R: 1.0 s | D: 1.0000 | A_LOSS: -1 | CLOSS: 2.3 | TLOSS: 1.0 | TVAL: 1.0 | AENT: 1.0
```

### Test
```
| eval | E: 10 | S: 5000 | R: 10.0
```
