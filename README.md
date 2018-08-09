## Talk The Walk: Navigating New York City Through Grounded Dialogue
This repository provides code for reproducing experiments
of the paper [Talk The Walk: Navigating New York City Through Grounded Dialogue](https://arxiv.org/abs/1807.03367)
by Harm de Vries, Kurt Shuster, Dhruv Batra, Devi Parikh, Jason Weston, and Douwe Kiela.

## Getting started

### (1) Setup the environment
First, create a conda environment with installed dependencies:
```bash
conda env create -f environment.yml
source activate ttw

# run your experiments

deactivate # Exit virtual environment
```

In case you get the error "no module named ttw" when running one of the experiments, please add the main directory to your python path:
```bash
export PYTHONPATH=/path/to/talkthewalk:$PYTHONPATH
```

### (2) Download the data
Download the data by running the following bash script:
```bash
sh scripts/download_data.sh DATA_DIR
```
where DATA_DIR specifies the directory where the data files will be downloaded to (defaults to ./data).

### (3) Run experiments
For all experiments, the data directory can be specified through the ```--data-dir``` argument.

### Experiment directory
Create a directory to store your experiments (logs and model checkpoints), for instance by ```mkdir exp``` to
create an experiment directory in the main talkthewalk folder. By default, experiments will be saved to ```./exp```
but you can change the experiment directory via the ```--exp-dir``` flag. The results of each experiment will be saved
in this directory under the experiment name, specified via ```--exp-name```.

#### Running emergent language experiments
To reproduce tourist location via discrete communication, run the following command to train the tourist and guide models:
```bash
python ttw/train/predict_location_discrete.py \
    --vocab-sz 500 --apply-masc \
    --T 1 --exp-name discrete_masc_T1 \
    --num-epochs 200 --cuda
```
For continuous communication, run:
```bash
python ttw/train/predict_location_continuous.py \
    --vocab-sz 500 --apply-masc \
    --T 1 --exp-name continuous_masc_T1 \
    --num-epochs 200 --cuda
```

#### Running natural language experiments
First, create a dictionary:
```bash
python ttw/dict.py --data-dir DATA_DIR
```
which will save the dictionary to ```DATA_DIR/dict.txt```.

Next, train the tourist with imitation learning:
```bash
python ttw/train/train_tourist.py \
    --exp-name tourist_imitation \
    --exp-dir EXP_DIR --cuda
```

To train a guide (from scratch) to perform location prediction from generated tourist utterances, run:
```bash
python ttw/train/predict_location_generated.py \
    --tourist-model EXP_DIR/tourist_imitation/tourist.pt \
    --decoding-strategy greedy --trajectories all \
    --train-guide --T 0 --cuda
```
where ```--trajectories all``` indicates to train on random walk trajectories of length ```--T```. If ```--trajectories human```, then the model will be trained
on human trajectories of the Talk The Walk dataset.

To optimize the tourist generation (with RL) in conjunction with a pre-trained guide: first, pretrain the guide:
```bash
python ttw/train/predict_location_language.py \
    --last-turns 1 \
    --exp-name guide_imitation \
    --apply-masc --T 3 --cuda
```
next, train the tourist with the following command:
```bash
python ttw/train/predict_location_generated.py \
    --tourist-model EXP_DIR/tourist_imitation/tourist.pt \
    --guide-model EXP_DIR/guide_imitation/guide.pt \
    --decoding-strategy sample --train-tourist --cuda
```

#### Evaluating on full task
For discrete comm, the command will be of the following form:
```bash
python scripts/evaluate_location.py \
    --tourist-model TOURIST_CHECKPOINT \
    --guide-model GUIDE_CHECKPOINT \
    --communication discrete --cuda
```

For natural language communication, run:
```bash
python scripts/evaluate_location.py \
    --tourist-model TOURIST_CHECKPOINT \
    --guide-model GUIDE_CHECKPOINT \
    --communication natural \
    --decoding-strategy greedy --cuda
```

#### Running landmark classification experiments
If you want to run experiments using fasttext features, please install fastText via anaconda's pip of the ttw environment
(follow instructions [here](https://github.com/facebookresearch/fastText/tree/master/python)). Next, download
pretrained vectors [wiki.en.zip](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip), extract wiki.en.bin and put the file into the data directory.

To run landmark classification with text recognition features, use the following command:
```bash
python ttw/train/classify_landmarks.py --textrecog-features
```

## License

Talk the Walk is CC-BY-NC licensed, as found in the LICENSE file.
