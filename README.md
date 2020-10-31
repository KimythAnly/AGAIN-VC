# AGAIN-VC

## Demo page
https://kimythanly.github.io/AGAIN-VC-demo/index

## Usage
### Preprocessing
```bash
python preprocess.py [--config <CONFIG>] [--njobs <NJOBS>]

# Example:
python preprocess.py -c config/preprocess.yaml
```
Preprocessing the wavefiles into acoustic features (eg. mel-spectrogram).

### Making indexes for training
```bash
python make_indexes.py [--config <CONFIG>]

# Example
python make_indexes.py -c config/make_indexes.yaml
```
Splitting the train/dev set from the preprocessed features.

### Training
```bash
python train.py 
                [--config <CONFIG>]
                [--dry] [--debug] [--seed <SEED>]
                [--load <LOAD>]
                [--njobs <NJOBS>] 
                [--total-steps <TOTAL_STEPS>]
                [--verbose-steps <VERBOSE_STEPS>] 
                [--log-steps <LOG_STEPS>]
                [--save-steps <SAVE_STEPS>]
                [--eval-steps <EVAL_STEPS>]
                
# Example
python train.py \
  -c config/train_again-c4s.yaml \
  --seed 1234567 \
  --total-steps 100000
```

### Inference
```bash
python inference.py
                    --load <LOAD>
                    --source <SOURCE>
                    --target <TARGET>
                    --output <OUTPUT>
                    [--config <CONFIG>]
                    [--dsp-config <DSP_CONFIG>]
                    [--seglen <SEGLEN>] [--dry] [--debug] [--seed <SEED>]
                    [--njobs <NJOBS>]

# Example
python inference.py \
  -c config/train_again-c4s.yaml \
  -l checkpoints/again/c4s \
  -s data/wav48/p225/p225_001.wav \
  -t data/wav48/p226/p226_001.wav \
  -o data/generated
```

### Pre-trained model
You can download our pre-trained model [here](https://drive.google.com/drive/folders/1qxVVS07VWdp1Kwsf-XI7TyD0fowA7bGp?usp=sharing).
