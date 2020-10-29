# AGAIN-VC

## Demo page
https://kimythanly.github.io/AGAIN-VC-demo/index

## Usage
### Preprocessing
```python preprocess.py [-h] [--config <CONFIG>] [--njobs <NJOBS>] [--dry] [--debug] [--seed <SEED>]```

### Making indexes for training
```python make_indexes.py [-h] [--config <CONFIG>] [--dry] [--debug] [--seed <SEED>]```

### Training
```
python train.py
                [-h] [--config <CONFIG>] [--dry] [--debug] [--seed <SEED>] [--load <LOAD>] [--njobs <NJOBS>] [--total-steps <TOTAL_STEPS>]
                [--verbose-steps <VERBOSE_STEPS>] [--log-steps <LOG_STEPS>] [--save-steps <SAVE_STEPS>] [--eval-steps <EVAL_STEPS>]
```

### Inference
```
python inference.py
                    [-h] [--config <CONFIG>] [--dsp-config <DSP_CONFIG>] --source <SOURCE> --target <TARGET> --output <OUTPUT>
                    [--seglen <SEGLEN>] [--dry] [--debug] [--seed <SEED>] --load <LOAD> [--njobs <NJOBS>]
```

### Pre-trained model
You can download our pre-trained model [here](https://drive.google.com/drive/folders/1qxVVS07VWdp1Kwsf-XI7TyD0fowA7bGp?usp=sharing).
