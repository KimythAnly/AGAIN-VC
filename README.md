# AGAIN-VC

## Demo page
https://kimythanly.github.io/AGAIN-VC-demo/index

## Usage
### Preprocessing
```bash
python preprocess.py [--config <CONFIG>] [--njobs <NJOBS>]
```

### Making indexes for training
```bash
python make_indexes.py [--config <CONFIG>]
```

### Training
```
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
```

### Inference
```
python inference.py
                    --load <LOAD>
                    --source <SOURCE>
                    --target <TARGET>
                    --output <OUTPUT>
                    [--config <CONFIG>]
                    [--dsp-config <DSP_CONFIG>]
                    [--seglen <SEGLEN>] [--dry] [--debug] [--seed <SEED>]
                    [--njobs <NJOBS>]
```

### Pre-trained model
You can download our pre-trained model [here](https://drive.google.com/drive/folders/1qxVVS07VWdp1Kwsf-XI7TyD0fowA7bGp?usp=sharing).
