from util.parser import get_parser
from util.config import Config
from util.mytorch import same_seeds
from agent.inferencer import Inferencer
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s | %(filename)s | %(message)s',\
     datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def get_args():
    parser = get_parser(description='Inference')

    # required
    parser.add_argument('--load', '-l', type=str, help='Load a checkpoint.', required=True)
    parser.add_argument('--source', '-s', help='Source path. A .wav file or a directory containing .wav files.', required=True)
    parser.add_argument('--target', '-t', help='Target path. A .wav file or a directory containing .wav files.', required=True)
    parser.add_argument('--output', '-o', help='Output directory.', required=True)

    # config
    parser.add_argument('--config', '-c', help='The train config with respect to the model resumed.', default='./config/train.yaml')
    parser.add_argument('--dsp-config', '-d', help='The dsp config with respect to the training data.', default='./config/preprocess.yaml')
   
    # dryrun
    parser.add_argument('--dry', action='store_true', help='whether to dry run')
    # debugging mode
    parser.add_argument('--debug', action='store_true', help='debugging mode')

    # seed
    parser.add_argument('--seed', type=int, help='random seed', default=961998)

    # [--log-steps <LOG_STEPS>]
    parser.add_argument('--njobs', '-p', type=int, help='', default=4)
    parser.add_argument('--seglen', help='Segment length.', type=int, default=None)

    return parser.parse_args()

if __name__ == '__main__':
    # config
    args = get_args()
    config = Config(args.config)
    same_seeds(args.seed)
    args.dsp_config = Config(args.dsp_config)

    # log some info
    logger.info(f'Config file:  {args.config}')
    logger.info(f'Checkpoint:  {args.load}')
    logger.info(f'Source path: {args.source}')
    logger.info(f'Target path: {args.target}')
    logger.info(f'Output path: {args.output}')

    # build inferencer
    inferencer = Inferencer(config=config, args=args)

    # inference
    inferencer.inference(source_path=args.source, target_path=args.target, out_path=args.output, seglen=args.seglen)
