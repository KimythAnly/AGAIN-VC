from util.parser import get_parser
from util.config import Config
from util.mytorch import same_seeds
from inferencer import Inferencer
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_args():
    parser = get_parser(description='Inference')

    # config
    parser.add_argument('--config', '-c', help='The train config with respect to the model resumed.', default='./config/train.yaml')
    parser.add_argument('--dsp-config', '-d', help='The dsp config with respect to the training data.', default='./config/preprocess.yaml')

    # 
    parser.add_argument('--source', '-s', help='Input source wavefile.', required=True)
    parser.add_argument('--target', '-t', help='Input target wavefile.', required=True)
    parser.add_argument('--output', '-o', help='Output wavefile.', required=True)
    parser.add_argument('--seglen', '-l', help='Segment length.', type=int, default=None)


    # dryrun
    parser.add_argument('--dry', action='store_true', help='whether to dry run')
    # debugging mode
    parser.add_argument('--debug', action='store_true', help='debugging mode')

    # seed
    parser.add_argument('--seed', type=int, help='random seed', default=961998)

    parser.add_argument('--load', '-l', type=str, help='', default='')
    parser.add_argument('--njobs', '-p', type=int, help='', default=4)

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    config = Config(args.config)
    args.dsp_config = Config(args.dsp_config)

    same_seeds(args.seed)
    inferencer = Inferencer(config=config, args=args)
    inferencer.load_wav_data()
    inferencer.inference()
