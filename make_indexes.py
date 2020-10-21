import logging
from indexer import get_indexer
from util.parser import get_parser
from util.config import Config


logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def get_args():
    parser = get_parser(description='Make indexes.')

    # config
    parser.add_argument('--config', '-c', default='./config/indexes.yaml')

    # dryrun
    parser.add_argument('--dry', action='store_true', help='whether to dry run')
    # debugging mode
    parser.add_argument('--debug', action='store_true', help='debugging mode')

    # seed
    parser.add_argument('--seed', type=int, help='random seed', default=961998)

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    config = Config(args.config)

    indexer = get_indexer(config)

    indexer.make_indexes(input_path=config.input_path, output_path=config.output_path, 
        split_all=config.split_all, split_train=config.split_train)