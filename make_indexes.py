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

    return parser.parse_args()

if __name__ == '__main__':
    # config
    args = get_args()
    config = Config(args.config)

    # build indexer
    indexer = get_indexer(config)

    # make indexes
    indexer.make_indexes(input_path=config.input_path, output_path=config.output_path, 
        split_all=config.split_all, split_train=config.split_train)