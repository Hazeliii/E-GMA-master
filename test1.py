import argparse
import utils.helper_functions as helper
from utils.logger import Logger
from loader.loader_dsec import DatasetProvider
from pathlib import Path
from torch.utils.data import DataLoader
from utils.dsec_utils import RepresentationType
from model.eraft import ERAFT
from utils.visualization import DsecFlowVisualizer
from test import *


def initialize_tester(type):
    # Warm Start
    if type == 'warm_start':
        return TestRaftEventsWarm
    # Classic
    else:
        return TestRaftEvents


def get_visualizer(args):
    return DsecFlowVisualizer


def test(args):
    save_path = helper.create_save_path(args.test_output)
    print('Storing output in folder {}'.format(save_path))

    # logger
    logger = Logger(save_path)
    logger.initialize_file('test')

    # Dataset
    loader = DatasetProvider(dataset_path=Path(args.dataet), representation_type=RepresentationType.VOXEL,
                             delta_t_ms=100,
                             type='standard',
                             visualize=args.visualize)
    loader.summary(logger)
    test_set = loader.get_test_dataset()

    test_set_loader = DataLoader(test_set, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    model = ERAFT(args, n_first_channels=15)
    model.load_state_dict(torch.load(args.model))

    visualizer = get_visualizer(args)
    test = initialize_tester(args.type)

    my_test = test(model=model, config=None, data_loader=test_set_loader, test_logger=logger, save_path=save_path,
                   visualizer=visualizer, additional_args=None)
    my_test.summary()
    my_test._test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='DSEC', type=str, help='Dataset path')
    parser.add_argument('--model', default='checkpoints/', type=str, help='Stored checkpoint path')
    parser.add_argument('--test_output', default='test_output', type=str, help='The path to store test output')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='The results are visualized if this argument is provided')
    parser.add_argument('--num_workers', default=0, type=int, help='The number of workers for data loading')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--type', default='standard', type=str, help='standard or warm_start')
    args = parser.parse_args()
