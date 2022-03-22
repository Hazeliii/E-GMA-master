from pathlib import Path
import numpy as np
from train_loader_dsec import EventSlicer as myslicer
from train_loader_dsec import Sequences
from loader_dsec import EventSlicer
import h5py
from utils.dsec_utils import flow_16bit_to_float
from utils.dsec_utils import RepresentationType
import imageio


def random_test():
    seq_path = Path('../download_Dsec/test/interlaken_00_b')
    test_timestamp_file = seq_path / 'test_forward_flow_timestamps.csv'
    file = np.genfromtxt(
        test_timestamp_file,
        delimiter=','
    )
    idx_to_visualize = file[:, 2]

    timestamps_images = np.loadtxt(seq_path / 'image_timestamps.txt', dtype='int64')
    image_indices = np.arange(len(timestamps_images))
    timestamps_flow = timestamps_images[::2][1:-1]
    indices = image_indices[::2][1:-1]

    print('idx_to_visualize:')
    print(idx_to_visualize)

    print('image_indices:', image_indices)

    print('timestamps_flow:', timestamps_flow)
    print('indices:', indices)


def load_events(dataset_path: Path, delta_t_ms: int=100):
    train_path = dataset_path / 'train'
    assert dataset_path.is_dir(), str(dataset_path)
    assert train_path.is_dir(), str(train_path)
    assert delta_t_ms == 100

    name_mapper_train = []
    for child in train_path.iterdir():
        print('The child is:', str(child))
        name_mapper_train.append(str(child).split("/")[-1])
        sequence = Sequences(child, delta_t_ms, 15, name_idx=len(name_mapper_train)-1, visualize=False)
        t, x_rect = sequence.get_events(index=0)
        print(t)
        print(x_rect)
        print(t.shape)
        print(x_rect.shape)
        break


def get_flow(flowfile: Path):
    assert flowfile.exists()
    assert flowfile.suffix == '.png'
    flow_16bit = imageio.imread(str(flowfile), format='PNG-FI')
    flow, valid2D = flow_16bit_to_float(flow_16bit)
    return flow, valid2D


if __name__ == '__main__':
    '''
    event_path = Path('../DSEC/train/thun_00_a/events_left/events.h5')
    assert event_path.exists()

    t_start = 49599300523
    t_end = 49599400524

    with h5py.File(str(event_path), 'r') as h5f:
        slicer = EventSlicer(h5f)
        myslicer = myslicer(h5f)
        events = slicer.get_events(t_start, t_end)
        my_events = myslicer.get_events(t_start, t_end)

    print(events['t'].shape)
    print(events['p'].shape)
    print(events['x'].shape)
    print(events['y'].shape)

    print(my_events['t'].shape)
    print(my_events['p'].shape)
    print(my_events['x'].shape)
    print(my_events['y'].shape)

    dataset_path = Path('/media/hazel/Data_files/DSEC')
    # load_events(dataset_path)
    '''
    flow_path = Path('/storage1/wqm/projects/DSEC/train/thun_00_a/flow/forward/000004.png')
    flow, valid2D = get_flow(flow_path)
    print(flow)
    print(valid2D.shape)
