import numpy as np
import random
from pathlib import Path
import shutil

'''
test sequences中没有ground truth， 则从每个train sequence中随机选取20%作为test
'''

if __name__ == '__main__':
    train_dataset = Path('/media/hazel/Data_files/DSEC/train')
    test_dataset = Path('/media/hazel/Data_files/DSEC/test')
    assert train_dataset.is_dir()
    assert test_dataset.is_dir()
    num_sequence = 0

    for child in train_dataset.iterdir():
        num_sequence += 1
        print('\n---------------------%s-----------------------' % (str(child.stem)))
        flow_dir = child / 'flow'
        assert flow_dir.is_dir()
        time_file = flow_dir / 'timestamps.txt'
        assert time_file.is_file()
        ev_flow_dir = flow_dir / 'forward'
        assert ev_flow_dir.is_dir()
        # 读取flow map文件地址
        flow_map_list = list()
        for map in ev_flow_dir.iterdir():
            assert str(map.name).endswith('.png')
            flow_map_list.append(str(map.name))
        flow_map_list.sort(key=lambda x:int(x.split('.')[0]))

        # timestamps读取
        file = np.genfromtxt(
            time_file,
            delimiter=','
        )
        start_times = file[:, 0]
        end_times = file[:, 1]

        num_flows = len(file)
        assert num_flows == len(flow_map_list)
        print('The %s has %d flows'%(str(child.stem), num_flows))
        num_test = int(num_flows * 0.2)
        print('The test flow:', num_test)
        # 取20%作为test
        idx = np.arange(num_flows)
        test_idx = random.sample(list(idx), num_test)
        test_idx = sorted(test_idx)
        print(test_idx)

        # test
        test_dir = test_dataset / str(child.stem)
        test_flow_dir = test_dir / 'flow'
        test_ev_flow_dir = test_flow_dir / 'forward'
        test_flow_dir.mkdir(parents=True, exist_ok=True)
        test_ev_flow_dir.mkdir(exist_ok=True)
        test_time_file = test_flow_dir / 'test_timestamps.txt'

        train_time_file = flow_dir / 'train_timestamps.txt'
        # 将timestamps.txt 分为test/train
        with test_time_file.open(mode='w') as test_file:
            with train_time_file.open(mode='w') as train_file:
                for i in range(num_flows):
                    time = '%d,%d' % (start_times[i], end_times[i])
                    if i in test_idx:
                        test_file.write(time)
                        test_file.write('\n')
                    else:
                        train_file.write(time)
                        train_file.write('\n')

        for i in test_idx:
            name = flow_map_list[i]
            src_location = str(ev_flow_dir / name)
            target_location = str(test_ev_flow_dir / name)
            shutil.move(src_location, target_location)

        print('The %s sequence is DONE!' % str(child))

    print('Total %d sequences are DONE!' % num_sequence)




