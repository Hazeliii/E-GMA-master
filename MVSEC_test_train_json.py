import json
from pathlib import Path
import random
import os
import numpy as np


if __name__ == '__main__':
    dataset_root = '/storage1/dataset/MVSEC/mvsec_45HZ'
    sequences = ['outdoor_day_1', 'outdoor_day_2']

    train_json = 'config/mvsec_train_outdoor_filter.json'
    test_json = 'config/mvsec_test_outdoor_filter.json'
    train_dataset = {}
    test_dataset = {}
    for name in sequences:
        print('----------------{}---------------------'.format(name))
        images_time = os.path.join(dataset_root, name, 'timestamps_images.txt')
        flows_time = os.path.join(dataset_root, name, 'timestamps_flow.txt')

        timestamp_files_flow = np.loadtxt(flows_time)
        timestamp_files_image = np.loadtxt(images_time)

        flow_time_min = timestamp_files_flow.min()
        flow_time_max = timestamp_files_flow.max()
        # 在image_time中找到第一个比flow_time_min大的
        image_time_idx_min = np.searchsorted(timestamp_files_image, flow_time_min, side='right')
        image_time_idx_max = np.searchsorted(timestamp_files_image, flow_time_max, side='right')

        if name == 'outdoor_day_2':
            image_time_idx_max -= 1
        # 删除【0, image_time_idx_min-1】之间的所有.h5文件
        # 删除image_time_idx_max及其之后的所有.h5文件

        print('In {}, the flow_time_min is {}, the min idx in timestamps_images.txt is {}, the max idx is {}'
              .format(name, flow_time_min, image_time_idx_min, image_time_idx_max))

        sequence_pth = Path(dataset_root) / name
        flow_pth = sequence_pth / 'davis/left/events'
        flow_list = list()
        for flow in flow_pth.iterdir():
            assert str(flow.name).endswith('.h5')
            idx = int(flow.name.split('.')[0])
            flow_list.append(idx)
        events_list = list(filter(lambda x: image_time_idx_min-1 < x < image_time_idx_max, flow_list))
        events_list.sort()

        print('The total .h5 file is {}'.format(events_list))

        total_num = len(events_list)
        test_num = int(total_num * 0.15)
        print("The total number of events is {}, the test number of events is {}".format(total_num, test_num))

        test_idx = random.sample(events_list, test_num)
        test_idx = sorted(test_idx)

        train_list = []
        test_list = []
        for idx in events_list:
            if idx in test_idx:
                test_list.append(idx)
            else:
                train_list.append(idx)
        train_dataset[name] = train_list
        test_dataset[name] = test_list
    test_str = json.dumps(test_dataset)
    train_str = json.dumps(train_dataset)

    with open(test_json, 'w') as file:
        file.write(test_str)
    with open(train_json, 'w') as file:
        file.write(train_str)


'''
outdoorday1:The total number of events is 11701, the test number of events is 1755
outdoorday2:The total number of events is 26682, the test number of events is 4002

'''