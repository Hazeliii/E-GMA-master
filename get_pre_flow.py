from loader.loader_dsec import *
from loader.loader_mvsec_flow import *
import json
from utils import dsec_utils
import torch.nn
from model import eraft
import argparse
from pathlib import Path
from test import *
import utils.helper_functions as helper
from utils import visualization
import numpy as np
import cv2
from skimage import data, filters, img_as_ubyte


def get_visualizer(args):
    # DSEC dataset
    if args.dataset.lower() == 'dsec':
        return visualization.DsecFlowVisualizer
    # MVSEC dataset
    else:
        return visualization.FlowVisualizerEvents


def initialize_tester(config):
    # Warm Start
    if config['subtype'].lower() == 'warm_start':
        return TestRaftEventsWarm
    # Classic
    else:
        return TestRaftEvents

def get_pre_flow_MVSEC(args):
    config_path = 'config/mvsec_45_test_outdoor.json'
    filter_pth = 'config/mvsec_grident_test_list.json'
    config = json.load(open(config_path))
    filters = json.load(open(filter_pth))
    additional_loader_returns = None
    saved_path = '/media/hazel/Data_files/MVSEC/EGMA_results/gridient_test_flow'

    test_set = MvsecFlow(
        args=config["data_loader"]["test"]["args"],
        filters=filters,
        type='test',
        path=args.path
    )
    test_set_loader = DataLoader(test_set,
                                 batch_size=config['data_loader']['test']['args']['batch_size'],
                                 shuffle=config['data_loader']['test']['args']['shuffle'],
                                 num_workers=args.num_workers,
                                 drop_last=True)
    model = torch.nn.DataParallel(
        eraft.ERAFT(args, n_first_channels=config['data_loader']['test']['args']['num_voxel_bins']))
    # Load Checkpoint
    checkpoint = torch.load(config['test']['checkpoint'])
    # model.load_state_dict(checkpoint['model'])
    model.load_state_dict(checkpoint)

    visualizer = get_visualizer(args)
    # Initialize Tester
    test = initialize_tester(config)

    test = test(
        model=model,
        config=config,
        data_loader=test_set_loader,
        test_logger=None,
        save_path=saved_path,
        visualizer=visualizer,
        additional_args=additional_loader_returns
    )

    # test.summary()
    test._test()
    flow_est_list = test._test()
    flow_pre = torch.squeeze(flow_est_list[0])


def get_pre_flow(args):
    config_path = 'config/dsec_standard.json'
    config = json.load(open(config_path))
    save_path = helper.create_save_path(config['save_dir'].lower(), config['name'].lower())

    loader = DatasetProvider(
        dataset_path=Path(args.path),
        representation_type=dsec_utils.RepresentationType.VOXEL,
        delta_t_ms=100,
        type=config['subtype'].lower(),
        visualize=False
    )
    test_set = loader.get_test_dataset()
    additional_loader_returns = {'name_mapping_test': loader.get_name_mapping_test()}
    test_set_loader = DataLoader(
        test_set,
        batch_size=config['data_loader']['test']['args']['batch_size'],
        shuffle=config['data_loader']['test']['args']['shuffle'],
        num_workers=args.num_workers,
        drop_last=True
    )
    model = torch.nn.DataParallel(eraft.ERAFT(args, n_first_channels=15))
    checkpoints = torch.load('/media/hazel/Data_files/DSEC/EGMA_B2_H4/checkpoints/80000_e-raft.pth')
    model.load_state_dict(checkpoints)
    visualizer = get_visualizer(args)
    my_test = initialize_tester(config)
    test = my_test(
        model=model,
        config=config,
        data_loader=test_set_loader,
        test_logger=None,
        save_path=save_path,
        visualizer=visualizer,
        additional_args=additional_loader_returns
    )
    flow_est_list = test._test()
    flow_pre = torch.squeeze(flow_est_list[0])
    flow_pre = flow_pre.permute(1, 2 ,0)
    return flow_pre # torch.Size([1, 2, 480, 640])


def rectify_events(rectify_map, height, width, x: np.ndarray, y: np.ndarray):
    # assert location in self.locations
    # From distorted to undistorted
    assert rectify_map.shape == (height, width, 2), rectify_map.shape
    assert x.max() < width
    assert y.max() < height
    return rectify_map[y, x]


def bilinear_vote(x_f, y_f):
    # 输入：（x_f, y_f）为坐标，浮点数
    # 输出：其相邻四个点的权重，离得越近越大 w_lu,w_ru,w_ld,w_rd (左上角，右上角，左下角，右下角）
    x_i = int(x_f)
    y_i = int(y_f)
    x_left = float(x_f - x_i)  # 距离左边的长度
    y_up = float(y_f - y_i)  # 到上边的距离

    w_lu = (1. - x_left)*(1. - y_up)
    w_ru = x_left*(1. - y_up)
    w_ld = (1. - x_left)*y_up
    w_rd = x_left*y_up

    return w_lu, w_ru, w_ld, w_rd


def imageGradient_skimage(img):
    # skimage
    edges = filters.sobel(img)
    # 浮点型转换为uint8
    edges = img_as_ubyte(edges)
    plt.figure()
    plt.imshow(edges, plt.cm.gray)
    plt.show()

    # sobel水平方向边缘检
    edges_h = filters.sobel_h(img)
    print(edges_h)
    edges_h = img_as_ubyte(edges_h)
    plt.figure()
    plt.imshow(edges_h, plt.cm.gray)
    plt.show()

    # sobel竖直方向边缘检测
    edges_v = filters.sobel_v(img)
    print(edges_v)
    edges_v = img_as_ubyte(edges_v)
    plt.figure()
    plt.imshow(edges_v, plt.cm.gray)
    plt.show()

    integrand = edges_h*edges_h + edges_v*edges_v
    contrast = np.mean(integrand)
    print(contrast)


def imageGradient_opencv(img):
    edges = cv2.Sobel(img, cv2.CV_16S, 1, 1)
    edges = cv2.convertScaleAbs(edges)
    plt.figure()
    plt.suptitle('sobel all')
    plt.imshow(edges, plt.cm.gray)
    plt.show()

    edges_h = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    edges_h = cv2.convertScaleAbs(edges_h)
    plt.figure()
    plt.suptitle('sobel edges h')
    plt.imshow(edges_h, plt.cm.gray)
    plt.show()

    edges_v = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    edges_v = cv2.convertScaleAbs(edges_v)
    plt.figure()
    plt.suptitle('sobel edges v')
    plt.imshow(edges_v, plt.cm.gray)
    plt.show()

    integrand = edges_h * edges_h + edges_v * edges_v
    contrast = np.mean(integrand)
    print(contrast)


def events_compensated(flow_pre, valid_pre):
    height = 480
    width = 640
    event_path = Path('/media/hazel/Data_files/DSEC/train/thun_00_a/events_left/events.h5')
    rect_file = Path('/media/hazel/Data_files/DSEC/train/thun_00_a/events_left/rectify_map.h5')
    assert event_path.exists()
    assert rect_file.exists()

    t_start = 49599900532
    t_end = 49600000464

    with h5py.File(str(event_path), 'r') as h5f:
        slicer = EventSlicer(h5f)
        events = slicer.get_events(t_start, t_end)
    with h5py.File(str(rect_file), 'r') as h5_rect:
        rectify_ev_map = h5_rect['rectify_map'][()]

    p = events['p']
    t = events['t']
    x = events['x']
    y = events['y']

    xy_rect = rectify_events(rectify_ev_map, height, width, x, y)
    x_rect = xy_rect[:, 0]
    y_rect = xy_rect[:, 1]

    # t/x_rect type : <class 'numpy.ndarray'>

    gt_flow_path = Path('/media/hazel/Data_files/DSEC/test/thun_00_a/flow/forward/000014.png')

    flow_path = Path('/media/hazel/Data_files/DSEC/train/thun_00_a/flow/forward/000002.png')
    flow, valid = Sequence.load_flow(flow_path)

    flow_gt, valid_gt = Sequence.load_flow(gt_flow_path)

    event_num = len(p)
    print('The total events number:', event_num)
    pre_valid_num = 0
    gt_valid_num = 0
    accumulated_img = np.zeros((height, width), np.float)
    accumulated_img_count = np.zeros((height, width), np.int)
    compensated_img = np.zeros((height, width), np.float)
    compensated_img_count = np.zeros((height, width), np.int)

    pre_img = np.zeros((height, width), np.float)
    gt_img = np.zeros((height, width), np.float)

    for i in range(event_num):
        x_i = x[i]
        y_i = y[i]
        t_i = t[i]
        p_i = (p[i] * 2) - 1  # {0, 1} -> {-1, 1}
        rectx_i = x_rect[i]
        recty_i = y_rect[i]

        accumulated_img[y_i, x_i] += p_i

        if valid_pre[y_i, x_i]:
            pre_valid_num += 1
            u_pre = (flow_pre[y_i, x_i][0]) / 100.
            v_pre = (flow_pre[y_i, x_i][1]) / 100.
            delta_t0_ms = float((t_i - t_start)) / 1000
            delta_tN_ms = float((t_end - t_i)) / 1000
            x_ref0 = x_i - u_pre * delta_t0_ms
            y_ref0 = y_i - v_pre * delta_t0_ms

            x_refN = x_i + u_pre * delta_tN_ms
            y_refN = y_i + v_pre * delta_tN_ms

            if (math.ceil(x_ref0) < width and math.ceil(y_ref0) < height):
                w_lu, w_ru, w_ld, w_rd = bilinear_vote(x_ref0, y_ref0)
                x_ref0_i = int(x_ref0)
                y_ref0_i = int(y_ref0)
                pre_img[y_ref0_i, x_ref0_i] += w_lu
                pre_img[y_ref0_i, x_ref0_i + 1] += w_ru
                pre_img[y_ref0_i + 1, x_ref0_i] += w_ld
                pre_img[y_ref0_i + 1, x_ref0_i + 1] += w_rd

        if valid_gt[y_i, x_i]:
            gt_valid_num += 1
            u_gt = (flow_gt[y_i, x_i][0]) / 100.
            v_gt = (flow_gt[y_i, x_i][1]) / 100.
            delta_t0_ms = float((t_i - t_start)) / 1000
            delta_tN_ms = float((t_end - t_i)) / 1000
            x_ref0 = x_i - u_gt * delta_t0_ms
            y_ref0 = y_i - v_gt * delta_t0_ms

            x_refN = x_i + u_gt * delta_tN_ms
            y_refN = y_i + v_gt * delta_tN_ms

            if math.ceil(x_ref0) < width and math.ceil(y_ref0) < height:
                w_lu, w_ru, w_ld, w_rd = bilinear_vote(x_ref0, y_ref0)
                x_ref0_i = int(x_ref0)
                y_ref0_i = int(y_ref0)
                gt_img[y_ref0_i, x_ref0_i] += w_lu
                gt_img[y_ref0_i, x_ref0_i + 1] += w_ru
                gt_img[y_ref0_i + 1, x_ref0_i] += w_ld
                gt_img[y_ref0_i + 1, x_ref0_i + 1] += w_rd
    print('The valid predicted events number:', pre_valid_num)
    print('The valid ground truth events number:', gt_valid_num)

    return pre_img, gt_img


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', default="/media/hazel/Data_files/DSEC1", type=str, help="Dataset path")
    parser.add_argument('-d', '--dataset', default="dsec", type=str, help="Which dataset to use: ([dsec]/mvsec)")
    parser.add_argument('-f', '--frequency', default=20, type=int,
                        help="Evaluation frequency of MVSEC dataset ([20]/45) Hz")
    parser.add_argument('-t', '--type', default='standard', type=str, help="Evaluation type ([warm_start]/standard)")
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='Provide this argument s.t. DSEC results are visualized. MVSEC experiments are always visualized.')
    parser.add_argument('-n', '--num_workers', default=0, type=int,
                        help='How many sub-processes to use for data loading')

    parser.add_argument('--position_only', default=False, action='store_false',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=True, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--num_heads', default=4, help='number of heads in attention and aggregation')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--corr_levels', default=4)
    parser.add_argument('--corr_radius', default=4)

    args = parser.parse_args()
    pre_flow = get_pre_flow(args)
    pre_flow = pre_flow.cpu().numpy()
    print(type(pre_flow))
    print(pre_flow.shape)
    mask =np.logical_or(pre_flow[:, :, 0] != 0, pre_flow[:, :, 1] != 0)
    print(mask.shape)

    pre_img, gt_img = events_compensated(pre_flow, mask)
    print('The gradient of gt img:')
    imageGradient_opencv(pre_img)
    print('The gradient of pre img:')
    imageGradient_opencv(gt_img)


