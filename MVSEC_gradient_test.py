from torch.utils.data import DataLoader

from loader.loader_mvsec_flow import *
import json
import torch.nn
from model import eraft
import argparse
from test import *
from utils import visualization
import numpy as np
import cv2
from skimage import filters, img_as_ubyte
import math
from motion_compensation_loss_func.cvpr2019_warp_and_genIWE import *


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


def get_pred_flow(args):
    config_path = 'config/mvsec_45_test_outdoor.json'
    filter_pth = 'config/mvsec_grident_test_list.json'
    '''
    {"outdoor_day_1": [4345, 6846, 8209], "outdoor_day_2": [17572, 23972, 21273]}
    '''
    config = json.load(open(config_path))
    filters = json.load(open(filter_pth))
    additional_loader_returns = None
    saved_pth = 'MVSEC_gradient_test'

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
    checkpoints = torch.load(config['test']['checkpoint'])
    model.load_state_dict(checkpoints)

    visualizer = get_visualizer(args)
    tester = initialize_tester(config)

    test = tester(model=model,
                  config=config,
                  data_loader=test_set_loader,
                  test_logger=None,
                  save_path=saved_pth,
                  visualizer=visualizer,
                  additional_args=additional_loader_returns)
    flow_est_list, gt_flow_list, event_list, _ = test.mvsec_test()
    '''
        print(type(events_0))   <class 'torch.Tensor'>
        print(events_0.shape)   torch.Size([1, 9554, 4])
        print(type(flow_pred))  <class 'torch.Tensor'>
        print(flow_pred.shape)  torch.Size([2, 256, 256])
        '''
    return flow_est_list, gt_flow_list, event_list


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
    print('In opencv: type(img) is {}'.format(type(img)))
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
    print(math.exp(-contrast))


def bilinear_vote(x_f, y_f):
    # 输入：（x_f, y_f）为坐标，浮点数
    # 输出：其相邻四个点的权重，离得越近越大 w_lu,w_ru,w_ld,w_rd (左上角，右上角，左下角，右下角）
    x_i = int(x_f)
    y_i = int(y_f)
    x_left = float(x_f - x_i)  # 距离左边的长度
    y_up = float(y_f - y_i)  # 到上边的距离

    w_lu = (1. - x_left) * (1. - y_up)
    w_ru = x_left * (1. - y_up)
    w_ld = (1. - x_left) * y_up
    w_rd = x_left * y_up
    return w_lu, w_ru, w_ld, w_rd


def events_compensation(flow_pre, mask, events):
    height = 256
    width = 256
    t_start = events[0][0]
    print('The t_start is {}.'.format(t_start))

    events_num = events.shape[0]
    print('The total events num is {}'.format(events_num))
    pre_valid_num = 0

    pol_accumulated_img = np.zeros((height, width), np.float)  # 极性
    count_accumulated_img = np.zeros((height, width), np.float)  # 数目

    for i in range(events_num):
        one_event = events[i]
        x = one_event[1] - 45.
        y = one_event[2] - 2.
        # 保证事件在crop的像素中，而且光流有效
        if (x in range(0, width)) and (y in range(0, height)) and mask[int(y), int(x)]:
            pre_valid_num += 1
            t = one_event[0]
            p = (one_event[3] * 2) - 1  # {0, 1} -> {-1, 1}
            # MVSEC中光流的time interval=50ms 但是事件的时间戳t以s为单位
            u = (flow_pre[int(y), int(x)][0]) / 50  # 每1ms走过的距离
            v = (flow_pre[int(y), int(x)][1]) / 50
            delta_t = float(t - t_start) / 1000  # 将时间差从s转换为ms
            x_ref = x - u * delta_t
            y_ref = y - v * delta_t  # 运动补偿之后的位置
            # print('Delta_T is {}, (u, v):({},{})  (x, y):({},{}) (x_ref,y_ref): ({}, {})'.format(delta_t, u, v, x,  y, x_ref, y_ref))

            if math.ceil(x_ref) < width and math.ceil(y_ref) < height:
                w_lu, w_ru, w_ld, w_rd = bilinear_vote(x_ref, y_ref)
                x_ref_i = int(x_ref)
                y_ref_i = int(y_ref)
                pol_accumulated_img[y_ref_i, x_ref_i] += w_lu * p
                pol_accumulated_img[y_ref_i, x_ref_i + 1] += w_ru * p
                pol_accumulated_img[y_ref_i + 1, x_ref_i] += w_ld * p
                pol_accumulated_img[y_ref_i + 1, x_ref_i + 1] += w_rd * p

                count_accumulated_img[y_ref_i, x_ref_i] += w_lu
                count_accumulated_img[y_ref_i, x_ref_i + 1] += w_ru
                count_accumulated_img[y_ref_i + 1, x_ref_i] += w_ld
                count_accumulated_img[y_ref_i + 1, x_ref_i + 1] += w_rd
    print('The valid events num is {}'.format(pre_valid_num))
    return count_accumulated_img, pol_accumulated_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', default="/storage1/dataset/MVSEC/mvsec_45HZ", type=str, help="Dataset path")
    parser.add_argument('-d', '--dataset', default="mvsec", type=str, help="Which dataset to use: ([dsec]/mvsec)")
    parser.add_argument('-f', '--frequency', default=45, type=int,
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

    # 从预训练好的模型中得到光流与对应事件
    flow_pred_list, gt_flow_list, events_list = get_pred_flow(args)
    assert len(flow_pred_list) == len(events_list)
    print('len(flow_pred_list) is :', len(flow_pred_list))

    flow_pred = torch.squeeze(flow_pred_list[0])
    gt_flow = torch.squeeze(gt_flow_list[0])  # ([2, 256, 256])

    # print('flow_pred.shape:', flow_pred.shape) flow_pred.shape: torch.Size([2, 256, 256])
    event = events_list[0][0]
    mask = (flow_pred[:, :, 0] != 0) | (flow_pred[:, :, 1] != 0)
    # print('mask.shape, type(mask):'torch.Size([256, 256]) <class 'torch.Tensor'>
    events_num = event.shape[0]
    start_time = 0.0
    height = 256
    width = 256

    warped_x, warped_y, ts, ps = warp_events_flow_torch(event, flow_pred)
    img_pos, img_neg = events_to_timestamp_image(warped_x, warped_y, ts, ps)
    loss = torch.sum(img_pos*img_pos) + torch.sum(img_neg*img_neg)

    # ground truth
    warped_x_gt, warped_y_gt, ts_gt, ps_gt = warp_events_flow_torch(event, gt_flow)
    img_pos_gt, img_neg_gt = events_to_timestamp_image(warped_x_gt, warped_y_gt, ts, ps)
    loss_gt = torch.sum(img_pos_gt * img_pos_gt) + torch.sum(img_neg_gt * img_neg_gt)
    print(' loss is ', loss)
    print(' ground truth loss is ', loss_gt)

'''
    pol_accumulated_img = torch.zeros((256, 256), dtype=torch.float)
    for i in range(events_num):
        one_event = event[i]
        x = one_event[1] - 45.
        y = one_event[2] - 2.
        if (x in range(0, width)) and (y in range(0, height)) and mask[int(y), int(x)]:
            t = one_event[0]
            p = (one_event[3] * 2) - 1  # {0, 1} -> {-1, 1}
            # MVSEC中光流的time interval=50ms 但是事件的时间戳t以s为单位
            u = (flow_pred[int(y), int(x)][0]) / 50  # 每1ms走过的距离
            v = (flow_pred[int(y), int(x)][1]) / 50
            delta_t = float(t -start_time) / 1000  # 将时间差从s转换为ms
            x_ref = x - u * delta_t
            y_ref = y - v * delta_t  # 运动补偿之后的位置
            # print(pol_accumulated_img.device, p.device)  cuda:0 cuda:0

            if math.ceil(x_ref) < width and math.ceil(y_ref) < height:
                w_lu, w_ru, w_ld, w_rd = bilinear_vote(x_ref, y_ref)
                x_ref_i = int(x_ref)
                y_ref_i = int(y_ref)
                pol_accumulated_img[y_ref_i, x_ref_i] += w_lu * p.cpu()
                pol_accumulated_img[y_ref_i, x_ref_i + 1] += w_ru * p.cpu()
                pol_accumulated_img[y_ref_i + 1, x_ref_i] += w_ld * p.cpu()
                pol_accumulated_img[y_ref_i + 1, x_ref_i + 1] += w_rd * p.cpu()

        if i % 500 == 0:
            print('Processing {}/{}'.format(i, events_num))
    print('pol_accumulated_img.shape:', pol_accumulated_img.shape)

    # imageGradient_opencv(pol_accumulated_img.numpy())
    edge_x, edge_y = sobel_xy(pol_accumulated_img)
    contrast_xy = torch.mean(torch.abs(edge_x) + torch.abs(edge_y))
    print('contrast_xy:', contrast_xy)
'''
'''
    for i in range(num):
        print('----------Processing {} th flow and events----------------'.format(i))
        
                print(type(flow_pred_np))  <class 'numpy.ndarray'>
                print(flow_pred_np.shape)  (256, 256, 2)
                print(type(event_np))  <class 'numpy.ndarray'>
                print(type(mask))  <class 'numpy.ndarray'>
                print(mask.shape)  (256, 256)

                break
        flow_name = 'pred_flow_{}.npy'.format(i)
        event_name = 'event_{}.npy'.format(i)
        flow_pred_np = np.load(os.path.join(saved_pth, flow_name))
        event_np = np.load(os.path.join(saved_pth, event_name))  # event_np.shape: (1, 9554, 4)
        event_np = event_np[0]
        t_start = event_np[0][0]
        mask = np.logical_or(flow_pred_np[:, :, 0] != 0, flow_pred_np[:, :, 1] != 0)
        count_img, pol_img = events_compensation(flow_pred_np, mask, event_np)
        print('The opencv gradien of count img:')
        imageGradient_opencv(count_img)
        print('The opencv gradien of pol img:')
        imageGradient_opencv(pol_img)

        imageGradient_skimage报错 ValueError: Images of type float must be between -1 and 1.
        从test结果来看 pol img的Gadient > count img
        '''







