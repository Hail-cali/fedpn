import argparse


'''
this opt file contains baseline model, backbone arc, data, model path, output path

you should check `experiments baseline.config.json` 
model's params is there 
'''


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='test', help='train | inference | test ')

    parser.add_argument('--basenet', default='resnet50', help='pretraiend base model')
    parser.add_argument('--model_name', default='fpn', help='')
    parser.add_argument('--model_dir', default='./experiments', help='temp ')
    parser.add_argument('--cfg_path', default='./experiments/baseline.config.json')
    parser.add_argument('--root', default='/home/hail09/FedPn')

    parser.add_argument('--dataset', default='coco')
    parser.add_argument('--data_path', default='dataset/coco/val2017')

    parser.add_argument('--train_sets', default=['train2017'])
    parser.add_argument('--val_sets', default=['val2017'])

    parser.add_argument('--train_size', default=0.8)
    parser.add_argument(
        '--file_path',
        default='./dataset/cifar-10-batches-py',
        type=str,
        help='Root directory path of data')

    parser.add_argument(
        '--gpu',
        default=0,
        type=int)

    parser.add_argument(
        '--use_cuda',
        action='store_true',
        help='If true, use GPU.')
    parser.set_defaults(std_norm=False)

    parser.add_argument(
        '--log_interval',
        default=5,
        type=int,
        help='Log interval for showing training loss')

    args = parser.parse_args()
    return args

