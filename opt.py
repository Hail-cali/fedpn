import argparse


'''
this opt file contains baseline model, backbone arc, data, model path, output path

you should check `experiments baseline.config.json` 
model's params is there 
'''


def parse_opt():
    parser = argparse.ArgumentParser(prog='setting API')

    parser.add_argument('--readme', default='', type=str)
    parser.add_argument('--tasks', default='seg', type=str, help='seg | cif')

    # base setting (model, basenet, mode, pretrained)
    parser.add_argument('--mode', default='train', type=str, help='train | inference | test ')
    parser.add_argument('--single_mode', default=0, type=bool, help='when training on single site, use this opt in sh')

    parser.add_argument('--basenet', default='mobilenet', help='pretraiend base model')
    parser.add_argument('--model_name', default='FedSMpn', help='FedMpn | FedSMpn')
    parser.add_argument('--pretrained', default=False, type=bool)


    # gpu & multi processing
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--workers', default=3, type=int)
    parser.add_argument('--gpu', default=1,  help='use gpu num')
    parser.add_argument('--use_cuda', action='store_true', help='If true, use GPU.')
    parser.add_argument('--rank', default=3)
    parser.add_argument('--world-size', default=2, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='', help='url used to set up distributed training env://')

    parser.add_argument('--device', default='cpu', help='allocated in api')

    parser.add_argument('--aux_loss', default=False,  action='store_true')
    parser.add_argument('--global_loss', default=False,  action='store_true')
    parser.add_argument('--freeze_cls', default=False,  action='store_true')
    parser.add_argument('--deploy_cls', default=False, action='store_true',
                        help='if True(input),use finetuned cls layer | default False , use whole finetune ')

    # file root
    parser.add_argument('--root', default='/home/hail09/FedPn')
    parser.add_argument('--model_dir', default='./experiments', help='exp dir in root dir')
    parser.add_argument('--cfg_path', default='./experiments/coco.personalized.config.json',
                        help='baseline.config.json | coco.personalized.config.json')

    parser.add_argument('--save_root', default='/home/hail09/FedPn/experiments/test_sample',
                        help='Location to save checkpoint models')


    # train setting
    parser.add_argument('--dataset', default='coco')
    parser.add_argument('--data_path', default='dataset/coco')

    parser.add_argument('--train_sets', default=['train2017'])
    parser.add_argument('--val_sets', default=['val2017'])


    parser.add_argument('--train_size', default=0.8)

    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--resume', default='/home/hail09/FedPn/experiments/test_sample',
                        help='resume checkpoint path | /home/hail09/FedPn/experiments/coco_mobile')
    parser.add_argument('--test_only', default=False, type=bool, help='if True, test mode, False, load state_dict')

    # log
    parser.add_argument('--tensorboard', default=True)
    parser.add_argument('--log_interval', default=5, type=int, help='Log interval for showing training loss')
    parser.add_argument('--max_log', default=5, type=int)

    # server Initialized
    parser.add_argument('--initial_cls', default='g_net_classifier.pth', type=str)
    parser.add_argument('--cls_path', default='', type=str)

    # client_setting
    parser.add_argument('--client_type', default='client_without_person', type=str,
                        help = 'client_without_person | client_without_car | client_without_dog | ') # not used

    parser.add_argument('--num_clients', default=3, type=int)
    parser.add_argument('--update_status', default=False, help='is needed to update params from cluster')
    parser.add_argument('--global_model_path', default='global.pth')
    parser.add_argument('--global_gpu_set', default=True, type=bool)

    parser.add_argument('--verbose', default=False)

    args = parser.parse_args()
    return args

