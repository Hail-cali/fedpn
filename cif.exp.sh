










# 153

python main_hail.py --readme 'HAIL.CIF'   --tasks 'cif' \
            --dataset 'cifar' --data_path 'dataset/cifar' \
            --num_clients 3   --aux_loss --global_loss --deploy_cls  \
            --start_epoch 0  --epochs 5 --gpu 1 \
            --cfg_path './experiments/cifar.cif.config.json' \
            --save_root '/home/hail09/FedPn/experiments/cifar_on_cif' \
            --resume '/home/hail09/FedPn/experiments/cifar_on_cif'

# ***************** 26 *************************
# use global loss
python main_hail.py --readme 'HAIL.CIF'   --tasks 'cif' \
            --dataset 'cifar' --data_path 'dataset/cifar' \
            --num_clients 5   --aux_loss --global_loss --deploy_cls  \
            --start_epoch 0  --epochs 10 --gpu 1 \
            --cfg_path './experiments/cifar.cif.config.json' \
            --save_root '/home/hail09/FedPn/experiments/cifar_on_cif' \
            --resume '/home/hail09/FedPn/experiments/cifar_on_cif'

# no use global loss
python main_hail.py --readme 'HAIL.CIF'   --tasks 'cif' \
            --dataset 'cifar' --data_path 'dataset/cifar' \
            --num_clients 5   --aux_loss  --deploy_cls  \
            --start_epoch 0  --epochs 10 --gpu 0 \
            --cfg_path './experiments/cifar.cif.config.json' \
            --save_root '/home/hail09/FedPn/experiments/cifar_off_cif' \
            --resume '/home/hail09/FedPn/experiments/cifar_off_cif'

# test
python main_hail.py --readme 'TEST.CIF'   --tasks 'cif' \
            --dataset 'cifar' --data_path 'dataset/cifar' \
            --num_clients 5   --aux_loss  --deploy_cls  \
            --start_epoch 0  --epochs 10 --gpu 0 \
            --cfg_path './experiments/cifar.cif.config.json' \
            --save_root '/home/hail09/FedPn/experiments/cifar_test' \
            --resume '/home/hail09/FedPn/experiments/cifar_test'

python main_hail.py --readme 'TEST.CIF.ON'   --tasks 'cif' \
            --dataset 'cifar' --data_path 'dataset/cifar' \
            --num_clients 5   --aux_loss  --global_loss --deploy_cls  \
            --start_epoch 0  --epochs 10 --gpu 0 \
            --cfg_path './experiments/cifar.cif.config.json' \
            --save_root '/home/hail09/FedPn/experiments/cifar_test_on' \
            --resume '/home/hail09/FedPn/experiments/cifar_test_on'


python main_hail.py --readme 'TEST.CIF.freeze'   --tasks 'cif' \
            --dataset 'cifar' --data_path 'dataset/cifar' \
            --num_clients 5   --aux_loss  --global_loss --deploy_cls   \
            --start_epoch 0  --epochs 10 --gpu 0 \
            --cfg_path './experiments/cifar.cif.config.json' \
            --save_root '/home/hail09/FedPn/experiments/cifar_f' \
            --resume '/home/hail09/FedPn/experiments/cifar_f'

python main_hail.py --readme 'TEST.CIF.ONE'   --tasks 'cif' \
            --dataset 'cifar' --data_path 'dataset/cifar' \
            --num_clients 7   --aux_loss  --global_loss --deploy_cls   \
            --start_epoch 0  --epochs 10 --gpu 0 \
            --cfg_path './experiments/cifar.cif.config.json' \
            --save_root '/home/hail09/FedPn/experiments/cifar_one' \
            --resume '/home/hail09/FedPn/experiments/cifar_one'


python main_hail.py --readme 'TEST.CIF.ONE_NO_GLO'   --tasks 'cif' \
            --dataset 'cifar' --data_path 'dataset/cifar' \
            --num_clients 7   --aux_loss   --deploy_cls   \
            --start_epoch 0  --epochs 10 --gpu 1 \
            --cfg_path './experiments/cifar.cif.config.json' \
            --save_root '/home/hail09/FedPn/experiments/cifar_no_one' \
            --resume '/home/hail09/FedPn/experiments/cifar_no_one'

python main_hail.py --readme 'TEST.CIF.freeze_use'   --tasks 'cif' \
            --dataset 'cifar' --data_path 'dataset/cifar' \
            --num_clients 7   --aux_loss  --global_loss --deploy_cls  --freeze_cls \
            --start_epoch 0  --epochs 5 --gpu 0 \
            --cfg_path './experiments/cifar.cif.config.json' \
            --save_root '/home/hail09/FedPn/experiments/cifar_freeze_on' \
            --resume '/home/hail09/FedPn/experiments/cifar_freeze_on'

python main_hail.py --readme 'TEST.CIF.freeze_no_use'   --tasks 'cif' \
            --dataset 'cifar' --data_path 'dataset/cifar' \
            --num_clients 7   --aux_loss   --deploy_cls --freeze_cls  \
            --start_epoch 0  --epochs 5 --gpu 0 \
            --cfg_path './experiments/cifar.cif.config.json' \
            --save_root '/home/hail09/FedPn/experiments/cifar_freeze_off' \
            --resume '/home/hail09/FedPn/experiments/cifar_freeze_off'


 python main_hail.py --readme 'FIT.CIF.'   --tasks 'cif' \
            --dataset 'cifar' --data_path 'dataset/cifar' \
            --num_clients 7   --aux_loss   --global_loss --deploy_cls --freeze_cls  \
            --start_epoch 0  --epochs 5 --gpu 0 \
            --cfg_path './experiments/cifar.cif.config.json' \
            --save_root '/home/hail09/FedPn/experiments/TEST_fit' \
            --resume '/home/hail09/FedPn/experiments/TEST_fit'


 python main_hail.py --readme 'FIT.CIF.'   --tasks 'cif' \
            --dataset 'cifar' --data_path 'dataset/cifar' \
            --num_clients 10   --aux_loss   --global_loss --deploy_cls  \
            --start_epoch 0  --epochs 20 --gpu 0 \
            --cfg_path './experiments/cifar.cif.config.json' \
            --save_root '/home/hail09/FedPn/experiments/TEST_10' \
            --resume '/home/hail09/FedPn/experiments/TEST_10'



python main_hail.py --readme 'FIT.CIF.new'   --tasks 'cif' \
            --dataset 'cifar' --data_path 'dataset/cifar' \
            --num_clients 10   --aux_loss   --global_loss --deploy_cls  \
            --start_epoch 0  --epochs 20 --gpu 0 \
            --cfg_path './experiments/cifar.cif.config.json' \
            --save_root '/home/hail09/FedPn/experiments/TEST_10_new' \
            --resume '/home/hail09/FedPn/experiments/TEST_10_new'

python main_hail.py --readme 'FIT.CIF.re'   --tasks 'cif' \
            --dataset 'cifar' --data_path 'dataset/cifar' \
            --num_clients 10   --aux_loss   --global_loss --deploy_cls  \
            --start_epoch 0  --epochs 20 --gpu 0 \
            --cfg_path './experiments/cifar.cif.config.json' \
            --save_root '/home/hail09/FedPn/experiments/TEST_10_re' \
            --resume '/home/hail09/FedPn/experiments/TEST_10_re'

python main_hail.py --readme 'FIT.CIF.one'   --tasks 'cif' \
            --dataset 'cifar' --data_path 'dataset/cifar' \
            --num_clients 10   --aux_loss   --global_loss --deploy_cls  \
            --start_epoch 0  --epochs 20 --gpu 0 \
            --cfg_path './experiments/cifar.cif.config.json' \
            --save_root '/home/hail09/FedPn/experiments/TEST_10_one' \
            --resume '/home/hail09/FedPn/experiments/TEST_10_one'

python main_hail.py --readme 'FIT.CIF.full'   --tasks 'cif' \
            --dataset 'cifar' --data_path 'dataset/cifar' \
            --num_clients 10   --aux_loss   --global_loss  \
            --start_epoch 0  --epochs 10 --gpu 0 \
            --cfg_path './experiments/cifar.cif.config.json' \
            --save_root '/home/hail09/FedPn/experiments/TEST_10_full' \
            --resume '/home/hail09/FedPn/experiments/TEST_10_full'

python main_hail.py --readme 'TEST'   --tasks 'cif' \
            --dataset 'cifar' --data_path 'dataset/cifar' \
            --num_clients 10   --aux_loss   --global_loss  \
            --start_epoch 0  --epochs 10 --gpu 0 \
            --cfg_path './experiments/cifar.cif.config.json' \
            --save_root '/home/hail09/FedPn/experiments/TEST_LST' \
            --resume '/home/hail09/FedPn/experiments/TEST_LST'