



python main_hail.py --readme 'TEST'   --tasks 'cif'  \
            --dataset 'cifar' --data_path 'dataset/cifar' \
            --num_clients 10   --aux_loss   --global_loss  \
            --start_epoch 0  --epochs 10 --gpu 0 \
            --cfg_path './experiments/cifar.cif.config.json' \
            --save_root '/home/hail09/FedPn/experiments/TEST_meta' \
            --resume '/home/hail09/FedPn/experiments/TEST_meta'



python main_hail.py --readme 'TEST.HAIL'   --tasks 'cif'  \
            --dataset 'cifar' --data_path 'dataset/cifar' \
            --num_clients 10   --aux_loss   --global_loss --deploy_cls \
            --start_epoch 0  --epochs 20 --gpu 1 \
            --cfg_path './experiments/cifar.cif.config.json' \
            --save_root '/home/hail09/FedPn/experiments/TEST_meta_10' \
            --resume '/home/hail09/FedPn/experiments/TEST_meta_10'


python main_hail.py --readme 'TEST.HAIL.re'   --tasks 'cif'  \
            --dataset 'cifar' --data_path 'dataset/cifar' \
            --num_clients 10   --aux_loss   --global_loss  \
            --start_epoch 0  --epochs 50 --gpu 1 \
            --cfg_path './experiments/cifar.cif.config.json' \
            --save_root '/home/hail09/FedPn/experiments/TEST_meta_10re' \
            --resume '/home/hail09/FedPn/experiments/TEST_meta_10re'

python main_hail.py --readme 'HAIL.APN'   --tasks 'cif'  \
            --dataset 'cifar' --data_path 'dataset/cifar' \
            --num_clients 10   --aux_loss   --global_loss   \
            --start_epoch 0  --epochs 50 --gpu 1 \
            --cfg_path './experiments/cifar.cif.config.json' \
            --save_root '/home/hail09/FedPn/experiments/TEST_APN' \
            --resume '/home/hail09/FedPn/experiments/TEST_APN'



python main_hail.py --readme 'HAIL.TUNE_PEN10'   --tasks 'cif'  \
            --dataset 'cifar' --data_path 'dataset/cifar' \
            --num_clients 10   --aux_loss   --global_loss   \
            --start_epoch 0  --epochs 100 --gpu 1 \
            --cfg_path './experiments/cifar.cif.config.json' \
            --save_root '/home/hail09/FedPn/experiments/TEST_TUNE' \
            --resume '/home/hail09/FedPn/experiments/TEST_TUNE'


python main_hail.py --readme 'HAIL.RESET'   --tasks 'cif'  \
            --dataset 'cifar' --data_path 'dataset/cifar' \
            --num_clients 10   --aux_loss   --global_loss   \
            --start_epoch 0  --epochs 100 --gpu 1 \
            --cfg_path './experiments/cifar.cif.config.json' \
            --save_root '/home/hail09/FedPn/experiments/TEST_RESET' \
            --resume '/home/hail09/FedPn/experiments/TEST_RESET'



python main_hail.py --readme 'HAIL.TEST'   --tasks 'cif'  \
            --dataset 'cifar' --data_path 'dataset/cifar' \
            --num_clients 10   --aux_loss   --global_loss   \
            --start_epoch 0  --epochs 100 --gpu 1 \
            --cfg_path './experiments/cifar.cif.config.json' \
            --save_root '/home/hail09/FedPn/experiments/TEST_10' \
            --resume '/home/hail09/FedPn/experiments/TEST_10'


python main_hail.py --readme 'HAIL.TEST_50'   --tasks 'cif'  \
            --dataset 'cifar' --data_path 'dataset/cifar' \
            --num_clients 50   --aux_loss   --global_loss   \
            --start_epoch 0  --epochs 50 --gpu 1 \
            --cfg_path './experiments/cifar.cif.config.json' \
            --save_root '/home/hail09/FedPn/experiments/TEST_50' \
            --resume '/home/hail09/FedPn/experiments/TEST_50'



python main_hail.py --readme 'HAIL.TEST_10_new'   --tasks 'cif'  \
            --dataset 'cifar' --data_path 'dataset/cifar' \
            --num_clients 10   --aux_loss   --global_loss   \
            --start_epoch 0  --epochs 50 --gpu 1 \
            --cfg_path './experiments/cifar.cif.config.json' \
            --save_root '/home/hail09/FedPn/experiments/TEST_10_new' \
            --resume '/home/hail09/FedPn/experiments/TEST_10_new'


python main_hail.py --readme 'HAIL.TEST_100'   --tasks 'cif'  \
            --dataset 'cifar' --data_path 'dataset/cifar' \
            --num_clients 100   --aux_loss   --global_loss   \
            --start_epoch 0  --epochs 50 --gpu 0 \
            --cfg_path './experiments/cifar.cif.config.json' \
            --save_root '/home/hail09/FedPn/experiments/TEST_100' \
            --resume '/home/hail09/FedPn/experiments/TEST_100'