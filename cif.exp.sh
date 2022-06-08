
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
            --num_clients 3   --aux_loss --global_loss --deploy_cls  \
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