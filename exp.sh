
# 26 new -----------------
# False global loss
python main_hail.py --readme 'HAIL.freeze_cls.use_net' \
      --num_clients 3   --freeze_cls --aux_loss  \
      --start_epoch 0  --epochs 5 --gpu 1 \
      --cfg_path './experiments/coco.personalized.config.json' \
      --save_root '/home/hail09/FedPn/experiments/coco_off_gls_fre' \
      --resume '/home/hail09/FedPn/experiments/coco_off_gls_fre'

# use global loss
python main_hail.py --readme 'HAIL.freeze_cls.use_net' \
      --num_clients 3  --global_loss --freeze_cls --aux_loss  \
      --start_epoch 0  --epochs 5 --gpu 3 \
      --cfg_path './experiments/coco.personalized.config.json' \
      --save_root '/home/hail09/FedPn/experiments/coco_on_gls_fre' \
      --resume '/home/hail09/FedPn/experiments/coco_on_gls_fre'

      #   -----  use pretrained cls only,
  # FAsle global loss
  python main_hail.py --readme 'HAIL.freeze_cls.use_cls' \
        --num_clients 3   --freeze_cls --aux_loss   --deploy_cls \
        --start_epoch 0  --epochs 5 --gpu 1 \
        --cfg_path './experiments/coco.personalized.config.json' \
        --save_root '/home/hail09/FedPn/experiments/coco_off_gls_fre' \
        --resume '/home/hail09/FedPn/experiments/coco_off_gls_fre'

  # use global loss
  python main_hail.py --readme 'HAIL.freeze_cls.use_cls' \
        --num_clients 3  --global_loss --freeze_cls --aux_loss  --deploy_cls  \
        --start_epoch 0  --epochs 5 --gpu 3 \
        --cfg_path './experiments/coco.personalized.config.json' \
        --save_root '/home/hail09/FedPn/experiments/coco_on_gls_fre' \
        --resume '/home/hail09/FedPn/experiments/coco_on_gls_fre'





# ----------------- 26 new end






# 153 new -----------------
# False global loss, use net
python main_hail.py --readme 'HAIL.freeze_cls.use_net' \
      --num_clients 3   --freeze_cls --aux_loss  \
      --start_epoch 0  --epochs 5 --gpu 0 \
      --cfg_path './experiments/coco.personalized.config.json' \
      --save_root '/home/hail09/FedPn/experiments/coco_off_fre_net' \
      --resume '/home/hail09/FedPn/experiments/coco_off_fre_net'


# use global loss use net
python main_hail.py --readme 'HAIL.freeze_cls.use_net' \
      --num_clients 3  --global_loss --freeze_cls --aux_loss  \
      --start_epoch 0  --epochs 5 --gpu 1 \
      --cfg_path './experiments/coco.personalized.config.json' \
      --save_root '/home/hail09/FedPn/experiments/coco_on_fre_net' \
      --resume '/home/hail09/FedPn/experiments/coco_on_fre_net'


          #   -----  use pretrained cls only,
      # FAsle global loss
      python main_hail.py --readme 'HAIL.freeze_cls.use_cls' \
            --num_clients 3   --freeze_cls --aux_loss   --deploy_cls \
            --start_epoch 0  --epochs 5 --gpu 1 \
            --cfg_path './experiments/coco.personalized.config.json' \
            --save_root '/home/hail09/FedPn/experiments/coco_off_fre_cls' \
            --resume '/home/hail09/FedPn/experiments/coco_off_fre_cls'

      # use global loss
      python main_hail.py --readme 'HAIL.freeze_cls.use_cls' \
            --num_clients 3  --global_loss --freeze_cls --aux_loss  --deploy_cls  \
            --start_epoch 0  --epochs 5 --gpu 0 \
            --cfg_path './experiments/coco.personalized.config.json' \
            --save_root '/home/hail09/FedPn/experiments/coco_on_fre_cls' \
            --resume '/home/hail09/FedPn/experiments/coco_on_fre_cls'



# 153
## clinet setting 3 off global_loss
python main_hail.py --num_clients 5 --epochs 5 --gpu 1  --global_loss 0  --start_epoch 0 \
      --cfg_path './experiments/coco.personalized.config.json' \
      --save_root '/home/hail09/FedPn/experiments/coco_off_gls_5' \
      --resume '/home/hail09/FedPn/experiments/coco_off_gls_5' \




python main_hail.py --num_clients 4 --epochs 5 --gpu 0  --global_loss 1  --start_epoch 0 \
      --cfg_path './experiments/coco.personalized.config.json' \
      --save_root '/home/hail09/FedPn/experiments/coco_on_global_4' \
      --resume '/home/hail09/FedPn/experiments/coco_on_global_4'\


python main_hail_sec.py --num_clients 3 --epochs 5 --gpu 0  --global_loss 1  --start_epoch 0 \
      --freeze_cls 1 \
      --cfg_path './experiments/coco.personalized.config.json' \
      --save_root '/home/hail09/FedPn/experiments/coco_freeze_global_3' \
      --resume '/home/hail09/FedPn/experiments/coco_freeze_global_3'\







#26
python main_hail.py --num_clients 3 --epochs 5 --gpu 3  --global_loss 0  --start_epoch 0 \
      --cfg_path './experiments/coco.personalized.config.json' \
      --save_root '/home/hail09/FedPn/experiments/coco_off_gls' \
      --resume '/home/hail09/FedPn/experiments/coco_off_gls' \
      --readme ''


python main_hail.py --num_clients 3 --epochs 5 --gpu 1  --global_loss 1  --start_epoch 0  \
      --cfg_path './experiments/coco.personalized.config.json' \
      --save_root '/home/hail09/FedPn/experiments/coco_on_global' \
      --resume '/home/hail09/FedPn/experiments/coco_on_global'


#26
python main_hail.py --num_clients 3 --epochs 6 --gpu 3  --global_loss 0  --start_epoch 3 \
      --cfg_path './experiments/coco.personalized.config.json' \
      --save_root '/home/hail09/FedPn/experiments/coco_off_gls_new' \
      --resume '/home/hail09/FedPn/experiments/coco_off_gls_new'


python main_hail.py --num_clients 3 --epochs 6 --gpu 1  --global_loss 1  --start_epoch 3  \
      --cfg_path './experiments/coco.personalized.config.json' \
      --save_root '/home/hail09/FedPn/experiments/coco_on_global_new' \
      --resume '/home/hail09/FedPn/experiments/coco_on_global_new'


# 26 - freeze classifier, train aux, global, pretrain whole net
python main_hail.py --num_clients 3 --epochs 5 --gpu 3  --global_loss 1  --start_epoch 0 \
      --cfg_path './experiments/coco.personalized.config.json' \
      --save_root '/home/hail09/FedPn/experiments/coco_freeze_cls' \
      --resume '/home/hail09/FedPn/experiments/coco_freeze_cls'







# on gloabl_loss  single
python main_hail.py --single_mode 1 --gpu 0 --global_loss 1 --start_epoch 0  \
     --save_root '/home/hail09/FedPn/experiments/coco_on_single' \
     --resume '/home/hail09/FedPn/experiments/coco_on_single'

# off gloabl_loss single
python main_hail.py --single_mode 1 --gpu 3 --global_loss 0 --start_epoch 0 \
     --save_root '/home/hail09/FedPn/experiments/coco_off_single' \
     --resume '/home/hail09/FedPn/experiments/coco_off_single'

# python main_hail.py --single_mode 1 --gpu 0 --global_loss 0 --cfg_path './experiments/coco.personalized.config.json'




#