split=1
shot=5

set -x

CUDA_VISIBLE_DEVICES=0 python3 -m tools.test_net --num-gpus 1 
    --config-file configs/PascalVOC-detection/split${split}/faster_rcnn_R_101_FPN_ETF_pre${split}_${shot}shot.yaml \

