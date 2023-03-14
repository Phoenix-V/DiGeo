split=1
shot=5
gpu=0

set -x

CUDA_VISIBLE_DEVICES=${gpu} python3 -m tools.test_net --num-gpus 1 --eval-only \
    --config-file configs/PascalVOC-detection/split${split}/faster_rcnn_R_101_FPN_ETF_pre${split}_${shot}shot.yaml 

CUDA_VISIBLE_DEVICES=${gpu} python3 -m tools.test_net --num-gpus 1 --eval-only \
    --config-file configs/PascalVOC-detection/split${split}/faster_rcnn_R_101_FPN_ETF_distill${split}_${shot}shot.yaml

