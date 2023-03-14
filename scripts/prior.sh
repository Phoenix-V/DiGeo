URL=8189
DATASET=${1:-voc}

set -x

if [ "$DATASET" = "voc" ];
then
    for split in 1 2 3
    do
        for shot in 1 2 3 5 10
        do
            CUDA_VISIBLE_DEVICES=0,1 python3 -m tools.train_net --num-gpus 2 --dist-url tcp://127.0.0.1:4${URL} \
                --config-file configs/PascalVOC-detection/split${split}/faster_rcnn_R_101_FPN_ETF_pre${split}_${shot}shot.yaml \
                --opts OUTPUT_DIR checkpoints/voc/faster_rcnn/t1 
        done
    done
elif [ "$DATASET" = "coco" ];
then
    for shot in 10 30
    do
        CUDA_VISIBLE_DEVICES=0,1 python3 -m tools.train_net --num-gpus 2 --dist-url tcp://127.0.0.1:4${URL} \
            --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ETF_pre_${shot}shot.yaml \
    done
fi
