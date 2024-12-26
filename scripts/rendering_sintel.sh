#! /bin/bash

GPU_ID=0
DATA_ROOT_DIR="results"
DATASETS=(
    sintel_rearranged
    )

SCENES=(
    alley_2
    ambush_4
    ambush_5
    ambush_6
    cave_2
    cave_4
    market_2
    market_5
    market_6
    shaman_3
    sleeping_1
    sleeping_2
    temple_2
    temple_3
    )

N_VIEWS=(
    50
    33
    50
    20
    50
    50
    50
    50
    40
    50
    50
    50
    50
    50
    )

# increase iteration to get better metrics (e.g. gs_train_iter=5000)
gs_train_iter=4000
tag="rendering_all_frames"

for i in "${!SCENES[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        SCENE=${SCENES[$i]}
        N_VIEW=${N_VIEWS[$i]}
        # SOURCE_PATH must be Absolute path
        SOURCE_PATH=${DATA_ROOT_DIR}/${DATASET}/${SCENE}/
        MODEL_PATH=${DATA_ROOT_DIR}/${DATASET}/${SCENE}/${tag}_${gs_train_iter}/

        # # ----- (1) Train: jointly optimize pose -----
        CMD_T="CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore ./train_gui.py \
        -s ${SOURCE_PATH} \
        -m ${MODEL_PATH}  \
        --iter ${gs_train_iter} \
        --dataset sintel \
        --gt_dynamic_mask data/sintel/training/dynamic_label_perfect \
        "

        CMD_RI="CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore ./render.py \
        -s ${SOURCE_PATH} \
        -m ${MODEL_PATH}  \
        --n_views ${N_VIEW}  \
        --scene ${SCENE} \
        --iter ${gs_train_iter} \
        --eval \
        --get_video \
        "

        echo "========= ${DATASET}/${SCENE}: Train: jointly optimize pose with dynamic masking ========="
        echo $CMD_T
        eval $CMD_T
        echo "========= ${DATASET}/${SCENE}: Render:Rendering Static Scene Reconstruction ========="
        echo $CMD_RI
        eval $CMD_RI
    done
done
