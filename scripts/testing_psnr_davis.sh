#! /bin/bash

GPU_ID=0
DATA_ROOT_DIR="results"
DATASETS=(
    davis_rearranged
    )

SCENES=(
    blackswan
    camel
    car-shadow
    dog
    horsejump-high
    motocross-jump
    parkour
    soapbox
    )

N_VIEWS=(
    50
    50
    40
    50
    50
    40
    50
    50
    )

# increase iteration to get better metrics (e.g. gs_train_iter=5000)
gs_train_iter=4000
tag="testing_pnsr"

for i in "${!SCENES[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        SCENE=${SCENES[$i]}
        N_VIEW=${N_VIEWS[$i]}
        # SOURCE_PATH must be Absolute path
        SOURCE_PATH=${DATA_ROOT_DIR}/${DATASET}/${SCENE}/
        MODEL_PATH=${DATA_ROOT_DIR}/${DATASET}/${SCENE}/${tag}_${gs_train_iter}/


        CMD_T="CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore ./train_test_psnr.py \
        -s ${SOURCE_PATH} \
        -m ${MODEL_PATH}  \
        --n_views ${N_VIEW}  \
        --scene ${SCENE} \
        --iter ${gs_train_iter} \
        --optim_pose \
        --dataset davis \
        --gt_dynamic_mask data/davis/DAVIS/Annotations/480p \
        "

        echo "========= ${DATASET}/${SCENE}: Train: jointly optimize pose with dynamic masking ========="
        echo $CMD_T
        eval $CMD_T
    done
done

python scripts/get_testing_psnr_davis.py