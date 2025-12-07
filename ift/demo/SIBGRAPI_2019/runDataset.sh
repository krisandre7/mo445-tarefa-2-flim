
METHOD_NAME="$1"
DS_PATH="$2"
TYPE="$3"
METHOD_OP="$4"
OBJ_PERC="$5"
MAP_THR="$6"
SM_NAME="$7"

IMG_DIR="${DS_PATH}/images"
GT_DIR="${DS_PATH}/gt"
SM_DIR="${DS_PATH}/${SM_NAME}"
RES_DIR="${DS_PATH}/results"
ID_FILE="${DS_PATH}/${TYPE}.txt"

if [ -z "$SM_NAME" ]; then
    METHOD_DIR="${RES_DIR}/${METHOD_NAME}"
    SM_NAME="${GT_PATH}"
else
    METHOD_DIR="${RES_DIR}/${METHOD_NAME}/${SM_NAME}"
fi

METHOD_LABEL_DIR="${METHOD_DIR}/label"
METHOD_SEED_DIR="${METHOD_DIR}/seed"

METHOD_LABEL_TYPE_DIR="${METHOD_LABEL_DIR}/${TYPE}"
METHOD_SEED_TYPE_DIR="${METHOD_SEED_DIR}/${TYPE}"

mkdir -p "${RES_DIR}"
mkdir -p "${RES_DIR}/${METHOD_NAME}"
mkdir -p "${METHOD_DIR}"
mkdir -p "${METHOD_LABEL_DIR}"    
mkdir -p "${METHOD_SEED_DIR}"      
mkdir -p "${METHOD_LABEL_TYPE_DIR}"    
mkdir -p "${METHOD_SEED_TYPE_DIR}"    

for NUM_SEEDS in `seq 10 10 100`; do
    echo "$NUM_SEEDS -----------------------------"
    EVAL_FILE="${METHOD_DIR}/eval_${TYPE}_${NUM_SEEDS}.txt"

    mkdir -p "${METHOD_LABEL_TYPE_DIR}/${NUM_SEEDS}"    
    mkdir -p "${METHOD_SEED_TYPE_DIR}/${NUM_SEEDS}"    

    echo "IMG_ID,BR,UE,SEEDS_SAMPLED,OBJ_PERC" > "${EVAL_FILE}"

    while IFS='' read -r IMAGE_ID; do
        if [ ! -z "${IMAGE_ID}" ]; then
            echo "\t$IMAGE_ID"
            IMG_FILE=$(find $IMG_DIR -name $IMAGE_ID'.*' | head -n1)
            GT_FILE=$(find $GT_DIR -name $IMAGE_ID'.*' | head -n1)
            SM_FILE=$(find $SM_DIR -name $IMAGE_ID'.*' | head -n1)

            RESULT=$(runISFxROOT "${IMG_FILE}" "${SM_FILE}" "${NUM_SEEDS}" "${OBJ_PERC}" "${MAP_THR}" "${METHOD_OP}" "${METHOD_LABEL_TYPE_DIR}/${NUM_SEEDS}/${IMAGE_ID}.png" "${METHOD_SEED_TYPE_DIR}/${NUM_SEEDS}/${IMAGE_ID}.png" "${GT_FILE}")

            echo $RESULT >> "${EVAL_FILE}"
        fi
    done < "${ID_FILE}"  
done     

7z a "${METHOD_LABEL_TYPE_DIR}.7z" -r "${METHOD_LABEL_TYPE_DIR}"
7z a "${METHOD_SEED_TYPE_DIR}.7z" -r "${METHOD_SEED_TYPE_DIR}"

rm -r "${METHOD_SEED_TYPE_DIR}" "${METHOD_LABEL_TYPE_DIR}"

