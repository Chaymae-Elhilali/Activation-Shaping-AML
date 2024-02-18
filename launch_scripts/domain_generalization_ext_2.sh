target_domain=${1}
EXTENSION_PARAMS=${2}
K=${3}
RECORD_MODE=${4}
LAYERS_LIST=${5}

python3 main.py \
--experiment=domain_generalization \
--experiment_name=domain_generalization_ext_2/${target_domain}/ \
--dataset_args="{'root': 'data/PACS', 'target_domain': '${target_domain}', 'K': ${K}, 'RECORD_MODE': '${RECORD_MODE}', 'LAYERS_LIST': '${LAYERS_LIST}', 'EXTENSION': 1 }"  \
--batch_size=128 \
--num_workers=5 \
--grad_accum_steps=1 ${EXTENSION_PARAMS}