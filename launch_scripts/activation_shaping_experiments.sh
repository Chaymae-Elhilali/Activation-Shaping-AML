target_domain=${1}
EXTENSION_PARAMS=${2}
ALPHA=${3}
LAYERS_LIST=${4}

python3 main.py \
--experiment=activation_shaping_experiments \
--experiment_name=activation_shaping_experiments/${target_domain}/ \
--dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': '${target_domain}','ALPHA': ${ALPHA}, 'LAYERS_LIST': '${LAYERS_LIST}'}" \
--batch_size=128 \
--num_workers=5 \
--grad_accum_steps=1 ${EXTENSION_PARAMS}