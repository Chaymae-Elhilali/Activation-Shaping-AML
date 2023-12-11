target_domain=${1}
test_only=${2}
ALPHA=${3}
APPLY_EVERY_N=${4}
SKIP_FIRST_N=${5}

python main.py \
--experiment=activation_shaping_experiments \
--experiment_name=activation_shaping_experiments/ \
--dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': '${target_domain}','SKIP_FIRST_N': ${SKIP_FIRST_N},'APPLY_EVERY_N': ${APPLY_EVERY_N}, 'ALPHA': ${ALPHA}}" \
--batch_size=128 \
--num_workers=5 \
--grad_accum_steps=1 ${test_only}