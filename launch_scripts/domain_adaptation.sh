target_domain=${1}
test_only=${2}
K=${3}
APPLY_EVERY_N=${4}
SKIP_FIRST_N=${5}
RECORD_MODE=${6}

python3 main.py \
--experiment=domain_adaptation \
--experiment_name=domain_adaptation/${target_domain}/ \
--dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': '${target_domain}','SKIP_FIRST_N': ${SKIP_FIRST_N},'APPLY_EVERY_N': ${APPLY_EVERY_N}, 'K': ${K}, 'RECORD_MODE': '${RECORD_MODE}' }"  \
--batch_size=128 \
--num_workers=5 \
--grad_accum_steps=1 ${test_only}