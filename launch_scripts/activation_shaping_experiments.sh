target_domain=${1}
test_only=${2}

python main.py \
--experiment=activation_shaping_experiments \
--experiment_name=activation_shaping_experiments/${target_domain}/ \
--dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': '${target_domain}'}" \
--batch_size=128 \
--num_workers=5 \
--grad_accum_steps=1 ${test_only}