device=$1

if [ -z "$1" ]; then
  echo "empty cuda input!"
  device=0
else
  device=$1
fi

for dataset in "Cora" "Citeseer" "Pubmed"; do
  for fill_method in "graphmae"; do
    for mask_type in "uniform" "structural"; do
      for task_type in "transductive" "graph_classification"; do
        python run_link.py --dataset_name $dataset \
        --filling_method $fill_method \
        --mask_type $mask_type \
        --gpu_idx $device \
        --task_type $task_type \
        --mae_seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
        --use_cfg
      done
    done
  done
done
