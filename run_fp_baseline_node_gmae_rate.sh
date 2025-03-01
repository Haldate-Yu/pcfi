device=$1

if [ -z "$1" ]; then
  echo "empty cuda input!"
  device=0
else
  device=$1
fi

for dataset in "Cora" "CiteSeer" "OGBN-Arxiv"; do
  for fill_method in "zero" "random" "mean" "neighborhood_mean" "feature_propagation" "pcfi"; do
    for mask_type in "uniform" "structural"; do
      for model in "mlp" "sgc" "sage" "gcn" "gat" "gcnmf" "pagnn" "lp"; do
        python run_node.py --dataset_name $dataset \
          --filling_method $fill_method \
          --mask_type $mask_type \
          --missing_rate 0.5 \
          --model $model \
          --gpu_idx $device
      done
    done
  done
done

for dataset in "PubMed"; do
  for fill_method in "zero" "random" "mean" "neighborhood_mean" "feature_propagation" "pcfi"; do
    for mask_type in "uniform" "structural"; do
      for model in "mlp" "sgc" "sage" "gcn" "gat" "gcnmf" "pagnn" "lp"; do
        python run_node.py --dataset_name $dataset \
          --filling_method $fill_method \
          --mask_type $mask_type \
          --missing_rate 0.75 \
          --model $model \
          --gpu_idx $device
      done
    done
  done
done