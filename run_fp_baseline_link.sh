device=$1

if [ -z "$1" ]; then
  echo "empty cuda input!"
  device=0
else
  device=$1
fi

for dataset in "Cora" "CiteSeer" "PubMed" "Photo" "Computers"; do
  for fill_method in "zero" "random" "mean" "neighborhood_mean" "feature_propagation" "pcfi"; do
    for mask_type in "uniform" "structural"; do
      python run_link.py --dataset_name $dataset \
        --filling_method $fill_method \
        --mask_type $mask_type \
        --missing_rate 0.9 \
        --gpu_idx $device
    done
  done
done
