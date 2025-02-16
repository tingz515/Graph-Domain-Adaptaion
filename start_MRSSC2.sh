cuda_id=$1
time_tag="$(date '+%Y%m%d')$2"
time_tag="20241022$2"

data_root="/data/ztjiaweixu/Code/ZTing"
# data_root="/root/datasets"
data_root="/apdcephfs/share_1563664/ztjiaweixu/datasets/dcgct"
output_dir="/apdcephfs/share_1563664/ztjiaweixu/zting/$time_tag"

time=1.0
dataset="MRSSC2" # MTRS, MRSSC2, office31, office-home
for source in source
do
    target=Target1_Target2_Target3
    
    export CUDA_VISIBLE_DEVICES=$cuda_id

    seed=0

    for i in $(seq 2)
    do
        tag=$(date "+%Y%m%d%H%M%S")
        python src/main_hyper_dcgct.py \
                --method 'CDAN' \
                --encoder 'ResNet50' \
                --dataset $dataset \
                --target_epochs 1 \
                --source_epochs 1 \
                --adapt_epochs 10 \
                --finetune_epochs 10 \
                --test_interval 2 \
                --source_batch 32 \
                --target_batch 32 \
                --test_batch 64 \
                --use_hyper 1 \
                --multi_mlp 0 \
                --unable_gnn 0 \
                --finetune_light 1 \
                --distill_light 1 \
                --mlp_pseudo 0 \
                --seed $seed \
                --source $source \
                --target $target \
                --data_root $data_root \
                --output_dir $output_dir \
                > ~/logs/${source}_${seed}_${tag}.out 2> ~/logs/${source}_${seed}_${tag}.err &
                echo "run $cuda_id $source $seed $tag"
                sleep ${time}
                let seed=$seed+1
    done

    let cuda_id=$cuda_id+1
done

# ps -ef | grep dcgct | awk '{print $2}'| xargs kill -9
