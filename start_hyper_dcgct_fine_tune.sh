cuda_id=$1
time_tag="$(date '+%Y%m%d')$2"
time_tag="20240530$2"
checkpoint_tag="2024051102"

data_root="/data/ztjiaweixu/Code/ZTing"
# data_root="/root/datasets"
data_root="/apdcephfs/share_1563664/ztjiaweixu/datasets/dcgct"
output_dir="/apdcephfs/share_1563664/ztjiaweixu/zting/$time_tag"
checkpoint_dir="/apdcephfs/share_1563664/ztjiaweixu/zting/$checkpoint_tag"

time=1.0
for source in AList NList PList RList UList
do
    if [ $(echo $source | grep "AList")x != ""x ];then
        target=NList_PList_RList_UList
    elif [ $(echo $source | grep "NList")x != ""x ];then
        target=AList_PList_RList_UList
    elif [ $(echo $source | grep "PList")x != ""x ];then
        target=AList_NList_RList_UList
    elif [ $(echo $source | grep "RList")x != ""x ];then
        target=AList_PList_NList_UList
    elif [ $(echo $source | grep "UList")x != ""x ];then
        target=AList_PList_RList_NList
    fi
    
    export CUDA_VISIBLE_DEVICES=$cuda_id

    seed=0

    for i in $(seq 4)
    do
        tag=$(date "+%Y%m%d%H%M%S")
        python src/main_hyper_dcgct_fine_tune.py \
                --method 'CDAN' \
                --encoder 'ResNet50' \
                --dataset 'MTRS' \
                --target_inner_iters 1 \
                --target_iters 2000 \
                --test_interval 500 \
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
                --checkpoint_dir $checkpoint_dir \
                --output_dir $output_dir \
                > ~/logs/${source}_${seed}_${tag}.out 2> ~/logs/${source}_${seed}_${tag}.err &
                echo "run $cuda_id $source $seed $tag"
                sleep ${time}
                let seed=$seed+1
    done

    let cuda_id=$cuda_id+1
done

# ps -ef | grep dcgct | awk '{print $2}'| xargs kill -9
