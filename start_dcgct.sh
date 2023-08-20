cuda_id=$1
time_tag="$(date '+%Y%m%d')$2"
# time_tag="20230818$2"

data_root="/data/ztjiaweixu/Code/ZTing"
data_root="/root/datasets"
output_dir="~/results/ZTing/$time_tag"
# source="webcam"
# target="dslr_amazon"
use_hyper=0

time=1.0
for source in webcam dslr amazon
do
    if [ $(echo $source | grep "webcam")x != ""x ];then
        target=dslr_amazon
    elif [ $(echo $source | grep "dslr")x != ""x ];then
        target=webcam_amazon
    elif [ $(echo $source | grep "amazon")x != ""x ];then
        target=dslr_webcam
    fi
    
    export CUDA_VISIBLE_DEVICES=$cuda_id

    seed=2023

    for i in $(seq 2)
    do
        tag=$(date "+%Y%m%d%H%M%S")
        python src/main_dcgct.py \
                --method 'CDAN' \
                --encoder 'ResNet50' \
                --dataset 'office31' \
                --source_iters 200 \
                --adapt_iters 3000 \
                --finetune_iters 15000 \
                --lambda_node 0.3 \
                --use_hyper $use_hyper \
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
