
time_tag="$(date '+%Y%m%d')$2"

data_root="/data/ztjiaweixu/Code/ZTing"
data_root="/root/datasets"
output_dir="./results/$time_tag"
# source="webcam"
# target="dslr_amazon"
seed=2023

for source in webcam dslr amazon
do
    if [ $(echo $source | grep "webcam")x != ""x ];then
        target=dslr_amazon
    elif [ $(echo $source | grep "dslr")x != ""x ];then
        target=webcam_amazon
    elif [ $(echo $source | grep "amazon")x != ""x ];then
        target=dslr_webcam
    fi
    
    export CUDA_VISIBLE_DEVICES=$1

    for i in $(seq 5)
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
                --source $source \
                --target $target \
                --data_root $data_root \
                --output_dir $output_dir \
                > ~/logs/${source}_${seed}_${tag}.out 2> ~/logs/${source}_${seed}_${tag}.err &
                echo "run $cuda_id $task $seed $tag"
                sleep ${time}
                let seed=$seed+1
    done

    let cuda_id=$cuda_id+1
done