#!/bin/bash

start=`date +%s`

bandwidth_list=(8 9 10 11 12)
system="10-10"

# run fp model result
echo "python /home/diwu/Project/UnarySim/app/uBrain/model_eval/model_eval_fp.py --task_mi -i=$system --rnn_hard > /home/diwu/Project/UnarySim/app/uBrain/model_eval/log_fp.log"
python /home/diwu/Project/UnarySim/app/uBrain/model_eval/model_eval_fp.py --task_mi -i=$system --rnn_hard > /home/diwu/Project/UnarySim/app/uBrain/model_eval/log_fp.log

# # run fxp model result
for i in $bandwidth_list; do
    let iw=0
    let fw=$i-1
    echo "python /home/diwu/Project/UnarySim/app/uBrain/model_eval/model_eval_fxp.py --task_mi -i=$system --rnn_hard -iw=$iw -fw=$fw > /home/diwu/Project/UnarySim/app/uBrain/model_eval/log_fxp_$i.log"
    python /home/diwu/Project/UnarySim/app/uBrain/model_eval/model_eval_fxp.py --task_mi -i=$system --rnn_hard -iw=$iw -fw=$fw > /home/diwu/Project/UnarySim/app/uBrain/model_eval/log_fxp_$i.log
done

# run hub model result
for i in $bandwidth_list; do
    let bitwidth_tc=$i
    let bitwidth_rc=$i
    echo "python /home/diwu/Project/UnarySim/app/uBrain/model_eval/model_eval_hub.py --task_mi -i=$system --rnn_hard -bwtc=$bitwidth_tc -bwrc=$bitwidth_rc > /home/diwu/Project/UnarySim/app/uBrain/model_eval/log_hub_$i.log"
    python /home/diwu/Project/UnarySim/app/uBrain/model_eval/model_eval_hub.py --task_mi -i=$system --rnn_hard -bwtc=$bitwidth_tc -bwrc=$bitwidth_rc > /home/diwu/Project/UnarySim/app/uBrain/model_eval/log_hub_$i.log
done


end=`date +%s`
runtime=$((end-start))

echo ""
echo "Total runtime: $runtime"
echo ""
