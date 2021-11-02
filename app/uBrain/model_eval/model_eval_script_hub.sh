#!/bin/bash

start=`date +%s`

bitwidth_rc_list=(10 11 12)
bitwidth_tc_list=(8 9 10 11 12)
system="10-10"

# run hub model result
for i in $bitwidth_rc_list; do
    for j in $bitwidth_tc_list; do
        let bitwidth_rc=$i
        let depa=$bitwidth_rc+4
        if (($bitwidth_rc < 12));
        then
            let depm=6
        else
            let depm=7
        fi
        let bitwidth_tc=$j
        echo "python /home/diwu/Project/UnarySim/app/uBrain/model_eval/model_eval_hub.py --task_mi -i=$system --rnn_hard -bwtc=$bitwidth_tc -bwrc=$bitwidth_rc -bstest=64 -depa=$depa -depm=$depm > /home/diwu/Project/UnarySim/app/uBrain/model_eval/log_hub_bwrc_${bitwidth_rc}_bwtc_${bitwidth_tc}.log"
        python /home/diwu/Project/UnarySim/app/uBrain/model_eval/model_eval_hub.py --task_mi -i=$system --rnn_hard -bwtc=$bitwidth_tc -bwrc=$bitwidth_rc -bstest=64 -depa=$depa -depm=$depm > /home/diwu/Project/UnarySim/app/uBrain/model_eval/log_hub_bwrc_${bitwidth_rc}_bwtc_${bitwidth_tc}.log
    done
done

end=`date +%s`
runtime=$((end-start))

echo ""
echo "Total runtime: $runtime seconds"
echo ""
