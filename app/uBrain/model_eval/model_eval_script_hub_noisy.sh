#!/bin/bash

start=`date +%s`

bitwidth_list=(4 5 6 7 8 9 10 11 12)
system="10-10"

# run hub noisy model result
for i in $bitwidth_list; do
    let bitwidth_tc=$i
    let bitwidth_rc=$i
    let depa=$bitwidth_rc+4
    if (($bitwidth_rc < 12));
    then
        let depm=6
    else
        let depm=7
    fi
    echo "python /home/diwu/Project/UnarySim/app/uBrain/model_eval/model_eval_hub_noisy.py --task_sp -i=$system --rnn_hard -bwtc=$bitwidth_tc -bwrc=$bitwidth_rc -bstest=64 -depa=$depa -depm=$depm > /home/diwu/Project/UnarySim/app/uBrain/model_eval/log_hub_noisy_$i.log"
    python /home/diwu/Project/UnarySim/app/uBrain/model_eval/model_eval_hub_noisy.py --task_sp -i=$system --rnn_hard -bwtc=$bitwidth_tc -bwrc=$bitwidth_rc -bstest=64 -depa=$depa -depm=$depm > /home/diwu/Project/UnarySim/app/uBrain/model_eval/log_hub_noisy_$i.log
done

end=`date +%s`
runtime=$((end-start))

echo ""
echo "Total runtime: $runtime seconds"
echo ""
