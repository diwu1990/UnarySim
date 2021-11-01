1. train with soft mgu
python UnarySim/app/uBrain/model_train/model_train_fp.py --task_mi -i="10-10" --set_store


2. train with hard mgu
python UnarySim/app/uBrain/model_train/model_train_fp.py --task_mi -i="10-10" --rnn_hard --set_store


3. inference with hard mgu
python UnarySim/app/uBrain/model_eval/model_eval_fp.py --task_mi -i="10-10" --rnn_hard
python UnarySim/app/uBrain/model_eval/model_eval_fxp.py --task_mi -i="10-10" --rnn_hard
python UnarySim/app/uBrain/model_eval/model_eval_hub.py --task_mi -i="10-10" --rnn_hard

