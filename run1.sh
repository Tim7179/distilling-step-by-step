#!/bin/bash\
.conda/bin/python run.py \
  --from_pretrained google/t5-v1_1-large --dataset OpenR1-Math-220k \
  --model_type standard --label_type llm \
  --eval_steps 1000 \
  --batch_size 1 --grad_steps 16