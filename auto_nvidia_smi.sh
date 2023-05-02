#!/bin/bash
watch -n 1 'nvidia-smi --query-gpu=index,gpu_name,temperature.gpu,power.draw,utilization.gpu --format=csv \
  | grep -E -i -o "^[0-9]+,.+" \
  | awk -F "," "{print \$1 \": \" \$2 \" | Temp: \" \$3 \"C | Power Draw: \" \$4 \"/300 W | Usage: \" \$5 \"\"}"'
