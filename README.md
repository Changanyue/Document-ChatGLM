
## Dataset

采用如下数据进行训练：

- [`synthdog-zh`](https://huggingface.co/datasets/naver-clova-ix/synthdog-zh): Chinese, 0.5M.



## 训练swin+gpt2
```bash

CUDA_VISIBLE_DEVICES="0" python src/swin_gpt2/train.py --config src/config/train_synthdog_gpt2.yaml --exp_version "donut_gpt2_pretrain_exp_0"





```