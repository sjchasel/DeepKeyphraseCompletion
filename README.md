# DeepKeyphraseCompletion

This repository contains the code for our paper "Deep Keyphrase Completion".
Our implementation is built on the source code from [deep-keyphrase](https://github.com/supercoderhawk/deep-keyphrase). Thanks for its work.

# Data Preprocess

data: https://drive.google.com/file/d/1c3LsCVrYcDU8UyltDHPJgcmp6-BexD0a/view?usp=sharing

```shell
mkdir data
# !! please unzip data put the files into above folder manually
bash scripts/prepare_kp20k.sh
```

# Train

```shell
bash scripts/train_copyrnn_kp20k.sh

# start tensorboard
# enter the experiment result dir, suffix is time that experiment starts
cd data/kp20k/copyrnn_kp20k_basic-20191212-080000
# start tensorboard services
tenosrboard --bind_all --logdir logs --port 6006
```

# Test

```shell
python predict_runner.py
```


