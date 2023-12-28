## TODO!
Korean Hate Speech Test

## Prepare
```
pip install -r requirements
```


## Training
```
bash train.sh
```


## Inference
```
bash infer.sh
```


## Analysis
```
bash analysis.sh
```

------------------------

## Key Arguments

- args.wrapper
    - [You can wrap your nn.Module by wrapper](https://github.com/jogamy/Template/tree/master/nn/wrapper_templates)
    - [Use Compatible datamodule](https://github.com/jogamy/Template/tree/master/data/datamodules)
    - [Use compatible trainer](https://github.com/jogamy/Template/tree/master/trainer)
    - Options: PL, HF, None, vLLM
        PL - L.Module, L.DataModule, L.Trainer
        HF - None, CustomDataModule, huggingface Trainer
        None - None, CustomDataModule, Custom Trainer
        vLLM - WIP
- args.task
    - You can use any architectures
    - e: dncoder
    - c: classifier
    - d: decoder
    - example: 
        - e1d1 = Transformer
        - e1c1 = BERT

- [More details](https://github.com/jogamy/Template/tree/master/args)




<!-- # Open Intent Classification (WIP)


## Key features
- [**Transformers**](https://https://huggingface.co/docs/transformers/index) 
- [**Lightning**](https://lightning.ai//) 


## 실행 방법1
config.yaml에 모델, 데이터, Trainer를 지정
```bash
python train.py --config <config.yaml>
```

실행 예제
```bash 
python train.py --config samples/feature_extractor.yaml
```

## 실행 방법2
모델, 데이터, Trainer를 각각 따로 지정
```bash 
python train.py --model <model-yaml> --trainer <trainer-yaml> --data <data-yaml> --model_name_or_path <plm-path> --known_cls_ratio <float> --seed <int> --mode <train-or-test>
```

실행 예제
```bash 
python train.py --model samples/model/adb.yaml --trainer samples/trainer/adb.yaml --data samples/data/stackvoerflow.yaml --model_name_or_path bert-base-cased --known_cls_ratio 0.25 --seed 5 --mode train
```
 -->
