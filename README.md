# brain_tumor_cls

## Intro
팀원 - 윤하영, 정재엽

본 프로젝트는 데이터처리언어 수업 프로젝트로 진행하였습니다.

뇌종양은 종양이 발생하는 위치가 뇌이기 때문에 그 만큼 심각성이 높은 질병입니다. 특히 일반적인 종양은 양성와 악성의 분류가 큰 의미를 갖고 있지만, 뇌종양의 경우 발생한 종양이 양성이더라도, 뇌에 발생하기 때문에 심각한 피해를 입을 수 있습니다. 그렇기 때문에 이 프로젝트에서는 Kaggle에서 접근할 수 있는 Brain tumor dataset과 CNN 기반모델들을 활용하여 주어진 이미지 내에 뇌종양의 유무를 분류할 수 있는 모델을 구축하고자 하였습니다. 해당 프로젝트에서는 모델 finetuning 방법을 사용하여 CNN 모델에 뇌종양 유무 분류 task를 해결할 수 있도록 학습시켰으며, 더 나아가 일반적으로 널리 사용되는 MobileNetV2, DenseNet121, ResNet 50 모델 중 가장 최적의 모델을 찾고자 하였습니다.

## 환경 구축
이 프로젝트를 진행하기 위해 Tensorflow와 CUDA 설정이 끝마쳐진 환경이 요구됩니다. 그러므로 아래와 같이 Docker 를 사용하여 환경을 구축합니다.

1. Docker image pull 
아래의 커멘드를 사용하여 docker image 설치 (Tensorflow 2.10.0 gpu 지원 버전, CUDA 포함)

``` docker pull tensorflow/tensorflow:2.10.0-gpu```


2. 도커 컨테이너 구축
다음의 명령어를 사용하여 컨테이너 구축

```sudo docker run --name brain_tumor --gpus all -it -v /home/workspace/tumor:/tumor -d --ipc=host tensorflow/tensorflow:2.10.0-gpu ```

3. 코드 실행
구축된 컨테이너에 접속한 다음, shell 파일 (.sh 파일)을 사용하여 ResNet, MobileNet, DenseNet의 fineutuning 시작

```sh main.sh```

## Dataset
현재 코드를 실행하기 위해서 필요한 데이터셋은 이 repo에 존재하지 않는다. 따로 brain tumor dataset 을 다운로드한 다음, 아래와 같은 형태로 dataset을 구축해야합니다.
사용한 kaggle 데이터셋은 다음 링크에서 다운받을 수 있습니다. [brain_tumor_dataset](https://www.kaggle.com/datasets/erhmrai/brain-tumor-dataset/code)
```
|
└┬ dataset
 |
 ├ train 
 |   ├  Class A
 |   └  Class B
 └  val 
     ├  Class A
     └  Class B
     
```

#### Preprecessing
주어진 이미지 데이터셋에 대해서 train set (80%), test set (20%)로 무작위 split [split_data.py](https://github.com/YHYeooooong/brain_tumor_cls/blob/main/split_data.py)



#### 모델 훈련
업로드 된 shell 스크립트 파일을 사용하여 Desnet121, ResNet50, MobileNetV2 훈련진행


#### 결론

1. ResNet50

![resnet50-1](https://github.com/YHYeooooong/brain_tumor_cls/assets/43724177/f90982bb-7eb2-459a-b2b7-1355299cae15)

![resnet50-2](https://github.com/YHYeooooong/brain_tumor_cls/assets/43724177/1353d896-9a74-497f-862c-a5ef95af9031)


2. DenseNet121

![densenet121-1](https://github.com/YHYeooooong/brain_tumor_cls/assets/43724177/28766863-1b6c-4c80-9186-a656a4610caa)

![densenet121-2](https://github.com/YHYeooooong/brain_tumor_cls/assets/43724177/9510c830-828b-4bb6-ab5d-2fdb55e10df9)


3. MobileNetV2

![mobilenetv2-1](https://github.com/YHYeooooong/brain_tumor_cls/assets/43724177/44e5c58d-62b5-4efa-9e34-ea2eea40311f)

![mobilenetv2-2](https://github.com/YHYeooooong/brain_tumor_cls/assets/43724177/12edb0fa-66c7-48f0-8047-6d27752f8ded)


각 모델은 93% 의 평균 분류 정확도를 달성하였으나, 적은 수의 이미지로 이루어진 데이터세트를 사용하여 제대로 된 훈련그래프를 확인하기에는 어려움이 존해하였습니다.

추후에 더 큰 이미지 데이터셋과 다양한 학습 하이퍼파라미터를 사용하여 모델들의 분류정확도 비교에 대한 추가연구가 필요합니다. 
