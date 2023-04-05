# brain_tumor_cls

## Intro
본 프로젝트는 데이터처리언어 수업 프로젝트로 진행하였습니다.

뇌종양은 종양이 발생하는 위치가 뇌이기 때문에 그 만큼 심각성이 높은 질병입니다. 특히 일반적인 종양은 양성와 악성의 분류가 큰 의미를 갖고 있지만, 뇌종양의 경우 발생한 종양이 양성이더라도, 뇌에 발생하기 때문에 심각한 피해를 입을 수 있습니다. 그렇기 때문에 이 프로젝트에서는 Kaggle에서 접근할 수 있는 Brain tumor dataset과 CNN 기반모델들을 활용하여 뇌종양의 유무를 분류할 수 있는 모델을 구축하고자 하였다. 해당 프로젝트에서는 모델 finetuning 방법을 사용하여 CNN 모델에 뇌종양 유무 분류 task를 해결할 수 있도록 학습시켰으며, 더 나아가 일반적으로 널리 사용되는 MobileNetV2, DenseNet121, ResNet 50 모델 중 가장 최적의 모델을 찾고자 하였다.

## 환경 구축
이 프로젝트를 진행하기 위해 Pytorch와 CUDA 설정이 끝마쳐진 환경이 요구된다. 그러므로 아래와 같이 Docker 를 사용하여 환경을 구축한다.

1. Docker image pull 
아래의 커멘드를 사용하여 docker image 설치 (Pytorch 1.7.0 버전, CUDA 11.0 버전)

``` docker pull pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime```


2. 도커 컨테이너 구축
다음의 명령어를 사용하여 컨테이너 구축

```sudo docker run --name brain_tumor --gpus all -it -v /home/workspace/tumor:/tumor -d --ipc=host pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime ```

3. 코드 실행
구축된 컨테이너에 접속한 다음, shell 파일 (.sh 파일)을 사용하여 ResNet, MobileNet, DenseNet의 fineutuning 시작

```sh main.sh```
