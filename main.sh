for i in 'MobileNetV2' 'DenseNet121' 'ResNet50'
do
    for j in 'False'
    do
        for k in 'adam' 'sgd'
        do
            for x in 'True' 'False'
            do
                CUDA_VISIBLE_DEVICES=0 python resnet50_lr0.0001_mom0_e300_reallast.py --model ${i} --aug ${j} --opt ${k} --freeze ${x}
            done
        done
    done
done