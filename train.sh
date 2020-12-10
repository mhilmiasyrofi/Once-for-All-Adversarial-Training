
#!/bin/bash
 
# Declare an array of string with type
# declare -a adv=("autoattack" "autopgd" "bim" "cw" "deepfool" "newtonfool" "fgsm" "newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")

# declare -a adv=("autoattack" "autopgd" "bim" "cw" "deepfool")

# declare -a adv=("fgsm" "newtonfool" "pgd")

# declare -a adv=("pixelattack" "spatialtransformation" "squareattack")

declare -a adv=("autopgd" "pixelattack")

# Iterate the string array using for loop
for a in ${adv[@]}; do
    python OAT.py \
        --adversarial_data $a \
        --dataset cifar10 \
        --batch_size 100 \
        --epochs 30 \
        --use2BN
done


# python OAT.py \
#     --adversarial_data deepfool \
#     --dataset cifar10 \
#     --batch_size 100 \
#     --epochs 30 \
#     --use2BN


python train.py \
    --adversarial_data deepfool \
    --dataset cifar10 \
    --batch_size 100 \
    --epochs 1 \
    --use2BN