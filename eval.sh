# declare -a train=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm")
# declare -a test=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm" "newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")

declare -a train=("newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")
declare -a test=("autoattack" "autopgd" "bim" "cw" "deepfool" "fgsm" "newtonfool" "pgd" "pixelattack" "spatialtransformation" "squareattack")


# Iterate the string array using for loop
for tr in ${train[@]}; do
    for ts in ${test[@]}; do
        python eval.py \
            --train_adversarial $tr \
            --test_adversarial $ts \
            --dataset cifar10 \
            --batch_size 100 \
            --epochs 30 \
            --use2BN
    done
done


# python eval.py \
#     --train_adversarial autoattack \
#     --test_adversarial fgsm \
#     --dataset cifar10 \
#     --batch_size 100 \
#     --epochs 30 \
#     --use2BN
    
