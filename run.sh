horovodrun -np 8 python train.py --lr=0.001 --num_epochs=6 --img_size=224 --optim="adam" --model_name="resnet34" --batch=512 --use_cutmix=False --use_cosine_annealing_with_warmup=False
horovodrun -np 8 python train.py --lr=0.001 --num_epochs=6 --img_size=224 --optim="adam" --model_name="resnet34" --batch=512 --use_cutmix=True --use_cosine_annealing_with_warmup=False
horovodrun -np 8 python train.py --lr=0.001 --num_epochs=6 --img_size=224 --optim="adam" --model_name="resnet34" --batch=512 --use_cutmix=True --use_cosine_annealing_with_warmup=True
