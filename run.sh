#horovodrun -np 8 python train.py --lr=0.001 --num_epochs=20 --img_size=224 --optim="adam" --model_name="resnet34" --batch=512 --use_cutmix=False --use_cosine_annealing_with_warmup=False
#horovodrun -np 8 python train.py --lr=0.001 --num_epochs=20 --img_size=224 --optim="adam" --model_name="resnet34" --batch=512 --use_cutmix=True --use_cosine_annealing_with_warmup=False
#horovodrun -np 8 python train.py --lr=0.001 --num_epochs=20 --img_size=224 --optim="adam" --model_name="resnet34" --batch=512 --use_cutmix=True --use_cosine_annealing_with_warmup=True
#horovodrun -np 8 python train.py --lr=0.001 --num_epochs=20 --img_size=224 --optim="adam" --model_name="resnet34" --batch=512 --use_cutmix=True --use_cosine_annealing_with_warmup=True --use_data_balancing=True
horovodrun -np 8 python train.py --lr=0.001 --num_epochs=20 --img_size=224 --optim="adam" --model_name="efficientnet-b3" --batch=128 --use_cutmix=True --use_cosine_annealing_with_warmup=True --use_data_balancing=True
