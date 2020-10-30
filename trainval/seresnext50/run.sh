python -m torch.distributed.launch --nproc_per_node=4 train0.py > train0.txt
python valid0.py > valid0.txt
python save_valid_features0.py > save_valid_features0.txt
python save_train_features0.py > save_train_features0.txt
