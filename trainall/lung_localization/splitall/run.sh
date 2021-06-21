python -m torch.distributed.launch --nproc_per_node=2 train0.py > train0.txt
python -m torch.distributed.launch --nproc_per_node=2 train1.py > train1.txt
python -m torch.distributed.launch --nproc_per_node=2 train2.py > train2.txt
python save_bbox_train.py
