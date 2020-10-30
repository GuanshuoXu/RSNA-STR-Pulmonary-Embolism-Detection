python -m torch.distributed.launch --nproc_per_node=4 train0.py > train0.txt
python valid0.py > valid0.txt
python -m torch.distributed.launch --nproc_per_node=4 train1.py > train1.txt
python valid1.py > valid1.txt
python -m torch.distributed.launch --nproc_per_node=4 train2.py > train2.txt
python valid2.py > valid2.txt
python save_bbox_train.py
python save_bbox_valid.py
