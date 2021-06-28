echo @@@@@@@@@@@@@@@@@@@@@@
echo $(date) - started
#python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig.py --dist 1 --fl 1 --name 'fldist'> train0_fixfldi_test.txt
#python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig.py --dist 0 --fl 1 --name 'fl'> train0_fixfl.txt
#python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig.py --dist 1 --fl 0 --name 'dist' --max 0.75 > train0_fixdist.txt
#python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig.py --dist 1 --fl 1 --name 'distfl0.2' --max 0.85 > train0_fixdistfl2.txt
#python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig.py --resume 1 --name 'resume' max 0.95 > train0_fix_res.txt
#python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig.py --pos 1 --name 'pos_tst' > train0_fixpos_tst.txt
#python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig.py --up 0.1 --name 'up2_test' > train0_fixup_tst.txt
#python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig.py  --name 'weakcrop' > train0_fixweakcrop.txt
#python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig.py  --name '432_4_cent_new' > train0_432_24_new.txt
python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig.py  --name 'def_new_re_xusplt_33' > test10/train0_new_re33.txt
#python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig.py  --resume 1 --name 'def_new_cont' > test10/train0_new_cont.txt
#python valid0.py > valid0.txt
#python save_valid_features0.py > save_valid_features0.txt
#python save_train_features0.py > save_train_features0.txt