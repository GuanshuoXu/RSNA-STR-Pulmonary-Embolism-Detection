echo @@@@@@@@@@@@@@@@@@@@@@
echo $(date) - started
#python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig.py --mask 1 --name 'mask'> train0_mask_def.txt
#python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig.py --dt 1 --name 'flip_dtt_33'> train0_flip_dt33.txt
#python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig.py --pre 1 --dt 1 --name 'pre_dtt_33'> train0_pre_dt3.txt
#python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig.py --fl 1 --name 'fl_20_'> train0_fl_20.txt
#python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig.py --dist 1 --fl 0 --name 'dist' --max 0.75 > train0_fixdist.txt
#python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig.py --dist 1 --fl 1 --name 'distfl0.2' --max 0.85 > train0_fixdistfl2.txt
#python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig.py --resume -1 --name 'test val' --max 0.95 > train0_fix_val.txt
#python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig.py --pos 1 --name 'pos_tst' > train0_fixpos_tst.txt
#python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig.py --ratio 0.02 --resume -1 --up 0.2 --name 'up2_test' > train0_fixup_tst2.txt
#python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig.py --ratio 0.02 --resume 1 --ep0 0 --up 0.2 --name 'up2_test' > train0_fixup_res_tst22.txt
#python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig.py  --name 'weakcrop' > train0_fixweakcrop.txt
#python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig.py  --name '432_4_cent_new' > train0_432_24_new.txt
#python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig.py  --name 'vert' > test10/vert.txt
#python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig.py  --resume 1 --ep0 6 --name 'no_wd_new' > train0_nowd_re_cont.txt
#python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig89.py  --dt 1 --name 'dt_long' > test10/dt_re_12.txt
python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig89.py  --name 'def_long' > test10/def_re_12.txt
#python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig.py  --opt 1 --resume 1 --ep0 0 --name 'def_opt1' > test10/train0__opt1_cont.txt
#python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig.py  --dt 1 --max 0.95 --min 0.03 --name 'dt_95_03' > test10/train0_95_03.txt
#python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig.py   --pre 1 --name 'pre_def' > test10/train0_pre_def.txt
#python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig.py    --name '432_4' > test10/train0__432.txt
#python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig.py    --size 476 --name '512_4new' > test10/train0__512newe.txt
#python -m torch.distributed.launch --nproc_per_node=4 train0_fix_sig.py  --mu 3  --name 'mu3' > test10/train0_mu3.txt
#python valid0.py > valid0.txt
#python save_valid_features0.py > save_valid_features0.txt
#python save_train_features0.py > save_train_features0.txt