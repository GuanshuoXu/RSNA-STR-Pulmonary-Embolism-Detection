echo @@@@@@@@@@@@@@@@@@@@@@
echo $(date) - started
CUDA_VISIBLE_DEVICES="0" python  linear_classifier.py > linear.txt
#CUDA_VISIBLE_DEVICES="0" python seresnext50_128.py > seresnext50_128.txt
#CUDA_VISIBLE_DEVICES="0" python seresnext101_128.py > seresnext101_128.txt
#CUDA_VISIBLE_DEVICES="0" python seresnext50_192.py > seresnext50_192.txt
#CUDA_VISIBLE_DEVICES="0" python seresnext101_192.py > seresnext101_192.txt
#python prediction_correction.py
