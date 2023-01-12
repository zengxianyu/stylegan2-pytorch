python generate.py --sample 20 --pics 50 --ckpt checkpoint/$file --output_dir samples/$file --size 256
python -m pytorch_fid /data/FFHQ/images256x256_sample1k samples/$file --device cuda:0  | grep FID | sed "s/^/file $file /" >> samples.fid.txt
