## Platform Requirements-

PyTorch 0.41+ |
Linux or macOS |
Python 3 |
CPU or NVIDIA GPU + CUDA CuDNN


## Installation-

Clone this repo:
1) `git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix`
2) `cd pytorch-CycleGAN-and-pix2pix`
3) Install dependencies such as touchvision, visdom, and dominate using `pip install -r requirements.txt`
4) Visualize training results and loss plots, run `python -m visdom.server`and click the URL http://localhost:8097
5) See `options/train_options.py`and `options/base_options.py` for the training flags; see `options/test_options.py and options/base_options.py` for the test flags.
6) Set --gpu_ids -1 to use CPU mode; set --gpu_ids 0,1,2 for multi-GPU mode (Large batch size is needed to benefit from multiple GPUs) 
7) The intermediate results are saved inside `checkpoint` folder as an HTML file

## Download Dataset
Use this link https://drive.google.com/file/d/1ZnVPnGGvlsD1FtPIFrxNgOyi4wEDseLw/view?usp=sharing
Extract all data in one folder, and then divide it into train and test set.

## How to run Pix2Pix?

### Training-
`python3 train.py --dataroot path_to_dataset --name cartoon_pix2pix --model pix2pix --direction BtoA`

### Testing-
`python3 test.py --dataroot path_to_dataset --name cartoon_pix2pix --model test --netG unet_256 --direction BtoA --dataset_mode single --norm batch`


## How to run CycleGAN?

### Training-
`python3 train.py --dataroot path_to_dataset --preprocess scale_width_and_crop --load_size 128 --crop_size 128 --name cartoon_cycle_gan --model cycle_gan`

### Testing-
`python3 test.py --dataroot path_to_dataset --name cartoon_cycle_gan --model test --no_dropout --preprocess scale_width --load_size 128`
