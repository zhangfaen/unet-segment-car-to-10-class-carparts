# U-Net: Semantic segmentation with PyTorch, Segment a car into 10-class car parts
<img src="https://raw.githubusercontent.com/zhangfaen/common/master/unet-logo.png" />

![input and output for a random image in the test dataset](https://raw.githubusercontent.com/zhangfaen/unet-segment-car-to-10-class-carparts/main/sample.png)


Customized implementation of the [U-Net](https://arxiv.org/abs/1505.04597) in PyTorch for Roboflow's [Tooth Detection](https://universe.roboflow.com/minkyu/tooth-detection-myvcg/) from high definition images.

## Quick start

### Train
```bash
nohup python train.py --epochs 10 --batch-size 16 --scale 0.6 --classes 10 --trained-model-file trained-models/unet-model-scale0.6-batchsize16.pth &

nohup python train.py --epochs 10 --batch-size 16 --scale 0.6 --classes 10 --trained-model-file trained-models/unet-model-scale0.6-batchsize16-585samples.pth &
```

### Test

```bash
python predict.py -i ./test-images-from-internet/test2.jpg  -o ./test-images-from-internet/test2-output.jpg --model trained-models/unet-model-scale0.6-batchsize16.pth --bilinear --scale 0.6

python predict.py -i ./test-images-from-internet/test1.jpg  -o ./test-images-from-internet/test1-output.jpg --model trained-models/unet-model-scale0.6-batchsize16.pth --bilinear --scale 0.6

python predict.py -i ./test-images-from-internet/test3.jpg  -o ./test-images-from-internet/test3-output.jpg --model trained-models/unet-model-scale0.6-batchsize16.pth --bilinear --scale 0.6


python predict.py -i ./test-images-from-internet/test3.jpg  -o ./test-images-from-internet/test3-output-585samples.jpg --model trained-models/unet-model-scale0.6-batchsize16-585samples.pth --bilinear --scale 0.6

python predict.py -i ./test-images-from-internet/test2.jpg  -o ./test-images-from-internet/test2-output-585samples.jpg --model trained-models/unet-model-scale0.6-batchsize16-585samples.pth --bilinear --scale 0.6


python predict.py -i ./test-images-from-internet/test1.jpg  -o ./test-images-from-internet/test1-output-585samples.jpg --model trained-models/unet-model-scale0.6-batchsize16-585samples.pth --bilinear --scale 0.6
```

The input images and target masks should be in the `data/imgs` and `data/masks` folders respectively (note that the `imgs` and `masks` folder should not contain any sub-folder or any other files, due to the greedy data-loader). For Carvana, images are RGB and masks are black and white.

You can use your own dataset as long as you make sure it is loaded properly in `utils/data_loading.py`.


---

Original paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox:

[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

![network architecture](https://i.imgur.com/jeDVpqF.png)

