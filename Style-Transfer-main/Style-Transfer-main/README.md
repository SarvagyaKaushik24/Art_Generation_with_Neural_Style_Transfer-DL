
# Neural Style Transfer 

The repositiory implements the method for Neural Style Transfer based on the [original paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
## Dependencies
This is assuming you have pytorch already setup
```bash
  pip install tqdm
  pip install numpy
```
## Running Tests

Download or clone the github repository and the above listed dependencies. To run, execute the folloing command 
```bash
  python main.py --style_img_pth YOUR_STYLE_IMAGE_PATH --content_img_pth YOUR_CONTENT_IMAGE_PATH --total_steps TRAINING_STEPS --learning_rate YOUR_LR --image_size OUTPUT_IMG_SIZE
```
