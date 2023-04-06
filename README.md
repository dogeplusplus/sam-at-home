# sam-at-home

A gradio interface to explore capabilities of Meta AI's [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything).

> Me: Mom, can we have SAM?
<br>Mom: No there is SAM at home.
<br>SAM at home:

![Segmentation of Sam the cat using Sam the model](example.png)
*Segmentation of Sam the cat using Sam the model*

## Setup
1. Set up virtual environment using the script `./firstTimeSetup.sh`, or with whichever virtual environment tool you prefer.
2. Activate virtual environment `source venv/bin/activate`
3. Download models (see below)
4. Run UI: `gradio app.py` or `make run`
5. Load UI in browser, usually on `localhost:7860`

### Downloading Models
Models can be downloaded using the `download_models.py` script. By default gets the `vit_h` model, with the option to download all to the `models` folder. Alternatively you can fetch them directly from the official repository and manually place them there.

```
usage: Model downloader for the Segment Anything Models [-h] [--model {all,vit_h,vit_b,vit_l}]

optional arguments:
  -h, --help            show this help message and exit
  --model {all,vit_h,vit_b,vit_l}
```

## Citations
```bibtex
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
```
