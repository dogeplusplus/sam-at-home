import os
import logging
import urllib.request

from pathlib import Path
from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser("Model downloader for the Segment Anything Models")
    parser.add_argument("--model", choices=["all", "vit_h", "vit_b", "vit_l"], default="vit_h")

    args = parser.parse_args()
    return args


def main(args):
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    model_to_url = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    }

    dest = Path("models")
    dest.mkdir(exist_ok=True)

    if args.model == "all":
        for model_url in model_to_url.values():
            model_file = os.path.split(model_url)[-1]
            urllib.request.urlretrieve(model_url, dest / model_file)
            logger.info(f"Downloaded model: {model_file}")
    else:
        model_url = model_to_url[args.model]
        model_file = os.path.split(model_url)[-1]
        urllib.request.urlretrieve(model_url, dest / model_file)
        logger.info(f"Downloaded model: {model_file}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
