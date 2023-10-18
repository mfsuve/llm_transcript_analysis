from argparse import ArgumentParser
from configparser import ConfigParser
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple
import numpy as np
from transformers import BartForSequenceClassification, BartTokenizer, pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def read_config(config_path: str):
    logging.info(f"Reading the config")
    config = ConfigParser()
    config.read(config_path)
    return config


def read_data(data_path: str) -> List[Dict[str, str]]:
    logging.info(f"Reading the transcript")
    with Path(data_path).open("r") as data_file:
        transcript = json.load(data_file)
    return transcript["dialogue"]


def load_pipeline(config: ConfigParser, resources_dir: str):
    if not Path(resources_dir).is_dir():
        logging.info(f"Creating {resources_dir} as the resource directory")
        Path(resources_dir).mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading the model")
    model = BartForSequenceClassification.from_pretrained(config["DEFAULT"]["model_name"], cache_dir=resources_dir)
    tokenizer = BartTokenizer.from_pretrained(config["DEFAULT"]["model_name"], cache_dir=resources_dir)
    return pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)


def analyze(config_path: str, data_path: str, resources_dir: str):
    config = read_config(config_path)
    pipe = load_pipeline(config, resources_dir)
    transcript = read_data(data_path)


def main():
    parser = ArgumentParser(description="LLM Transcript Analysis")
    parser.add_argument("-c", "--configpath", required=True)
    parser.add_argument("-d", "--datapath", required=True)
    parser.add_argument("-r", "--resourcesdir", required=True)
    args = parser.parse_args()
    analyze(args.configpath, args.datapath, args.resourcesdir)


if __name__ == "__main__":
    main()
