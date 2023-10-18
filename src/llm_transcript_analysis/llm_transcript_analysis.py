from argparse import ArgumentParser
from configparser import ConfigParser
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Union
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

    possible_sentiments = [sentiment for _, sentiment in config.items("possible_sentiments")]
    possible_intentions = [intention for _, intention in config.items("possible_intentions")]

    results: List[Dict[str, Union[str, List[str]]]] = []
    for interaction in transcript:
        speaker = interaction["speaker"]
        message = interaction["message"]

        if speaker != "client":
            results.append(interaction)
            continue

        # Single sentiment per message
        sentiment: str = pipe(message, candidate_labels=possible_sentiments)["labels"][0]
        # Multiple intentions per message
        intention_results = pipe(message, candidate_labels=possible_intentions, multi_label=True)
        # Get the possible intentions with score greater than 95%
        intentions: List[str] = [
            label for label, score in zip(intention_results["labels"], intention_results["scores"]) if score > 0.95
        ]
        # If no intention was found, pick the most confident one
        if len(intentions) == 0:
            intentions = [intention_results["labels"][0]]

        interaction["sentiment"] = sentiment
        interaction["intentions"] = intentions
        results.append(interaction)

    return results


def print_results(results: List[Dict[str, Union[str, List[str]]]]):
    print()
    for interaction_result in results:
        speaker: str = interaction_result["speaker"]
        message: str = interaction_result["message"]
        print(f" > {speaker.capitalize()}: {message}")

        if speaker != "client":
            print()
            continue

        sentiment: str = interaction_result["sentiment"]
        intentions: List[str] = interaction_result["intentions"]

        print(f"    * Sentiment: {sentiment.title()}")
        print(f"    * Intentions: {', '.join(intention.title() for intention in intentions) }")
        print()


def main():
    parser = ArgumentParser(description="LLM Transcript Analysis")
    parser.add_argument("-c", "--configpath", required=True)
    parser.add_argument("-d", "--datapath", required=True)
    parser.add_argument("-r", "--resourcesdir", required=True)
    args = parser.parse_args()

    results = analyze(args.configpath, args.datapath, args.resourcesdir)
    print_results(results)


if __name__ == "__main__":
    main()
