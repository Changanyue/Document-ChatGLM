
import json
import os
import random
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import PIL
from PIL import Image
from PIL import ImageOps

import torch
from datasets import load_dataset, load_from_disk
from nltk import edit_distance

import numpy as np


from torch.utils.data import Dataset
from transformers.modeling_utils import PreTrainedModel

from torchvision import transforms
from torchvision.transforms.functional import resize, rotate


def json2token(obj: Any,
               sort_json_key: bool = True,
               tokenizer=None):
    """
    Convert an ordered JSON object into a token sequence
    """
    if type(obj) == dict:
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        else:
            output = ""
            if sort_json_key:
                keys = sorted(obj.keys(), reverse=True)
            else:
                keys = obj.keys()
            for k in keys:
                output += (
                        fr"<s_{k}>"
                        + json2token(obj[k], sort_json_key)
                        + fr"</s_{k}>"
                )
            return output
    elif type(obj) == list:
        return r"<sep/>".join(
            [json2token(item, sort_json_key) for item in obj]
        )
    else:
        obj = str(obj)
        if f"<{obj}/>" in tokenizer.all_special_tokens:
            obj = f"<{obj}/>"  # for categorical special tokens
        return obj


def token2json(tokens,
               tokenizer=None,
               is_inner_value=False):
    """
    Convert a (generated) token seuqnce into an ordered JSON format
    """
    print("tokenizer.get_added_vocab(): ", tokenizer.get_added_vocab())

    output = dict()

    while tokens:
        start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
        if start_token is None:
            break
        key = start_token.group(1)
        end_token = re.search(fr"</s_{key}>", tokens, re.IGNORECASE)
        start_token = start_token.group()
        if end_token is None:
            tokens = tokens.replace(start_token, "")
        else:
            end_token = end_token.group()
            start_token_escaped = re.escape(start_token)
            end_token_escaped = re.escape(end_token)
            content = re.search(f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE)
            if content is not None:
                content = content.group(1).strip()
                if r"<s_" in content and r"</s_" in content:  # non-leaf node
                    value = token2json(content, is_inner_value=True)
                    if value:
                        if len(value) == 1:
                            value = value[0]
                        output[key] = value
                else:  # leaf nodes
                    output[key] = []
                    for leaf in content.split(r"<sep/>"):
                        leaf = leaf.strip()
                        if (
                                leaf in tokenizer.get_added_vocab()
                                and leaf[0] == "<"
                                and leaf[-2:] == "/>"
                        ):
                            leaf = leaf[1:-2]  # for categorical special tokens
                        output[key].append(leaf)
                    if len(output[key]) == 1:
                        output[key] = output[key][0]

            tokens = tokens[tokens.find(end_token) + len(end_token):].strip()
            if tokens[:6] == r"<sep/>":  # non-leaf nodes
                return [output] + token2json(tokens[6:], is_inner_value=True)

    if len(output):
        return [output] if is_inner_value else output
    else:
        return [] if is_inner_value else {"text_sequence": tokens}


from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )

def prepare_input(img: PIL.Image.Image, random_padding: bool = False, config=None) -> torch.Tensor:
    """
    Convert PIL Image to tensor according to specified input_size after following steps below:
        - resize
        - rotate (if align_long_axis is True and image is not aligned longer axis with canvas)
        - pad
    """
    img = img.convert("RGB")
    if config.align_long_axis and (
        (config.input_size[0] > config.input_size[1] and img.width > img.height)
        or (config.input_size[0] < config.input_size[1] and img.width < img.height)
    ):
        img = rotate(img, angle=-90, expand=True)
    img = resize(img, min(config.input_size))
    img.thumbnail((config.input_size[1], config.input_size[0]))

    delta_width = config.input_size[1] - img.width
    delta_height = config.input_size[0] - img.height

    if random_padding:
        pad_width = np.random.randint(low=0, high=delta_width + 1)
        pad_height = np.random.randint(low=0, high=delta_height + 1)
    else:
        pad_width = delta_width // 2
        pad_height = delta_height // 2

    padding = (
        pad_width,
        pad_height,
        delta_width - pad_width,
        delta_height - pad_height,
    )
    return to_tensor(ImageOps.expand(img, padding))


class GPT2DonutDataset(Dataset):
    """
        DonutDataset which is saved in huggingface datasets format. (see details in https://huggingface.co/docs/datasets)
        Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt),
        and it will be converted into input_tensor(vectorized image) and input_ids(tokenized string)

        Args:
            dataset_name_or_path: name of dataset (available at huggingface.co/datasets) or the path containing image files and metadata.jsonl
            ignore_id: ignore_index for torch.nn.CrossEntropyLoss
            task_start_token: the special token to be fed to the decoder to conduct the target task
        """

    def __init__(
            self,
            config,
            dataset_name_or_path: str,
            tokenizer: PreTrainedModel,
            max_length: int,
            split: str = "train",
            ignore_id: int = -100,
            task_start_token: str = "<s>",
            prompt_end_token: str = None,
            sort_json_key: bool = True,
    ):
        super().__init__()

        self.config = config

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token if prompt_end_token else task_start_token
        self.sort_json_key = sort_json_key

        # load dataset
        self.dataset = load_from_disk(dataset_name_or_path)

        self.dataset_length = len(self.dataset)

        self.gt_token_sequences = []
        for sample in self.dataset:
            ground_truth = json.loads(sample["ground_truth"])
            # print("ground_truth: ", ground_truth)

            if "gt_parses" in ground_truth:  # when multiple ground truths are available, e.g., docvqa
                assert isinstance(ground_truth["gt_parses"], list)
                gt_jsons = ground_truth["gt_parses"]
            else:
                assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
                gt_jsons = [ground_truth["gt_parse"]]

            self.gt_token_sequences.append(
                [
                    task_start_token
                    + json2token(
                        gt_json,
                        sort_json_key=self.sort_json_key,
                        tokenizer=self.tokenizer
                    )
                    + self.tokenizer.eos_token
                    for gt_json in gt_jsons  # load json from list of json
                ]
            )

        self.prompt_end_token_id = self.tokenizer.convert_tokens_to_ids(self.prompt_end_token)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels.
        Convert gt data into input_ids (tokenized string)

        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        sample = self.dataset[idx]
        # print("sample: ", sample.keys())

        # input_tensor
        image_tensor = prepare_input(
            Image.open(os.path.join("/public/home/xlwang2/codes/LLM-document", sample["image_path"])),
            random_padding=self.split == "train",
            config=self.config
        )

        # input_ids
        # can be more than one, e.g., DocVQA Task 1
        # 对信息抽取任务，肯定是只有一个
        processed_parse = random.choice(self.gt_token_sequences[idx])
        input_ids = self.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        # print("input_ids: ", input_ids)

        if self.split == "train":
            labels = input_ids.clone()
            labels[
                labels == self.tokenizer.pad_token_id
                ] = self.ignore_id  # model doesn't need to predict pad token
            labels[
                : torch.nonzero(labels == self.prompt_end_token_id).sum() + 1
            ] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
            return image_tensor, input_ids, labels
        else:
            prompt_end_index = torch.nonzero(
                input_ids == self.prompt_end_token_id
            ).sum()  # return prompt end index instead of target output labels
            return image_tensor, input_ids, prompt_end_index, processed_parse
