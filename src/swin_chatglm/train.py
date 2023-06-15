"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import argparse
import datetime
import json
import os
import random
from io import BytesIO
from os.path import basename
from pathlib import Path

import sys

from transformers import GPT2Tokenizer


sys.path.append("./")
from my_models.swin_chatglm.tokenization_chatglm import ChatGLMTokenizer

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.plugins import CheckpointIO
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.seed import seed_everything
from sconf import Config

from my_models.swin_chatglm.util import Chatglm2DonutDataset
from my_models.swin_chatglm.lightning_module import DonutDataPLModule, DonutModelPLModule


class CustomCheckpointIO(CheckpointIO):
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        del checkpoint["state_dict"]
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, storage_options=None):
        checkpoint = torch.load(path + "artifacts.ckpt")
        state_dict = torch.load(path + "pytorch_model.bin")
        checkpoint["state_dict"] = {"model." + key: value for key, value in state_dict.items()}
        return checkpoint

    def remove_checkpoint(self, path) -> None:
        return super().remove_checkpoint(path)


@rank_zero_only
def save_config_file(config, path):
    if not Path(path).exists():
        os.makedirs(path)
    save_path = Path(path) / "config.yaml"
    print(config.dumps())
    with open(save_path, "w") as f:
        f.write(config.dumps(modified_color=None, quote_str=True))
        print(f"Config is saved at {save_path}")


def train(config):
    seed_everything(config.get("seed", 42), workers=True)

    data_module = DonutDataPLModule(config)

    # add datasets to data_module
    datasets = {"train": [], "dev": []}
    special_tokens = ["<sep/>"]
    for i, dataset_name_or_path in enumerate(config.dataset_name_or_paths):
        task_name = os.path.basename(dataset_name_or_path)  # e.g., cord-v2, docvqa, rvlcdip, ...
        
        # add categorical special tokens (optional)
        if task_name == "rvlcdip":
            special_tokens.extend([
                "<advertisement/>", "<budget/>", "<email/>", "<file_folder/>", 
                "<form/>", "<handwritten/>", "<invoice/>", "<letter/>", 
                "<memo/>", "<news_article/>", "<presentation/>", "<questionnaire/>", 
                "<resume/>", "<scientific_publication/>", "<scientific_report/>", "<specification/>"
            ])
        if task_name == "docvqa":
            special_tokens.extend(["<yes/>", "<no/>"])

        task_start_token = f"<s_{task_name}>"
        prompt_end_token = f"</s_{task_name}>"
        if task_start_token not in special_tokens:
            special_tokens.extend(task_start_token)
        if prompt_end_token not in special_tokens:
            special_tokens.extend(prompt_end_token)

    tokenizer = ChatGLMTokenizer.from_pretrained(
        "/public/home/xlwang2/codes/Med_Prompts/models--THUDM--chatglm-6b/snapshots/a8ede826cf1b62bd3c78bdfb3625c7c5d2048fbd",
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": sorted(set(special_tokens))}
    )
    tokenizer.pad_token = tokenizer.eos_token

    for i, dataset_name_or_path in enumerate(config.dataset_name_or_paths):
        task_name = os.path.basename(dataset_name_or_path)  # e.g., cord-v2, docvqa, rvlcdip, ...
        for split in ["train", "dev"]:
            task_start_token = f"<s_{task_name}>"
            prompt_end_token = f"</s_{task_name}>"

            datasets[split].append(
                Chatglm2DonutDataset(
                    config=config,
                    dataset_name_or_path=os.path.join(dataset_name_or_path, f"fold_{split}"),
                    tokenizer=tokenizer,
                    max_length=config.max_length,
                    split=split,
                    sort_json_key=config.sort_json_key,
                    task_start_token=task_start_token,
                    prompt_end_token=prompt_end_token,
                )
            )
            # prompt_end_token is used for ignoring a given prompt in a loss function
            # for docvqa task, i.e., {"question": {used as a prompt}, "answer": {prediction target}},
            # set prompt_end_token to "<s_answer>"
    data_module.train_datasets = datasets["train"]
    data_module.val_datasets = datasets["dev"]

    # model pl module
    model_module = DonutModelPLModule(config)
    model_module.model.decoder.model.tokenizer = tokenizer
    model_module.model.decoder.model.resize_token_embeddings(len(tokenizer))

    logger = TensorBoardLogger(
        save_dir=config.result_path,
        name=config.exp_name,
        version=config.exp_version,
        default_hp_metric=False,
    )

    lr_callback = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_metric",
        dirpath=Path(config.result_path) / config.exp_name / config.exp_version,
        filename="artifacts",
        save_top_k=1,
        save_last=False,
        mode="min",
    )

    custom_ckpt = CustomCheckpointIO()
    trainer = pl.Trainer(
        # resume_from_checkpoint=config.get("resume_from_checkpoint_path", None),
        num_nodes=config.get("num_nodes", 1),
        # gpus=torch.cuda.device_count(),
        strategy="ddp",
        accelerator="gpu",
        plugins=custom_ckpt,
        max_epochs=config.max_epochs,
        max_steps=config.max_steps,
        val_check_interval=config.val_check_interval,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        gradient_clip_val=config.gradient_clip_val,
        precision=16,
        num_sanity_val_steps=0,
        logger=logger,
        callbacks=[lr_callback, checkpoint_callback],
    )

    trainer.fit(model_module, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--exp_version", type=str, required=False)
    args, left_argv = parser.parse_known_args()

    config = Config(args.config)
    config.argv_update(left_argv)

    config.exp_name = basename(args.config).split(".")[0]
    config.exp_version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if not args.exp_version else args.exp_version

    save_config_file(config, Path(config.result_path) / config.exp_name / config.exp_version)
    train(config)
