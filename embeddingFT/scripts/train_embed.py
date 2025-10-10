#!/usr/bin/env python3
"""
Simple Embedding Fine-Tuning Script
Based on the provided clean script with config-driven parameters
"""

import os
import logging
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainingArguments, SentenceTransformerTrainer
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.util import mine_hard_negatives
from sentence_transformers.training_args import BatchSamplers
from huggingface_hub import login as hf_login
import yaml

# Load config
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_env(cache_dir="/models"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_CACHE"] = cache_dir

def prepare_datasets(dataset_name="ayushexel/xyneft", sample_size=1080, test_size=250, seed=12):
    ds = load_dataset(dataset_name, split="train").select(range(sample_size))
    split = ds.train_test_split(test_size=test_size, seed=seed)
    return ds, split["train"], split["test"]

def mine_negatives(dataset, embed_model, num_negatives=1, batch_size=512, **kwargs):
    return mine_hard_negatives(
        dataset,
        embed_model,
        num_negatives=num_negatives,
        batch_size=batch_size,
        output_format="triplet",
        **kwargs
    )

def evaluate_baseline(model, ds_full, ds_eval, mining_model_name, evaluator_name="baseline"):
    embed_model_cpu = SentenceTransformer(mining_model_name, device="cpu")
    hard_eval = mine_negatives(
        ds_eval,
        embed_model=embed_model_cpu,
        corpus=list(ds_full["answer"]),
        include_positives=True
    )
    evaluator = TripletEvaluator(
        anchors=hard_eval["query"],
        positives=hard_eval["answer"],
        negatives=hard_eval["negative_1"],
        name=evaluator_name
    )
    logging.info(f"Evaluating baseline with {evaluator_name} evaluator")
    results = evaluator(model)
    logging.info(f"Baseline results: {results}")
    return results

def train_model(config):
    setup_env(config.get("cache_dir", "/models"))
    
    # Load data
    ds_full, ds_train, ds_eval = prepare_datasets(
        dataset_name=config["dataset_name"],
        sample_size=config["sample_size"],
        test_size=config["test_size"],
        seed=config["seed"]
    )
    
    # Load models
    model_name = config["model_name"]
    model = SentenceTransformer(model_name)
    
    evaluate_baseline(model, ds_full, ds_eval, config["mining_model_name"], evaluator_name="baseline_pre_training")
    
    embed_model = SentenceTransformer(config["mining_model_name"], device="cpu")
    
    # Mine hard negatives for training
    hard_train = concatenate_datasets([
        mine_negatives(ds_train.select(range(i, min(i + 200_000, len(ds_train)))), embed_model, margin=0, range_min=0, range_max=100, sampling_strategy="top")
        for i in range(0, len(ds_train), 200_000)
    ])
    
    # Mine hard negatives for evaluation
    hard_eval = mine_negatives(ds_eval, embed_model, corpus=list(ds_full["answer"]), include_positives=True)
    
    # Setup training
    loss = MultipleNegativesRankingLoss(model)
    evaluator = TripletEvaluator(
        anchors=hard_eval["query"],
        positives=hard_eval["answer"],
        negatives=hard_eval["negative_1"],
        name="ft-dev"
    )
    
    args = SentenceTransformerTrainingArguments(
        output_dir=os.path.join(config["cache_dir"], f"{model_name.split('/')[-1]}-{config['num_epochs']}e"),
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        fp16=False,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=2,
        logging_steps=10,
        run_name=f"xynft-{model_name.split('/')[-1]}-{config['num_epochs']}e",
        report_to="none"  # Local logging only
    )
    
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=hard_train,
        loss=loss,
        evaluator=evaluator,
    )
    
    trainer.train()
    evaluator(trainer.model)
    final_dir = os.path.join(config["cache_dir"], f"{args.run_name}/model/final")
    trainer.model.save_pretrained(final_dir)
    trainer.model.push_to_hub(args.run_name)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config file path")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    
    # Load config and train
    config = load_config(args.config)
    train_model(config)
