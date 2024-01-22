import transformers
from transformers import Trainer
from dataclasses import dataclass, field
import typing
import json

from fastchat.train.train import DataArguments, make_supervised_data_module, LazySupervisedDataset, SupervisedDataset


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: typing.Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    flash_attn: bool = False

def main():
    print("hello world")

    model_name_or_path = "/mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-13b-v1.5"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        cache_dir=None,
        device_map=None
    )
    print("got model")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=None,
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
    )
    print("got tokenizer")

    parser = transformers.HfArgumentParser((DataArguments, TrainingArguments))
    (data_args, training_args) = parser.parse_args_into_dataclasses()

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    print("got trainer")

    eval_data_path = "./data/raw/data_date121314_dataNum911.json"
    eval_json = json.load(open(eval_data_path, "r"))
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer)
    print("got eval_dataset")

    ignore_keys_for_eval = None
    dataset_metrics = trainer.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_temp",
                    )
    print("dataset_metrics: ", dataset_metrics)
    
    print("finished...")


if __name__ == "__main__":
    main()
