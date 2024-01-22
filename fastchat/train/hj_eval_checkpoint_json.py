import transformers
from transformers import Trainer
from dataclasses import dataclass, field
import typing
import json
import torch

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
    model_path: str = field(default=None, metadata={"help": "checkpoint dir"})
    model_name_or_path: str = field(default=None, metadata={"help": "model dir"})

def main():
    print("hello world")

    parser = transformers.HfArgumentParser((DataArguments, TrainingArguments))
    (data_args, training_args) = parser.parse_args_into_dataclasses()
    print("got args")

    if training_args.model_path is None:
        assert training_args.model_name_or_path is not None
        # model_name_or_path = "/mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-13b-v1.5"
        model = transformers.AutoModelForCausalLM.from_pretrained(
            training_args.model_name_or_path,
            cache_dir=None,
            device_map=None
        )
        print("got model")

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            training_args.model_name_or_path,
            cache_dir=None,
            model_max_length=2048,
            padding_side="right",
            use_fast=False,
        )
        print("got tokenizer")
    else:
        assert training_args.model_name_or_path is None
        from fastchat.model.model_adapter import get_model_adapter
        model_path = training_args.model_path
        adapter = get_model_adapter(model_path)
        kwargs = {"torch_dtype": torch.float16}
        model, tokenizer = adapter.load_model(model_path, kwargs)

    data_args.data_path = data_args.eval_data_path
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    print("got trainer")

    eval_json = json.load(open(data_args.eval_data_path, "r"))
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    total_eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer)
    print("got total_eval_dataset")

    ignore_keys_for_eval = None
    dataset_metrics = trainer.evaluate(
                        eval_dataset=total_eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_temp",
                    )
    print("dataset_metrics: ", dataset_metrics)
    
    print("finished...")


if __name__ == "__main__":
    main()
