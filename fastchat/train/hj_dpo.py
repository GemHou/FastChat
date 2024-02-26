import torch
import copy
from trl import AutoModelForCausalLMWithValueHead
# from trl import PPOTrainer, PPOConfig
# from trl import DPOTrainer
from transformers import Seq2SeqTrainingArguments
from dataclasses import dataclass
from transformers import DataCollatorForSeq2Seq
from typing import Any, Dict, List, Sequence, Tuple
from datasets import load_dataset
from transformers.tokenization_utils import PreTrainedTokenizer
import llmtuner
from llmtuner.data.preprocess import get_preprocess_and_print_func
from llmtuner.data.template import get_template_and_fix_tokenizer
from llmtuner.data.parser import get_dataset_list
from llmtuner.data.loader import load_single_dataset, merge_dataset
from llmtuner.train.dpo.trainer import CustomDPOTrainer
import random
from torch.optim import Adam
from transformers.optimization import get_constant_schedule
import time
from datasets import Dataset

from fastchat.model.model_adapter import get_model_adapter
from fastchat.serve.hj_utils_llm import load_llm_setting, infer_llm

IGNORE_INDEX = -100


@dataclass
class DPODataCollatorWithPadding(DataCollatorForSeq2Seq):
    r"""
    Data collator for pairwise data.
    """

    def _pad_labels(self, batch: torch.Tensor, positions: List[Tuple[int, int]]) -> torch.Tensor:
        padded_labels = []
        for feature, (prompt_len, answer_len) in zip(batch, positions):
            if self.tokenizer.padding_side == "left":
                start, end = feature.size(0) - answer_len, feature.size(0)
            else:
                start, end = prompt_len, prompt_len + answer_len
            padded_tensor = self.label_pad_token_id * torch.ones_like(feature)
            padded_tensor[start:end] = feature[start:end]
            padded_labels.append(padded_tensor)
        return torch.stack(padded_labels, dim=0).contiguous()  # in contiguous memory

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        concatenated_features = []
        label_positions = []
        for key in ("chosen_ids", "rejected_ids"):
            for feature in features:
                prompt_len, answer_len = len(feature["prompt_ids"]), len(feature[key])
                concatenated_features.append(
                    {
                        "input_ids": feature["prompt_ids"] + feature[key],
                        "attention_mask": [1] * (prompt_len + answer_len),
                    }
                )
                label_positions.append((prompt_len, answer_len))

        batch = self.tokenizer.pad(
            concatenated_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["labels"] = self._pad_labels(batch["input_ids"], label_positions)
        return batch
    
    def collate_batch(self, features):
        # Shuffle the features for each epoch
        random.shuffle(features)
        print("shuffle!!!")
        return super().collate_batch(features)

def prepare_args():
    start_time = time.time()
    model_path = "/mnt/nfs/houjing/repo/FastChat/data/interim/vicuna-7b-lora-CQ-v0-1217-epoch100/checkpoint-2500"
    model_args = llmtuner.hparams.ModelArguments(model_name_or_path=model_path)

    training_args_dict = dict(remove_unused_columns=False, 
                              output_dir="./",
                              per_device_train_batch_size=2,  # important to GPU memory!!!
                              per_device_eval_batch_size=1,
                              )
    training_args = Seq2SeqTrainingArguments(**training_args_dict)
    training_args.num_train_epochs = 1
    training_args.logging_steps = 1
    training_args.learning_rate = 5e-5
    del training_args.accelerator_config
    training_args.dataloader_num_workers = 1
    training_args.dataloader_prefetch_factor = 2

    data_args = llmtuner.hparams.DataArguments()
    data_args.dataset = "comparison_gpt4_en"
    data_args.dataset_dir = '/mnt/nfs/houjing/repo/FastChat/fastchat/train'
    data_args.cutoff_len = 256  # important to GPU memory!!!
    data_args.template = "default"

    finetuning_args = llmtuner.hparams.FinetuningArguments()
    print("prepare_args time: ", time.time() - start_time)
    return model_path,model_args,training_args,data_args,finetuning_args

def prepare_model(model_args):
    start_time = time.time()
    kwargs = {"torch_dtype": torch.float32, "revision": 'main'}  # float16
    adapter = get_model_adapter(model_args.model_name_or_path)
    model_peft, tokenizer = adapter.load_model(model_args.model_name_or_path, kwargs)
    model_peft.to("cuda")
    print("prepare_model time: ", time.time() - start_time)
    return model_peft,tokenizer

def transform_dataset(training_args, data_args, tokenizer, dataset):
    ignore_pad_token_for_loss = True
    data_collator = DPODataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )

    template = get_template_and_fix_tokenizer(data_args.template, tokenizer)

    stage = "rm"

    preprocess_func, print_function = get_preprocess_and_print_func(
        tokenizer, template, data_args, training_args, stage
    )
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache),
            desc="Running tokenizer on dataset",
        )
    dataset = dataset.map(preprocess_func, batched=True, remove_columns=column_names, **kwargs)
    return dataset,data_collator

def prepare_dataset_from_json(model_args, training_args, data_args, tokenizer):
    start_time = time.time()
    all_datasets = []
    for dataset_attr in get_dataset_list(data_args):  # TODO: add split
        all_datasets.append(load_single_dataset(dataset_attr, model_args, data_args))
    dataset = merge_dataset(all_datasets, data_args, training_args)

    print("prompt: ", dataset['prompt'])
    print("response: ", dataset['response'])
    print("system: ", dataset['system'])
    print("tools: ", dataset['tools'])

    dataset, data_collator = transform_dataset(training_args, data_args, tokenizer, dataset)
    print("prepare_dataset_from_json time 1: ", time.time() - start_time)
    return dataset,data_collator

def prepare_dataset_from_dict(training_args, data_args, tokenizer, dict_data):
    start_time = time.time()

    # 转换为Dataset类
    dataset = Dataset.from_dict(dict_data)

    print("prompt2: ", dataset['prompt'])
    print("response2: ", dataset['response'])
    print("system2: ", dataset['system'])
    print("tools2: ", dataset['tools'])

    dataset, data_collator = transform_dataset(training_args, data_args, tokenizer, dataset)
    print("prepare_dataset_from_dict time: ", time.time() - start_time)
    return dataset,data_collator

def prepare_trainer(training_args, finetuning_args, model_peft, tokenizer, dataset, data_collator):
    start_time = time.time()
    learning_rate = 5e-6
    optimizer = Adam(model_peft.parameters(), lr=learning_rate)
    lr_scheduler = get_constant_schedule(optimizer)

    llmtuner_dpo_trainer = CustomDPOTrainer(model=model_peft,  # only access peft model
                            tokenizer=tokenizer,
                            args=training_args,
                            train_dataset=dataset,
                            data_collator=data_collator,
                            # device=device,
                            beta=finetuning_args.dpo_beta,
                            loss_type=finetuning_args.dpo_loss,
                            ftx_gamma=finetuning_args.dpo_ftx,
                            optimizers=(optimizer, lr_scheduler),
                            )
    print("prepare_trainer time: ", time.time() - start_time)
    return llmtuner_dpo_trainer


def main():
    model_path, model_args, training_args, data_args, finetuning_args = prepare_args()  # time: 0.0018s
    
    model_peft, tokenizer = prepare_model(model_args)  # time: 42.5s

    generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end = load_llm_setting(model_path, model_peft)

    str_prompt = "who are you?"
    print("str_prompt: ", str_prompt)
    print("str_llm_answer: ")
    str_llm_answer = infer_llm(model_path, "cuda", model_peft, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end, str_prompt, temperature=0)

    dict_data = {
        "prompt": [
            [{"content": "who are you?", "role": "user"}],
            [{"content": "who are you?", "role": "user"}],
        ],
        "response": [
            [{"content": "I am a Game AI trained by Shanghai AI Laboratory.", "role": "assistant"},
            {"content": "I am Vicuna, a language model trained by researchers from Large Model Systems Organization (LMSYS).", "role": "assistant"}],
            [{"content": "I am a Game AI trained from Shanghai AI Laboratory.", "role": "assistant"},
            {"content": "I am Vicuna, a language model trained by researchers from Large Model Systems Organization (LMSYS).", "role": "assistant"}],
        ],
        "system": ["", ""],
        "tools": ["", ""],
    }

    for i in range(100):
        dataset, data_collator = prepare_dataset_from_dict(training_args, data_args, tokenizer, dict_data)  # time: 10.4s

        llmtuner_dpo_trainer = prepare_trainer(training_args, finetuning_args, model_peft, tokenizer, dataset, data_collator)  # time: 0.016s

        train_result = llmtuner_dpo_trainer.train()  # time!!!!!!!!!!!!!

        str_prompt = "who are you?"
        print("str_prompt: ", str_prompt)
        print("str_llm_answer: ")
        str_llm_answer = infer_llm(model_path, "cuda", model_peft, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end, str_prompt, temperature=0)

    print("finished...")

if __name__ == "__main__":
    main()
