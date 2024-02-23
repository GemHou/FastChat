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


def main():
    device = "cuda"  # cuda cpu
    model_path = "/mnt/nfs/houjing/repo/FastChat/data/interim/vicuna-7b-lora-CQ-v0-1217-epoch100/checkpoint-2500"
    kwargs = {"torch_dtype": torch.float32, "revision": 'main'}  # float16
    adapter = get_model_adapter(model_path)
    model_peft, tokenizer = adapter.load_model(model_path, kwargs)
    model_peft.to(device)

    # model_peft_copy_disable = copy.deepcopy(model_peft)  # important to GPU memory!!!
    # model_peft_copy_disable.disable_adapter_layers()

    # model_trl: "AutoModelForCausalLMWithValueHead" = AutoModelForCausalLMWithValueHead.from_pretrained(model_peft)  # peft/transformers -> trl

    # training_args_dict = training_args.to_dict()
    # training_args_dict.update(dict(remove_unused_columns=False))  # important for pairwise dataset
    training_args_dict = dict(remove_unused_columns=False, 
                              output_dir="./",
                              per_device_train_batch_size=8,  # important to GPU memory!!!
                              per_device_eval_batch_size=1,
                              )
    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    data_args = llmtuner.hparams.DataArguments()
    model_args = llmtuner.hparams.ModelArguments(model_name_or_path=model_path)

    # dataset = load_dataset('json', data_files='/mnt/nfs/houjing/repo/FastChat/fastchat/train/comparison_gpt4_data_en.json')
    # dataset = dataset['train']

    data_args.dataset = "comparison_gpt4_en"
    data_args.dataset_dir = '/mnt/nfs/houjing/repo/FastChat/fastchat/train'
    data_args.cutoff_len = 256  # important to GPU memory!!!
    all_datasets = []
    for dataset_attr in get_dataset_list(data_args):  # TODO: add split
        all_datasets.append(load_single_dataset(dataset_attr, model_args, data_args))
    dataset = merge_dataset(all_datasets, data_args, training_args)

    ignore_pad_token_for_loss = True
    data_collator = DPODataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )

    data_args.template = "default"

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

    print("dataset")

    del training_args.accelerator_config

    # training_args.train_batch_size=1
    # training_args.eval_batch_size=1
    # training_args.per_device_train_batch_size=1
    # training_args.per_device_eval_batch_size=1

    # trl_dpo_trainer = DPOTrainer(model=model_trl,  # trl ->
    #                         tokenizer=tokenizer,
    #                         args=training_args,
    #                         train_dataset=dataset,
    #                         # data_collator=data_collator,
    #                         # device=device,
    #                         )
    
    training_args.dataloader_num_workers = 1
    training_args.dataloader_prefetch_factor = 2
    finetuning_args = llmtuner.hparams.FinetuningArguments()
    llmtuner_dpo_trainer = CustomDPOTrainer(model=model_peft,  # only access peft model
                            tokenizer=tokenizer,
                            args=training_args,
                            train_dataset=dataset,
                            data_collator=data_collator,
                            # device=device,
                            beta=finetuning_args.dpo_beta,
                            loss_type=finetuning_args.dpo_loss,
                            ftx_gamma=finetuning_args.dpo_ftx,
                            )
    
    generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end = load_llm_setting(model_path, model_peft)

    str_prompt = "who are you?"
    print("str_prompt: ", str_prompt)
    print("str_llm_answer: ")
    str_llm_answer = infer_llm(model_path, device, model_peft, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end, str_prompt, temperature=0)

    # train_result = trl_dpo_trainer.train()
    train_result = llmtuner_dpo_trainer.train()

    str_prompt = "who are you?"
    print("str_prompt: ", str_prompt)
    print("str_llm_answer: ")
    str_llm_answer = infer_llm(model_path, device, model_peft, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end, str_prompt, temperature=0)

    print("finished...")

if __name__ == "__main__":
    main()
