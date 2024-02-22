import torch
import copy
from trl import AutoModelForCausalLMWithValueHead
# from trl import PPOTrainer, PPOConfig
from trl import DPOTrainer
from transformers import Seq2SeqTrainingArguments
from dataclasses import dataclass
from transformers import DataCollatorForSeq2Seq
from typing import Any, Dict, List, Sequence, Tuple
from datasets import load_dataset

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


# def get_preprocess_and_print_func(
#     tokenizer: "PreTrainedTokenizer",
#     template: "Template",
#     data_args: "DataArguments",
#     training_args: "Seq2SeqTrainingArguments",
#     stage: Literal["pt", "sft", "rm", "ppo"],
# ) -> Tuple[Callable, Callable]:
#     if stage == "pt":
#         preprocess_func = partial(preprocess_pretrain_dataset, tokenizer=tokenizer, data_args=data_args)
#         print_function = partial(print_unsupervised_dataset_example, tokenizer=tokenizer)
#     elif stage == "sft" and not training_args.predict_with_generate:
#         if data_args.sft_packing:
#             preprocess_func = partial(
#                 preprocess_packed_supervised_dataset, tokenizer=tokenizer, template=template, data_args=data_args
#             )
#         else:
#             preprocess_func = partial(
#                 preprocess_supervised_dataset, tokenizer=tokenizer, template=template, data_args=data_args
#             )

#         print_function = partial(print_supervised_dataset_example, tokenizer=tokenizer)
#     elif stage == "rm":
#         preprocess_func = partial(
#             preprocess_pairwise_dataset, tokenizer=tokenizer, template=template, data_args=data_args
#         )
#         print_function = partial(print_pairwise_dataset_example, tokenizer=tokenizer)
#     else:
#         preprocess_func = partial(
#             preprocess_unsupervised_dataset, tokenizer=tokenizer, template=template, data_args=data_args
#         )
#         print_function = partial(print_unsupervised_dataset_example, tokenizer=tokenizer)

#     return preprocess_func, print_function



def main():
    device = "cuda"  # cuda cpu
    model_path = "/mnt/nfs/houjing/repo/FastChat/data/interim/vicuna-7b-lora-CQ-v0-1217-epoch100/checkpoint-2500"
    kwargs = {"torch_dtype": torch.float16, "revision": 'main'}
    adapter = get_model_adapter(model_path)
    model_peft, tokenizer = adapter.load_model(model_path, kwargs)
    model_peft.to(device)

    model_peft_copy_disable = copy.deepcopy(model_peft)
    model_peft_copy_disable.disable_adapter_layers()

    model_trl: "AutoModelForCausalLMWithValueHead" = AutoModelForCausalLMWithValueHead.from_pretrained(model_peft)  # peft/transformers -> trl

    # training_args_dict = training_args.to_dict()
    # training_args_dict.update(dict(remove_unused_columns=False))  # important for pairwise dataset
    training_args_dict = dict(remove_unused_columns=False, output_dir="./")
    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    dataset = load_dataset('json', data_files='/mnt/nfs/houjing/repo/FastChat/fastchat/train/comparison_gpt4_data_en.json')

    # ignore_pad_token_for_loss = True
    # data_collator = DPODataCollatorWithPadding(
    #     tokenizer=tokenizer,
    #     pad_to_multiple_of=8,
    #     label_pad_token_id=IGNORE_INDEX if ignore_pad_token_for_loss else tokenizer.pad_token_id,
    # )

    dataset = dataset['train']

    # preprocess_func, print_function = get_preprocess_and_print_func(
    #     tokenizer, template, data_args, training_args, stage
    # )
    # column_names = list(next(iter(dataset)).keys())
    # kwargs = {}
    # if not data_args.streaming:
    #     kwargs = dict(
    #         num_proc=data_args.preprocessing_num_workers,
    #         load_from_cache_file=(not data_args.overwrite_cache),
    #         desc="Running tokenizer on dataset",
    #     )
    # dataset = dataset.map(preprocess_func, batched=True, remove_columns=column_names, **kwargs)

    print("dataset")

    dpo_trainer = DPOTrainer(model=model_trl,  # trl ->
                            tokenizer=tokenizer,
                            args=training_args,
                            train_dataset=dataset,
                            # data_collator=data_collator,
                            # device=device,
                            )
    
    generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end = load_llm_setting(model_path, model_trl)

    str_prompt = "who are you?"
    print("str_prompt: ", str_prompt)
    print("str_llm_answer: ")
    str_llm_answer = infer_llm(model_path, device, model_trl, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end, str_prompt, temperature=0)

    print("finished...")

if __name__ == "__main__":
    main()
