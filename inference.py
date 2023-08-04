# coding=utf-8
# Copyright 2023 42dot Inc.
#
# @author: sang.park@42dot.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


# This code is heavily derived from FastChat implementation.
def prepare_logits_processor(
        temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


def load_model(
        model: str,
        temperature: float,
        repetition_penalty: float,
        top_p: float,
        top_k: int,
        device: str,
):
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, add_bos_token=True)
    model = AutoModelForCausalLM.from_pretrained(model).to(device)

    return model, tokenizer, prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )


@torch.inference_mode()
def generate_stream(
        model,
        tokenizer,
        logits_processor,
        prompt,
        temperature: float,
        repetition_penalty: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
        device: str,
):
    input_ids = tokenizer(prompt).input_ids
    output_ids = list(input_ids)
    input_echo_len = len(input_ids)
    stream_interval = 2

    for i in range(max_new_tokens):
        if i == 0:
            out = model(
                input_ids=torch.as_tensor([input_ids], device=device),
                use_cache=True,
            )
            logits = out.logits
            past_key_values = out.past_key_values
        else:  # Use past_key_values and generate only one token for speed improvement.
            out = model(
                input_ids=torch.as_tensor([[last_token]], device=device),
                use_cache=True,
                past_key_values=past_key_values,
            )
            logits = out.logits
            past_key_values = out.past_key_values

        # If repetition_penalty is set, inject all_output_ids.
        if repetition_penalty > 1.0:
            all_output_ids = torch.as_tensor([output_ids], device=device)
        else:
            all_output_ids = None

        # Transformer's LogitsProcessor using the next token's logit.
        last_token_logits = logits_processor(all_output_ids, logits[:, -1, :])[0]

        if temperature < 1e-5 or top_p < 1e-8:  # Greedy for very small option value.
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            # Sampled from the multinomial probability distribution.
            indices = torch.multinomial(probs, num_samples=2)
            tokens = [int(token) for token in indices.tolist()]

        last_token = tokens[0]
        output_ids.append(last_token)

        stopped = True if last_token == tokenizer.eos_token_id else False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            tmp_output_ids = output_ids[input_echo_len:]
            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )

            yield {
                "text": output,
                "finish_reason": None,
            }

        if stopped:
            break

    if i == max_new_tokens - 1:
        finish_reason = "max_new_tokens"
    elif stopped:
        finish_reason = "endoftext"
    else:
        finish_reason = "unknown"

    yield {
        "text": output,
        "output_ids": output_ids,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
    }

    # Clean
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()
