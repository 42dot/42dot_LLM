# coding=utf-8
# Copyright 2023 42dot Inc.
#
# @author   sang.park@42dot.ai
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

import json
import subprocess
import time

import fire
import torch
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from rich import print_json
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from inference import generate_stream, load_model


# This code is heavily derived from FastChat implementation.
class RichChatIO:
    bindings = KeyBindings()

    @bindings.add("escape", "enter")
    def _(event):
        event.app.current_buffer.newline()

    def __init__(self, multiline: bool = False, mouse: bool = False):
        self._prompt_session = PromptSession(history=InMemoryHistory())
        self._console = Console()
        self._multiline = multiline
        self._mouse = mouse

    def prompt_for_input(self, role) -> str:
        self._console.print(f"[bold]{role}:")

        prompt_input = self._prompt_session.prompt(
            multiline=False,
            mouse_support=self._mouse,
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=self.bindings if self._multiline else None,
        )
        self._console.print()
        return prompt_input

    def prompt_for_output(self, role: str):
        self._console.print(f"[bold]{role}:")

    def stream_output_debug(self, output_stream):
        """Stream output for debug mode."""

        for i, outputs in enumerate(output_stream):
            # If this is the last output, stop.
            if 'candidates' not in outputs:
                break
            self._console.print("=" * 12, end='')
            self._console.print(f' #{i + 1} ', end='')
            self._console.print("=" * 12)
            # Print candidates tokens.
            for candidates in outputs['candidates']:
                for k, v in candidates.items():
                    self._console.print(f"{k}: {v}")
            self._console.print("-" * 3)
            # Print selected tokens.
            for j, selected in enumerate(outputs['selected']):
                for k, v in selected.items():
                    if j == 0:
                        # It's an actual selected token.
                        self._console.print(f"{k}: {v}", style="r")
                    else:
                        self._console.print(f"{k}: {v}")
            self._console.print()

        return outputs['text'], outputs['finish_reason']

    def stream_output(self, output_stream):
        """Stream output from a role."""

        # Create a Live context for updating the console output
        with Live(console=self._console, refresh_per_second=4) as live:
            # Read lines from the stream
            for outputs in output_stream:
                if not outputs:
                    continue
                text = outputs["text"]
                # Render the accumulated text as Markdown
                # NOTE: this is a workaround for the rendering "unstandard markdown"
                #  in rich. The chatbots output treat "\n" as a new line for
                #  better compatibility with real-world text. However, rendering
                #  in markdown would break the format. It is because standard markdown
                #  treat a single "\n" in normal text as a space.
                #  Our workaround is adding two spaces at the end of each line.
                #  This is not a perfect solution, as it would
                #  introduce trailing spaces (only) in code block, but it works well
                #  especially for console output, because in general the console does not
                #  care about trailing spaces.
                lines = []
                for line in text.splitlines():
                    lines.append(line)
                    if line.startswith("```"):
                        # Code block marker - do not add trailing spaces, as it would
                        #  break the syntax highlighting
                        lines.append("\n")
                    else:
                        lines.append("  \n")
                markdown = Markdown("".join(lines))
                # Update the Live console output
                live.update(markdown)
        self._console.print()

        return text, outputs['finish_reason']

    def print_output(self, text: str, end='\n', highlight=True, style=None):
        self._console.print(text, end=end, highlight=highlight, style=style)

    def print_banner(self):
        self._console.print("=" * 60)
        self._console.print(' ' * 20, end='')
        self._console.print("[bold]ChatBaker by 42dot :car:")
        self._console.print("=" * 60)


# This code is heavily derived from https://gist.github.com/afspies/7e211b83ca5a8902849b05ded9a10696
def assign_free_gpus(threshold_vram_usage=7000):
    """
    Assigns free gpu to the current process.
    This function should be called after all imports,
    Args:
        threshold_vram_usage (int, optional): A GPU is considered free if the vram usage is below the threshold
                                              Defaults to 7000 (MiB).
    """

    def _check():
        # Get the list of GPUs via nvidia-smi
        smi_query_result = subprocess.check_output(
            "nvidia-smi -q -d Memory | grep -A4 GPU", shell=True
        )
        # Extract the usage information
        gpu_info = smi_query_result.decode("utf-8").split("\n")
        gpu_info = list(filter(lambda info: "Used" in info, gpu_info))
        gpu_info = [
            int(x.split(":")[1].replace("MiB", "").strip()) for x in gpu_info
        ]  # Remove garbage
        # Returns the first available GPU.
        return [
            str(i) for i, mem in enumerate(gpu_info) if mem < threshold_vram_usage
        ][0]

    if not torch.cuda.is_available():
        return 'cpu'
    gpu_to_use = _check()
    if not gpu_to_use:
        return 'cpu'
    return f'cuda:{gpu_to_use}'


def chat_loop(
        chatio: RichChatIO,
        model_path: str,
        temperature: float,
        repetition_penalty: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
        debug: bool,
        device: str,
):
    chatio.print_banner()

    # Automatic CPU/GPU allocation.
    if device == 'auto':
        device = assign_free_gpus()
        if device.startswith('cuda'):
            chatio.print_output(f"[yellow]Using [u]GPU:{device[-1]}[/u][/yellow]")
        else:
            chatio.print_output("[yellow]Using [u]CPU[/u][/yellow]")

    # Load a model.
    with chatio._console.status("Loading ChatBaker model ...") as _:
        t = time.time()
        model, tokenizer, logits_processor = load_model(
            model_path,
            temperature,
            repetition_penalty,
            top_p,
            top_k,
            device,
        )
    chatio.print_output(f'ChatBaker model has been loaded. {round(time.time() - t, 2)}s elapsed.', highlight=False)

    system_prompt = (
        '호기심 많은 인간 (human)과 인공지능 봇 (AI bot)의 대화입니다. '
        '봇의 이름은 챗베이커 (ChatBaker)이고 포티투닷 (42dot)에서 개발했습니다. '
        '봇은 인간의 질문에 대해 친절하게 유용하고 상세한 답변을 제공합니다. '
    )
    conv = []
    inputs = None

    while True:
        # Make conversation history if you've had previous conversations.
        if inputs and outputs:
            conv.append(f'<human>: {inputs} <bot>: {outputs}{tokenizer.eos_token}')

        try:
            inputs = chatio.prompt_for_input('<human>')
        except EOFError:
            inputs = ""
        if not inputs:
            chatio.print_output('대화를 입력해 주세요.')
            continue

        # Remove the oldest conversation if the prompt size exceeds the limit.
        while True:
            prompt = (f'{system_prompt}'
                      f'{"".join(conv)}'
                      f'<human>: {inputs} <bot>:')
            if len(tokenizer.encode(prompt)) < 2048 - max_new_tokens:
                break
            conv.pop(0)

        # Streaming output with model generation.
        chatio.prompt_for_output('<bot>')
        t = time.time()
        if debug:
            generate_stream_func = chatio.stream_output_debug
        else:
            generate_stream_func = chatio.stream_output

        outputs, finish_reason = generate_stream_func(generate_stream(
            model,
            tokenizer,
            logits_processor,
            prompt,
            temperature,
            repetition_penalty,
            top_p,
            top_k,
            max_new_tokens,
            device,
            debug,
        ))

        # Provides a variety of information useful for debug mode.
        if debug:
            prompt_tokens = len(tokenizer.encode(prompt))
            completion_tokens = len(tokenizer.encode(outputs))
            msg = {
                "prompt": prompt,
                "outputs": outputs,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "finish_reason": finish_reason,
                "device": device,
                "speed (token/s)": round(completion_tokens / (time.time() - t), 2),
            }
            print_json(json.dumps(msg))


def main(
        # TODO: 아래는 오픈 직전에 허깅페이스 주소로 변경해야 합니다.
        model_path='/6917396/models/v0.1.3_enko_1.3b_free_3ep_yk',
        # model='/Users/H6917396/workspace/gitlab.42dot.ai/hyperai/ChatBaker/model',
        temperature=0.5,
        repetition_penalty=1.2,
        top_p=0.95,
        top_k=20,
        max_new_tokens=512,
        debug=False,
        device='auto',
):
    try:
        chat_loop(
            RichChatIO(),
            model_path=model_path,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            debug=debug,
            device=device,
        )
    except KeyboardInterrupt:
        print("exit ...")


if __name__ == '__main__':
    fire.Fire(main)
