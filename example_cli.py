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
import re
import time

import fire
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
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
        self._completer = WordCompleter(
            words=["!!exit", "!!reset", "!!save", "!!load"], pattern=re.compile("$")
        )
        self._console = Console()
        self._multiline = multiline
        self._mouse = mouse

    def prompt_for_input(self, role) -> str:
        self._console.print(f"[bold]{role}:")

        prompt_input = self._prompt_session.prompt(
            completer=self._completer,
            multiline=False,
            mouse_support=self._mouse,
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=self.bindings if self._multiline else None,
        )
        self._console.print()
        return prompt_input

    def prompt_for_output(self, role: str):
        self._console.print(f"[bold]{role}:")

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

        return text, outputs['finish_reason'] if 'finish_reason' in outputs else text

    def print_output(self, text: str):
        self.stream_output([{"text": text}])

    def print_banner(self):
        self._console.print("=" * 60)
        self._console.print(' ' * 20, end='')
        self._console.print("[bold]ChatBaker by 42dot :car:")
        self._console.print("=" * 60)
        self._console.print("[bold]Loading ChatBaker model ...")


def chat_loop(
        chatio: RichChatIO,
        model: str,
        temperature: float,
        repetition_penalty: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
        debug: bool,
        device: str,
):
    chatio.print_banner()

    # Load a model.
    t = time.time()
    model, tokenizer, logits_processor = load_model(
        model,
        temperature,
        repetition_penalty,
        top_p,
        top_k,
        device,
    )
    chatio.print_output(f'{round(time.time() - t, 2)}s elapsed.')

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

        # Remove oldest conversation if the prompt size exceeds the limit.
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
        outputs, finish_reason = chatio.stream_output(generate_stream(
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
        ))

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
        # model='/6917396/models/v0.1.3_enko_1.3b_free_3ep_yk',
        model='/Users/H6917396/workspace/gitlab.42dot.ai/hyperai/ChatBaker/model',
        temperature=0.5,
        repetition_penalty=1.2,
        top_p=0.95,
        top_k=20,
        max_new_tokens=512,
        debug=False,
        device='cuda',
):
    try:
        chat_loop(
            RichChatIO(),
            model=model,
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
