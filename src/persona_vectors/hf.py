from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from persona_vectors.types import GeneratedSample


class HuggingFaceCausalLMBackend:
    def __init__(
        self,
        model_name: str,
        *,
        device_map: str | dict[str, Any] = "auto",
        torch_dtype: str | torch.dtype = "auto",
        load_in_4bit: bool = False,
        trust_remote_code: bool = True,
    ) -> None:
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        self._layers = _resolve_layers(self.model)
        self.num_layers = len(self._layers)

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None,
    ) -> GeneratedSample:
        prompt_ids = self._prompt_input_ids(system_prompt=system_prompt, user_prompt=user_prompt)
        prompt_ids = prompt_ids.to(self.model.device)

        generation_kwargs: dict[str, Any] = {
            "input_ids": prompt_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else 1.0,
            "top_p": top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if seed is not None:
            generator = torch.Generator(device=prompt_ids.device)
            generator.manual_seed(seed)
            generation_kwargs["generator"] = generator

        with torch.inference_mode():
            try:
                outputs = self.model.generate(**generation_kwargs)
            except ValueError as exc:
                if "['generator']" not in str(exc):
                    raise
                generation_kwargs.pop("generator", None)
                if seed is not None:
                    torch.manual_seed(seed)
                outputs = self.model.generate(**generation_kwargs)

        generated = outputs[0, prompt_ids.shape[1]:].detach().cpu()
        generated = _strip_terminal_special_tokens(generated, self.tokenizer)
        return GeneratedSample(
            prompt_token_ids=prompt_ids[0].detach().cpu().tolist(),
            response_token_ids=generated.tolist(),
            response_text=self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
        )

    def response_layer_means(self, *, prompt_token_ids: list[int], response_token_ids: list[int]) -> list[list[float]]:
        if not response_token_ids:
            raise ValueError("Response has no decodable tokens; cannot compute response-average activations")

        all_ids = torch.tensor([prompt_token_ids + response_token_ids], device=self.model.device)
        response_length = len(response_token_ids)
        layer_means: list[list[float]] = [[] for _ in self._layers]
        handles: list[torch.utils.hooks.RemovableHook] = []

        for idx, layer_module in enumerate(self._layers):
            def _make_hook(layer_idx: int) -> Any:
                def _hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: Any) -> None:
                    hidden = output[0] if isinstance(output, tuple) else output
                    response_slice = hidden[:, -response_length:, :].mean(dim=1)[0]
                    layer_means[layer_idx] = response_slice.detach().float().cpu().tolist()
                return _hook
            handles.append(layer_module.register_forward_hook(_make_hook(idx)))

        try:
            with torch.inference_mode():
                self.model(input_ids=all_ids, output_hidden_states=False, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        return layer_means

    def steering_scope(self, *, layer_index: int, vector: list[float], alpha: float) -> AbstractContextManager[None]:
        if layer_index < 1 or layer_index > self.num_layers:
            raise ValueError(f"layer_index must be in [1, {self.num_layers}]")
        if alpha == 0:
            return nullcontext()

        layer_module = self._layers[layer_index - 1]
        vector_tensor = torch.tensor(vector, dtype=self.model.dtype, device=self.model.device).view(1, 1, -1)

        @contextmanager
        def _scope() -> Any:
            def hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: Any) -> Any:
                if isinstance(output, tuple):
                    hidden_state = output[0]
                    adjusted = hidden_state.clone()
                    adjusted[:, -1:, :] = adjusted[:, -1:, :] + alpha * vector_tensor
                    return (adjusted, *output[1:])
                adjusted = output.clone()
                adjusted[:, -1:, :] = adjusted[:, -1:, :] + alpha * vector_tensor
                return adjusted

            handle = layer_module.register_forward_hook(hook)
            try:
                yield
            finally:
                handle.remove()

        return _scope()

    def _prompt_input_ids(self, *, system_prompt: str, user_prompt: str) -> torch.Tensor:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        if getattr(self.tokenizer, "chat_template", None):
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered = _fallback_chat_template(messages)

        batch = self.tokenizer(rendered, return_tensors="pt")
        return batch["input_ids"]


def _resolve_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    raise TypeError("Unsupported model architecture; add a layer resolver for this model family")


def _fallback_chat_template(messages: list[dict[str, str]]) -> str:
    pieces = []
    for message in messages:
        pieces.append(f"{message['role'].upper()}: {message['content'].strip()}")
    pieces.append("ASSISTANT:")
    return "\n\n".join(pieces)


def _strip_terminal_special_tokens(token_ids: torch.Tensor, tokenizer: Any) -> torch.Tensor:
    special_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    values = token_ids.tolist()
    while values and values[-1] in special_ids:
        values.pop()
    return torch.tensor(values, dtype=token_ids.dtype) if values != token_ids.tolist() else token_ids
