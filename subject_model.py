from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any
from dataclasses import dataclass
import functools
import torch
import re
import seaborn as sns

@dataclass
class AblationSetting:
    layer: int
    neurons: list[int]

####################################################################################################

class ModelSubject:
    def __init__(self, model_name) -> None:
        self.model_name: str = model_name
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def register_forward_hook(self, container: Any) -> None:
        raise NotImplementedError("You should implement this method in a subclass.")

    def inference(self, inputs: list[str]) -> Any:
        raise NotImplementedError("You should implement this method in a subclass.")

####################################################################################################

class ModelSubject_GPT2(ModelSubject):
    def __init__(self, model_name) -> None:
        assert model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], 'model_name should be one of gpt2, gpt2-medium, gpt2-large, gpt2-xl'
        super().__init__(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.n_layer: int = self.model.config.n_layer
        self.n_neuron: int = self.model.config.n_embd * 4
        self.n_vocab: int = self.model.config.vocab_size
        self.handles: list = []

    def inference(self, inputs: list[str]) -> Any:
        # For activation times collection
        # hook register function
        def register_forward_hook(container: Any, input_lens: list[int]) -> None:
            def hook(module, input, output, i, container):
                # input: (batch_size, seq_len, n_neuron)
                for j in range(input[0].shape[0]):
                    max_values = input[0][j, :input_lens[j], :].max(dim=0)[0]
                    container[i] += (max_values > 0).float()

            for i in range(self.n_layer):
                handle = self.model.transformer.h[i].mlp.c_proj.register_forward_hook(
                    functools.partial(hook, i=i, container=container)
                )
                self.handles.append(handle)
        
        # inference process
        encoded_inputs = self.tokenizer(inputs, return_tensors='pt', padding=True, truncation=True, return_attention_mask=True)
        input_ids = encoded_inputs['input_ids'].to(self.device)
        input_lens = encoded_inputs['attention_mask'].sum(dim=1).tolist()

        container = torch.zeros(self.n_layer, self.n_neuron, device=self.device)

        register_forward_hook(container, input_lens)

        self.model(input_ids)

        for handle in self.handles:
            handle.remove()
        self.handles = []

        return container

    def inference_ablation(self, inputs: list[str], ablations: list[AblationSetting]):
        # For ablation
        # hook register function
        def register_ablation_hook(ablation: AblationSetting) -> None:
            def hook(module, input, output):
                with torch.no_grad():
                    for neuron in ablation.neurons:
                        output[0, :, neuron] = 0

            handle = self.model.transformer.h[ablation.layer].mlp.c_fc.register_forward_hook(hook)
            self.handles.append(handle)

        # inference process
        encoded_inputs = self.tokenizer(inputs, return_tensors='pt', padding=True, truncation=True, return_attention_mask=True)
        input_ids = encoded_inputs['input_ids'].to(self.device)
        input_lens = encoded_inputs['attention_mask'].sum(dim=1).tolist()

        for ablation in ablations:
            register_ablation_hook(ablation)

        output = self.model(input_ids)

        for handle in self.handles:
            handle.remove()
        self.handles = []

        logit_list = []
        for i in range(len(inputs)):
            logits = output.logits[i, input_lens[i] - 1, :].to('cpu')
            # probs = torch.softmax(logits, dim=-1)
            # logit_list.append(probs)
            logit_list.append(logits)

        return logit_list