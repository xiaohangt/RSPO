"""
a very crappy script to upload a model to the hub.
Usage:
python scripts/upload_model.py --model_name_or_path checkpoints/sppo_reverseklnoent-0.5-PromptABC-Mistral-7B-Instruct-SPPO-Iter2 --output_dir .

The "output_dir" option is a placeholder just to make the script work.
"""
import logging
import torch
import sys
import yaml
from transformers import AutoModelForCausalLM, set_seed
from peft import PeftConfig, PeftModel
from alignment import (H4ArgumentParser, ModelArguments, DataArguments, SPPOConfig,
                       get_kbit_device_map, get_quantization_config, is_adapter_model,
                       get_tokenizer)


logger = logging.getLogger(__name__)

def setup_model(model_args, training_args):
    # torch_dtype = (
    #     model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    # )
    torch_dtype = torch.bfloat16
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = model_args.model_name_or_path
    if is_adapter_model(model, model_args.model_revision):
        logger.info(f"Loading SFT adapter for {model_args.model_name_or_path=}")
        peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)
        model_kwargs = dict(
            revision=model_args.base_model_revision,
            trust_remote_code=model_args.trust_remote_code,
            use_flash_attention_2=model_args.use_flash_attention_2,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            device_map=get_kbit_device_map() if quantization_config is not None else None,
            quantization_config=quantization_config,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(
            base_model,
            model_args.model_name_or_path,
            revision=model_args.model_revision,
        )
        model_kwargs = None

    # ref_model = model_args.ref_model_name_or_path
    ref_model = model_args.model_name_or_path
    ref_model_kwargs = model_kwargs

    # if model_args.use_peft:
    #     ref_model = None
    #     ref_model_kwargs = None

    return model, ref_model, model_kwargs, ref_model_kwargs



parser = H4ArgumentParser((ModelArguments, DataArguments, SPPOConfig))
model_args, data_args, training_args = parser.parse()


model_name, ref_model_name, model_kwargs, ref_model_kwargs = setup_model(model_args, training_args)

model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
# model = AutoModelForCausalLM.from_pretrained('UCLA-AGI/Mistral7B-PairRM-SPPO-Iter1', **model_kwargs)

# Check the dtype of model parameters
# for name, param in model.named_parameters():
#     print(f"Parameter: {name}, dtype: {param.dtype}")



print(model_name)
model_name = model_name.split('/')[-1]
print(model_name)
model.push_to_hub(f'RegularizedSelfPlay/{model_name}', private=False)

tokenizer = get_tokenizer(model_args, data_args)
tokenizer.push_to_hub(f'RegularizedSelfPlay/{model_name}', private=False)
