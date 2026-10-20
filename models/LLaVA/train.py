
import os
import json
import argparse
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from qwen_vl_utils import process_vision_info


class LLaVA_SOD_Dataset(Dataset):

    def __init__(
        self,
        json_path: str,
        img_dir: str,
        processor,
    ):
        self.img_dir = Path(img_dir)
        self.processor = processor

        with open(json_path, "r", encoding="utf-8") as file:
            self.data = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = self.img_dir / item["image"]

        human_text = item["conversations"][0]["value"]
        gpt_text = item["conversations"][1]["value"]

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str(img_path),
                    },
                    {
                        "type": "text",
                        "text": human_text.replace("<image>\n", ""),
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": gpt_text,
                    }
                ],
            },
        ]

        image_inputs, video_inputs = process_vision_info(messages)

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        return {
            "text": text,
            "images": image_inputs,
            "videos": video_inputs,
        }


@dataclass
class MultimodalDataCollator:

    processor: AutoProcessor

    def __call__(self, features):
        texts = [feature["text"] for feature in features]

        images = []
        for feature in features:
            if feature["images"]:
                images.extend(feature["images"])

        videos = []
        for feature in features:
            if feature["videos"]:
                videos.extend(feature["videos"])

        batch = self.processor(
            text=texts,
            images=images if images else None,
            videos=videos if videos else None,
            return_tensors="pt",
            padding=True,
        )

        labels = batch["input_ids"].clone()

        pad_token_id = self.processor.tokenizer.pad_token_id
        labels[labels == pad_token_id] = -100

        batch["labels"] = labels

        return batch


def print_parameter_statistics(model) -> None:

    total_params = 0
    trainable_params = 0
    lora_params = 0
    lora_a_params = 0
    lora_b_params = 0

    lora_tensor_count = 0
    trainable_tensor_count = 0

    non_lora_trainable = []

    for name, parameter in model.named_parameters():
        parameter_count = parameter.numel()
        lower_name = name.lower()

        total_params += parameter_count

        if parameter.requires_grad:
            trainable_params += parameter_count
            trainable_tensor_count += 1

        if "lora_" in lower_name:
            lora_params += parameter_count
            lora_tensor_count += 1

            if "lora_a" in lower_name:
                lora_a_params += parameter_count
            elif "lora_b" in lower_name:
                lora_b_params += parameter_count

        if parameter.requires_grad and "lora_" not in lower_name:
            non_lora_trainable.append(
                {
                    "name": name,
                    "count": parameter_count,
                    "shape": tuple(parameter.shape),
                }
            )

    non_lora_params = total_params - lora_params

    if total_params > 0:
        lora_percentage = lora_params / total_params * 100
        trainable_percentage = trainable_params / total_params * 100
    else:
        lora_percentage = 0.0
        trainable_percentage = 0.0

    print("\n" + "=" * 70)
    print("LoRA 训练参数统计")
    print("=" * 70)
    print(f"模型总参数量             : {total_params:,}")
    print(f"非 LoRA 参数量          : {non_lora_params:,}")
    print(f"LoRA 参数量             : {lora_params:,}")
    print(f"  ├─ LoRA A 参数量      : {lora_a_params:,}")
    print(f"  └─ LoRA B 参数量      : {lora_b_params:,}")
    print(f"实际可训练参数量         : {trainable_params:,}")
    print(f"LoRA 参数占模型比例      : {lora_percentage:.6f}%")
    print(f"可训练参数占模型比例      : {trainable_percentage:.6f}%")
    print(f"LoRA 参数张量数量        : {lora_tensor_count:,}")
    print(f"可训练参数张量数量        : {trainable_tensor_count:,}")
    print("=" * 70)

    print("\nPEFT 自带统计结果：")
    model.print_trainable_parameters()
    print()


def train(inspect_params_only: bool = False):

    LOCAL_MODEL_DIR = (
        "./models/LLaVA-OV15-8B-Instruct"
    )

    DATA_JSON = "./dataset.json"
    IMG_DIR = "./train/image"
    OUTPUT_DIR = "./output"


    print(
        f"🔥 正在从本地 {LOCAL_MODEL_DIR} "
        "使用 4-bit 量化加载模型..."
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_DIR,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )


    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(
        model,
        peft_config,
    )

    print_parameter_statistics(model)

    if inspect_params_only:
        print("=" * 70)
        print("参数统计完成。")
        print("已启用 --inspect-params-only，不会开始训练。")
        print("=" * 70)
        return


    processor = AutoProcessor.from_pretrained(
        LOCAL_MODEL_DIR,
        trust_remote_code=True,
    )

    dataset = LLaVA_SOD_Dataset(
        json_path=DATA_JSON,
        img_dir=IMG_DIR,
        processor=processor,
    )

    collator = MultimodalDataCollator(
        processor=processor,
    )


    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=10,
        learning_rate=2e-4,
        num_train_epochs=5,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        optim="adamw_torch",
        remove_unused_columns=False,
        report_to="none",
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )


    trainer.train()


    trainer.model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print(
        f"训练完成！LoRA 权重已保存至: {OUTPUT_DIR}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLaVA-OneVision 4-bit LoRA 微调"
    )

    parser.add_argument(
        "--inspect-params-only",
        action="store_true",
        help="只加载模型并输出 LoRA 参数量，不执行训练",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    train(
        inspect_params_only=args.inspect_params_only,
    )


if __name__ == "__main__":
    main()
