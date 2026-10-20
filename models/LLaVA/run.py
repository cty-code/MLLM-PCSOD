
import os
import re
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel
from qwen_vl_utils import process_vision_info

try:
    from transformers import BitsAndBytesConfig 
except Exception:
    BitsAndBytesConfig = None

PROMPT: str ="The image displays one or more salient objects against a pure black background. Please identify the broad categories of the distinct main entities you see. You MUST strictly format your output starting with 'The salient objects are ' followed by a comma-separated list of very short noun phrases (maximum 3 to 4 words per phrase). Crucially, you must name the ENTIRE object as a whole (e.g., 'red doll', 'person') rather than breaking it down into individual parts or clothing. Do NOT list clothing items as separate objects (e.g., never output 'red hat, red dress', instead output 'red doll').Focus ONLY on the overall main color and the basic category name in English. Do NOT describe any small details, textures, text, logos, or minor items placed on the objects. For a composite object consisting of multiple associated parts, use 'and' to connect its components within a single phrase. Do not include any other explanations or periods. (For example: red tricycle, person in orange shirt, blue pole and orange grip)"


class LLaVAOneVisionLoRA:

    def __init__(
        self,
        model_dir: str,
        lora_dir: str,
        fourbit: bool = False,
        dtype: Optional[torch.dtype] = None,
        device_map: str = "auto",
        trust_remote_code: bool = True,
    ):
        self.model_dir = str(model_dir)
        self.lora_dir = str(lora_dir)
        
        if not Path(self.model_dir).is_dir():
            raise FileNotFoundError(f"基础模型目录不存在：{self.model_dir}")
        if not Path(self.lora_dir).is_dir():
            raise FileNotFoundError(f"LoRA 权重目录不存在：{self.lora_dir}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        load_kwargs = dict(trust_remote_code=trust_remote_code, device_map=device_map)
        if fourbit:
            if BitsAndBytesConfig is None:
                raise RuntimeError("启用 fourbit=True 需要安装 bitsandbytes：pip install bitsandbytes")
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            load_kwargs["dtype"] = (dtype if dtype is not None else torch.float16)

        base_model = AutoModelForCausalLM.from_pretrained(self.model_dir, **load_kwargs)
        self.processor = AutoProcessor.from_pretrained(self.model_dir, trust_remote_code=trust_remote_code)


        self.model = PeftModel.from_pretrained(base_model, self.lora_dir)
        self.model.eval()

    @staticmethod
    def _build_messages(image_path: str, prompt: str):
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text",  "text": prompt},
            ]
        }]

    def _prepare_inputs(self, messages) -> Dict:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        return inputs

    def generate(self, messages, max_new_tokens: int = 50, **gen_kwargs) -> str:
        inputs = self._prepare_inputs(messages)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                do_sample=False, 
                **gen_kwargs
            )
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, output_ids)]
        text = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        return text.strip()

    def describe(self, image_path: str, prompt: Optional[str] = None, max_new_tokens: int = 50, **gen_kwargs) -> str:
        if not Path(image_path).is_file():
            raise FileNotFoundError(f"图片不存在：{image_path}")
        _prompt = PROMPT if (prompt is None) else prompt
        msgs = self._build_messages(image_path, _prompt)
        return self.generate(msgs, max_new_tokens=max_new_tokens, **gen_kwargs)



def _this_dir() -> Path:
    return Path(os.path.abspath(os.path.dirname(__file__)))

def _natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.stem)]

def list_images(img_dir: Path, patterns: Tuple[str, ...] = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")) -> List[Path]:
    img_dir = Path(img_dir)
    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(img_dir.glob(pat)))
    files = sorted(set(files), key=_natural_key)
    return files

def ensure_out_dir(name: str) -> Path:
    out = _this_dir() / name
    out.mkdir(parents=True, exist_ok=True)
    return out

def run_batch(
    model_dir: str,
    lora_dir: str,
    img_dir: str,
    start_index: int = 0,
    end_index_inclusive: Optional[int] = None,
    out_dir_name: str = "lora_test_results",
    fourbit: bool = False,
    max_new_tokens: int = 50,
    **gen_kwargs,
) -> Dict[str, int]:
    
    ov = LLaVAOneVisionLoRA(model_dir=model_dir, lora_dir=lora_dir, fourbit=fourbit)
    images = list_images(Path(img_dir))
    if not images:
        raise RuntimeError(f"未在目录中发现图片：{img_dir}")

    n = len(images)
    if end_index_inclusive is None:
        end_index_inclusive = n - 1

    if start_index < 0 or end_index_inclusive >= n or start_index > end_index_inclusive:
        raise IndexError(f"索引越界：共有 {n} 张，start={start_index}, end={end_index_inclusive}")

    out_dir = ensure_out_dir(out_dir_name)
    err_log = out_dir / "errors.log"

    ok = fail = 0
    for i, img_path in enumerate(images[start_index : end_index_inclusive + 1], start=start_index):
        try:
            text = ov.describe(str(img_path), prompt=None, max_new_tokens=max_new_tokens, **gen_kwargs)
            out_file = out_dir / f"{img_path.stem}.txt"
            out_file.write_text(text + "\n", encoding="utf-8")
            print(f"[{i}/{n-1}] OK -> {out_file.name} | Result: {text}")
            ok += 1
        except Exception as e:
            fail += 1
            with err_log.open("a", encoding="utf-8") as f:
                f.write(f"{img_path.name}\t{type(e).__name__}: {e}\n")
            print(f"[{i}/{n-1}] FAIL -> {img_path.name}: {e}")

    return {"total": end_index_inclusive - start_index + 1, "ok": ok, "fail": fail}


def parse_args():
    ap = argparse.ArgumentParser(description="LLaVA-OneVision LoRA 微调后本地批量测试")
    ap.add_argument("--model-dir", required=True, help="本地基础模型目录")
    ap.add_argument("--lora-dir", required=True, help="训练好的 LoRA 权重目录")
    ap.add_argument("--img-dir", required=True, help="输入测试图片文件夹")
    ap.add_argument("--start", type=int, default=0, help="开始索引（从0开始，含）")
    ap.add_argument("--end", type=int, default=None, help="结束索引（含），默认处理到最后一张")
    ap.add_argument("--out-dir-name", type=str, default="lora_test_results", help="输出文件夹名（与代码同级）")
    ap.add_argument("--fourbit", action="store_true", help="启用 4bit 量化以节省显存")
    ap.add_argument("--max-new-tokens", type=int, default=50, help="最大生成的 Token 数量（建议保持较小以防啰嗦）")
    return ap.parse_args()

def main():
    args = parse_args()
    stats = run_batch(
        model_dir=args.model_dir,
        lora_dir=args.lora_dir,
        img_dir=args.img_dir,
        start_index=args.start,
        end_index_inclusive=args.end,
        out_dir_name=args.out_dir_name,
        fourbit=args.fourbit,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"批量测试完成！统计信息: {stats}")

if __name__ == "__main__":
    main()
