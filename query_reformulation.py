# ============================================================
# Query Reformulation for PUBHEALTH Claims
# Local LLM (Mistral 7B Instruct)

import json
import os
import warnings
import logging
import hashlib

from huggingface_hub import InferenceClient
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from query_reformulation_config import MODEL_NAME, CACHE_PATH, BATCH_SIZE, MAX_NEW_TOK, TEMPERATURE,  SPLITS, PROMPT_TEMPLATE

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Query_Reformulater:
    def __init__(self, model_name=MODEL_NAME, model_load_mode="api",
                 HF_token = None, batch_size = BATCH_SIZE, temperature=TEMPERATURE, 
                 max_new_tokens = MAX_NEW_TOK, prompt_template=PROMPT_TEMPLATE,
                 cache_from=None, cache_to=None):
        self.model_name = model_name
        self.model_load_mode = model_load_mode
        self.HF_token = HF_token
        self.batch_size = batch_size
        self.cache_from = cache_from
        self.cache_to = cache_to
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.prompt_template = prompt_template

        self.cache = {}
        self.load_cache()   

        if self.model_load_mode == "api":
            self.tokenizer, self.model = None, self.load_model()
        else:
            self.tokenizer, self.model = self.load_model_local()
   
    def load_model_local(self):
        """
        Load Mistral 7B in 4-bit NF4 quantization.
        Fits on a single Kaggle T4 (16GB) using ~4-5GB VRAM.
        """
        logger.info(f"Loading tokenizer: {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",  # required for batch generation
        )
        tokenizer.pad_token = tokenizer.eos_token

        logger.info("Loading model in 4-bit (NF4)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model.eval()

        logger.info("Model loaded successfully.")
        return tokenizer, model

    def load_model(self):
        client = InferenceClient(
            model=self.model_name,
            token=self.HF_token,
            provider="auto",
            timeout=120,
        )
        return client
    
    @staticmethod
    def claim_hash(claim: str) -> str:
        return hashlib.sha256(claim.strip().lower().encode()).hexdigest()

    def _inference_local(self, prompts):
        tokenizer = self.tokenizer
        model = self.model

        outputs = []

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=768,
        ).to(next(model.parameters()).device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Remove the original prompt tokens from the output
        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_length:]

        batch_outputs = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        outputs.extend([text.strip() for text in batch_outputs])

        return outputs
    
    def _inference_api(self, prompts):
        client = self.model
        outputs = []

        for prompt in prompts:
            response = client.chat_completion(
                messages=[
                {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_new_tokens,
                temperature=self.temperature
            )

            text = response.choices[0].message.content
            outputs.append(text.strip())

        return outputs

    def inference(self, prompts):
        """
        Takes a list of prompt strings and returns a list of generated response strings.
        Works for both:
        - API mode: self.tokenizer is None, self.model is Hugging Face InferenceClient
        - Local mode: self.tokenizer is tokenizer, self.model is AutoModelForCausalLM
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = list(prompts)

        if len(prompts) == 0:
            return []

        if self.model_load_mode == "api":
            return self._inference_api(prompts)

        return self._inference_local(prompts)

    def reformulate_batch(self, claims: list) -> list:
        """
        Reformulate a list of claims in batches.
        Skips claims already in cache.
        Saves cache to disk after every batch.
        """
        results  = [None] * len(claims)
        uncached = []  # (original_index, claim)
        if self.cache is not None:
            for i, claim in enumerate(claims):
                key = Query_Reformulater.claim_hash(claim)
                if key in self.cache:
                    results[i] = self.cache[key]
                else:
                    uncached.append((i, claim))
        else:
            uncached = [(i, claims[i]) for i in range(len(claims))]
            self.cache = {}

        logger.info(
            f"Total: {len(claims)} | "
            f"Cached: {len(claims) - len(uncached)} | "
            f"To process: {len(uncached)}"
        )

        for batch_start in range(0, len(uncached), self.batch_size):
            batch = uncached[batch_start : batch_start + self.batch_size]
            indices, batch_claims = zip(*batch)

            prompts = [self.prompt_template.format(claim=c) for c in batch_claims]

            outputs = self.inference(prompts)

            for j, (orig_idx, claim) in enumerate(zip(indices, batch_claims)):
                raw = outputs[j]
                result = raw
                key = Query_Reformulater.claim_hash(claim)
                self.cache[key] = result
                results[orig_idx] = result
            processed = min(batch_start + self.batch_size, len(uncached))
            logger.info(
                f"Processed {processed}/{len(uncached)} uncached claims"
            )
            self.save_cache()
        return results

    def save_cache(self):
        if self.cache_to is None:
            return

        folder = os.path.dirname(self.cache_to)
        if folder:
            os.makedirs(folder, exist_ok=True)

        with open(self.cache_to, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
    def load_cache(self):
        if self.cache_from is None:
            return

        if not os.path.exists(self.cache_from):
            return

        with open(self.cache_from, "r", encoding="utf-8") as f:
            self.cache = json.load(f)