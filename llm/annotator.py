"""
LLM annotator for generating structured annotations with confidence scores.

This module provides the QwenAnnotator class for local Qwen-7B inference
with deterministic generation and structured JSON output parsing.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from .prompts import PromptTemplate, TaskType, get_prompt_template, get_json_schema
import config


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnnotationResult:
    """Container for annotation results with validation and error classification."""
    
    def __init__(self, text: str, label: str, confidence: str, rationale: str, 
                 raw_response: str = "", is_valid: bool = True, error: str = "", error_type: str = ""):
        """
        Initialize annotation result.
        
        Args:
            text: Original input text
            label: Predicted label
            confidence: Verbal confidence score
            rationale: Explanation for the prediction
            raw_response: Raw LLM response
            is_valid: Whether the annotation passed validation
            error: Error message if validation failed
            error_type: Type of error (parse_error_recoverable, schema_violation, etc.)
        """
        self.text = text
        self.label = label
        self.confidence = confidence
        self.rationale = rationale
        self.raw_response = raw_response
        self.is_valid = is_valid
        self.error = error
        self.error_type = error_type
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "text": self.text,
            "label": self.label,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "raw_response": self.raw_response,
            "is_valid": self.is_valid,
            "error": self.error
        }


class QwenAnnotator:
    """Annotator using Qwen3-14B model for structured annotation."""
    
    def __init__(self, model_name: str = None, task_type: TaskType = TaskType.SENTIMENT,
                 temperature: float = None, device: str = None, seed: int = 42):
        """
        Initialize Qwen annotator with config fallback.
        
        Args:
            model_name: Name of Qwen model (default: from config)
            task_type: Type of annotation task
            temperature: Generation temperature (default: from config, or 0.7 if do_sample=True)
            device: Device to use (cuda/cpu), auto-detected if None
            seed: Random seed for reproducibility
        """
        # Read from config with fallback
        self.model_name = model_name or config.LLM_MODELS.get(config.LLM_CONFIG.get("model_name", "qwen-14b"), "Qwen/Qwen3-14B")
        
        # Only get temperature if not provided and do_sample is True
        if temperature is not None:
            self.temperature = temperature
        elif config.LLM_CONFIG.get("do_sample", True):
            self.temperature = config.LLM_CONFIG.get("temperature", 0.7)
        else:
            self.temperature = None  # Not used in greedy decoding
        
        self.seed = seed
        self.task_type = task_type
        self.prompt_template = get_prompt_template(task_type, require_rationale=False)  # Phase 1: no rationale
        self.prompt_template_with_rationale = get_prompt_template(task_type, require_rationale=True)  # Phase 2
        self.json_schema = get_json_schema(task_type)
        
        # Auto-detect device
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Set seed for reproducibility
        self._set_seed(seed)
        
        self.model = None
        self.tokenizer = None
        logger.info(f"Initializing Qwen annotator: {self.model_name} on {device}")
        if self.temperature is not None:
            logger.info(f"Task: {task_type.value}, Temperature: {self.temperature}, Seed: {seed}")
        else:
            logger.info(f"Task: {task_type.value}, Greedy decoding, Seed: {seed}")
    
    def set_task_type(self, task_type: TaskType):
        """
        Change the task type without reloading the model.
        
        Args:
            task_type: New task type to use
        """
        self.task_type = task_type
        self.prompt_template = get_prompt_template(task_type, require_rationale=False)
        self.prompt_template_with_rationale = get_prompt_template(task_type, require_rationale=True)
        self.json_schema = get_json_schema(task_type)
        logger.info(f"Switched to task type: {task_type.value}")
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        import random
        import numpy as np
        import torch
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    @staticmethod
    def _normalize_label(s: str) -> str:
        """
        Normalize label string to handle variations.
        
        Args:
            s: Input label string
            
        Returns:
            Normalized label string
        """
        s = s.strip().lower().replace(" ", "_")
        # Handle common aliases
        alias = {
            "nei": "not_enough_info",
            "non_toxic": "non-toxic",
            "nontoxic": "non-toxic",
            "not_toxic": "non-toxic"
        }
        return alias.get(s, s)
    
    @staticmethod
    def _truncate_text(text: str, tokenizer, max_tokens: int = 512, head_ratio: float = 0.5) -> str:
        """
        Truncate text using head+tail strategy to preserve context.
        
        Args:
            text: Input text
            tokenizer: Tokenizer instance
            max_tokens: Maximum tokens to keep
            head_ratio: Ratio of tokens to keep from head (rest from tail)
            
        Returns:
            Truncated text
        """
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return text
        
        head_tokens = int(max_tokens * head_ratio)
        tail_tokens = max_tokens - head_tokens
        
        truncated_tokens = tokens[:head_tokens] + tokens[-tail_tokens:]
        return tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    
    def load_model(self):
        """
        Load the Qwen model and tokenizer (supports both 7B and 14B models).
        
        Loads model with appropriate settings for annotation tasks.
        Supports both remote HuggingFace models and local paths.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # Enable TF32 for faster matmul on Ampere+ GPUs (PyTorch 2.9+ API)
            if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
                torch.backends.cuda.matmul.fp32_precision = 'tf32'
            if hasattr(torch.backends.cudnn, 'conv') and hasattr(torch.backends.cudnn.conv, 'fp32_precision'):
                torch.backends.cudnn.conv.fp32_precision = 'tf32'
            
            logger.info(f"Loading Qwen model: {self.model_name}...")
            
            # Detect if it's a local path or remote model
            from pathlib import Path
            is_local = Path(self.model_name).exists()
            
            if is_local:
                logger.info(f"Loading from local path: {self.model_name}")
            else:
                logger.info("Loading from HuggingFace Hub (first-time download may take time)")
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # CRITICAL: Set left padding for decoder-only models (Qwen)
            # This prevents the warning: "right-padding was detected"
            self.tokenizer.padding_side = 'left'
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Tokenizer loaded successfully (padding_side={self.tokenizer.padding_side})")
            
            # Check GPU availability before loading model
            if self.device == "cuda":
                if not torch.cuda.is_available():
                    logger.error("CUDA is not available! Model will fall back to CPU (very slow).")
                    logger.error("This will consume large amounts of system RAM and may be extremely slow.")
                    self.device = "cpu"
                else:
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    logger.info(f"GPU detected: {gpu_name}")
                    logger.info(f"GPU memory available: {gpu_memory:.2f} GB")
                    logger.info("Model will be loaded on GPU for efficient inference")
            else:
                logger.warning("Running on CPU - this will be slow and memory-intensive!")
                logger.warning("For better performance, use a CUDA-enabled GPU")
            
            # Load model config first to disable sliding window
            from transformers import AutoConfig
            model_config = AutoConfig.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Disable sliding window to prevent warnings
            if hasattr(model_config, 'sliding_window'):
                model_config.sliding_window = None
            if hasattr(model_config, 'use_sliding_window'):
                model_config.use_sliding_window = False
            
            # Load model with modified config
            logger.info("Loading model weights...")
            try:
                # Try flash attention 2 first
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    config=model_config,
                    torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    attn_implementation="flash_attention_2"
                )
                logger.info("Using Flash Attention 2 for optimal speed")
            except (ImportError, ValueError):
                # Fallback to sdpa (PyTorch native optimized attention)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    config=model_config,
                    torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    attn_implementation="sdpa"
                )
                logger.info("Using SDPA (PyTorch native optimized attention)")
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Override model's default generation config to avoid warnings
            if config.LLM_CONFIG.get("do_sample", True) is False:
                # For greedy decoding, disable sampling in model's generation config
                self.model.generation_config.do_sample = False
                self.model.generation_config.temperature = None
                self.model.generation_config.top_p = None
                self.model.generation_config.top_k = None
                logger.info("Model generation config updated: greedy decoding mode")
            
            self.model.eval()
            
            # Report final device placement
            if self.device == "cuda":
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                logger.info(f"Model loaded successfully on GPU")
                logger.info(f"GPU memory allocated: {allocated:.2f} GB")
                logger.info(f"GPU memory reserved: {reserved:.2f} GB")
            else:
                logger.info(f"Model loaded on CPU (not recommended for large models)")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error(f"Model path: {self.model_name}")
            logger.error("Please check:")
            logger.error("1. Model files exist in the specified path")
            logger.error("2. transformers library is up to date: pip install -U transformers")
            logger.error("3. torch is properly installed")
            raise
    
    def annotate_single(self, text: str) -> AnnotationResult:
        """
        Annotate a single text input using Qwen3 with proper slicing.
        
        Args:
            text: Input text to annotate
            
        Returns:
            AnnotationResult with prediction and metadata
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            import torch
            
            # Format messages using chat template
            messages = self.prompt_template.format_messages(text)
            
            # Apply chat template
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self.tokenizer([formatted], return_tensors="pt").to(self.device)
            
            # Set seed for reproducibility
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            
            # Build generation kwargs
            do_sample = config.LLM_CONFIG.get("do_sample", True)
            gen_kwargs = {
                "max_new_tokens": config.LLM_CONFIG.get("max_tokens", 128),
                "do_sample": do_sample,
                "repetition_penalty": config.LLM_CONFIG.get("repetition_penalty", 1.05),
                "pad_token_id": pad_id,
                "eos_token_id": self.tokenizer.eos_token_id
            }
            
            # Only add sampling params if do_sample is True
            if do_sample and self.temperature is not None:
                gen_kwargs["temperature"] = self.temperature
                gen_kwargs["top_p"] = config.LLM_CONFIG.get("top_p", 0.8)
                gen_kwargs["top_k"] = config.LLM_CONFIG.get("top_k", 20)
            
            # Generate with AMP and inference mode
            with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # Decode only the generated part (slice by actual input length)
            input_len = int(inputs["attention_mask"][0].sum().item())
            resp_ids = outputs[0, input_len:]
            response = self.tokenizer.decode(resp_ids, skip_special_tokens=True).strip()
            
            # Explicit memory cleanup
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Clean response: extract only assistant part if full dialogue is present
            if 'assistant' in response:
                # Split by 'assistant' and take the last part
                response = response.split('assistant')[-1].strip()
            
            # Validate and parse response
            validation_result = self.validate_response(response)
            
            if validation_result["valid"]:
                data = validation_result["data"]
                return AnnotationResult(
                    text=text,
                    label=data["label"],
                    confidence=data["confidence"],
                    rationale=data["rationale"],
                    raw_response=response,
                    is_valid=True,
                    error="",
                    error_type=validation_result.get("error_type", "")
                )
            else:
                return AnnotationResult(
                    text=text,
                    label="",
                    confidence="low",
                    rationale="",
                    raw_response=response,
                    is_valid=False,
                    error=validation_result["error"],
                    error_type=validation_result.get("error_type", "unknown_error")
                )
                
        except Exception as e:
            logger.error(f"Error during annotation: {e}")
            return AnnotationResult(
                text=text,
                label="",
                confidence="low",
                rationale="",
                raw_response="",
                is_valid=False,
                error=str(e),
                error_type="exception"
            )
    
    def annotate_batch(self, texts: List[str], batch_size: int = 8) -> List[AnnotationResult]:
        """
        Annotate a batch of texts with optimized processing.
        
        Features:
        - Input truncation (head+tail 512 tokens)
        - Length-based bucketing for efficient batching
        - Two-stage generation (fast label+confidence first)
        
        Args:
            texts: List of input texts to annotate
            batch_size: Number of texts to process in parallel
            
        Returns:
            List of AnnotationResult objects
        """
        import torch
        
        # Truncate texts to 512 tokens (head 256 + tail 256)
        truncated_texts = [self._truncate_text(t, self.tokenizer, max_tokens=512) for t in texts]
        
        # Sort by length for better batching (reduce padding overhead)
        indexed_texts = list(enumerate(truncated_texts))
        indexed_texts.sort(key=lambda x: len(self.tokenizer.encode(x[1], add_special_tokens=False)))
        
        results = [None] * len(texts)  # Preserve original order
        
        for i in tqdm(range(0, len(indexed_texts), batch_size), desc="Annotating batches"):
            batch_items = indexed_texts[i:i+batch_size]
            indices = [idx for idx, _ in batch_items]
            batch = [text for _, text in batch_items]
            
            # Format all messages in batch
            msgs = [self.prompt_template.format_messages(t) for t in batch]
            formatted = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
            
            # Tokenize batch with padding
            inputs = self.tokenizer(formatted, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(self.device)
            
            # Set seed for reproducibility
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            
            # Build generation kwargs
            do_sample = config.LLM_CONFIG.get("do_sample", True)
            gen_kwargs = {
                "max_new_tokens": config.LLM_CONFIG.get("max_tokens", 128),
                "do_sample": do_sample,
                "repetition_penalty": config.LLM_CONFIG.get("repetition_penalty", 1.05),
                "pad_token_id": pad_id,
                "eos_token_id": self.tokenizer.eos_token_id
            }
            
            # Only add sampling params if do_sample is True
            if do_sample and self.temperature is not None:
                gen_kwargs["temperature"] = self.temperature
                gen_kwargs["top_p"] = config.LLM_CONFIG.get("top_p", 0.8)
                gen_kwargs["top_k"] = config.LLM_CONFIG.get("top_k", 20)
            
            # Generate with AMP and inference mode for speed
            with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # Decode each sample by its actual input length
            input_lens = inputs["attention_mask"].sum(dim=1).tolist()
            # Move outputs to CPU immediately to free GPU memory
            outputs_cpu = outputs.cpu()
            
            for j, L in enumerate(input_lens):
                resp_ids = outputs_cpu[j, int(L):]
                resp = self.tokenizer.decode(resp_ids, skip_special_tokens=True).strip()
                
                # Clean response: extract only assistant part if full dialogue is present
                if 'assistant' in resp:
                    resp = resp.split('assistant')[-1].strip()
                
                # Stop at first newline or closing brace (avoid trailing tokens)
                if '\n' in resp:
                    resp = resp.split('\n')[0].strip()
                if '}\n' in resp:
                    resp = resp.split('}\n')[0] + '}'
                
                # Validate response
                vr = self.validate_response(resp)
                if vr["valid"]:
                    d = vr["data"]
                    orig_idx = indices[j]
                    results[orig_idx] = AnnotationResult(
                        texts[orig_idx], d["label"], d["confidence"], d.get("rationale", ""), 
                        resp, True, "", vr.get("error_type", ""))
                else:
                    orig_idx = indices[j]
                    results[orig_idx] = AnnotationResult(
                        texts[orig_idx], "", "low", "", resp, False, vr["error"], 
                        vr.get("error_type", "unknown_error"))
            
            # Explicit memory cleanup after each batch
            del inputs, outputs, outputs_cpu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results
    
    def annotate_dataframe(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """
        Annotate a DataFrame column.
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing text to annotate
            
        Returns:
            DataFrame with annotation results added as new columns
            
        DataFrame annotation is not implemented in this version.
        Use annotate_batch() for list of texts instead.
        """
        # DataFrame annotation not implemented
        # 1. Extract text column
        # 2. Run batch annotation
        # 3. Add results as new columns
        # 4. Preserve original DataFrame structure
        
        logger.info(f"DataFrame annotation with {len(df)} rows not supported")
        
        # Placeholder implementation
        df_result = df.copy()
        df_result["llm_label"] = "not_implemented"
        df_result["llm_confidence"] = "not_implemented"
        df_result["llm_rationale"] = "not_implemented"
        df_result["annotation_valid"] = False
        
        return df_result
    
    def _extract_json_from_response(self, response: str) -> tuple[str, str]:
        """
        Extract JSON from response with robust fallback strategies.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Tuple of (json_str, error_type)
        """
        response_clean = response.strip()
        error_type = ''
        
        # Priority 1: Extract from <json></json> tags (last occurrence to handle repeats)
        if '<json>' in response_clean and '</json>' in response_clean:
            start = response_clean.rfind('<json>') + 6  # Use rfind to get last occurrence
            end = response_clean.rfind('</json>')
            if start < end:
                json_str = response_clean[start:end].strip()
                if json_str:  # Ensure non-empty
                    return json_str, error_type
        
        # Priority 2: Remove markdown code blocks
        if '```json' in response_clean or response_clean.startswith('```'):
            # Extract content between ``` markers
            lines = response_clean.split('\n')
            in_code = False
            json_lines = []
            for line in lines:
                if line.strip().startswith('```'):
                    if in_code:
                        break
                    in_code = True
                    continue
                if in_code and line.strip():
                    json_lines.append(line)
            if json_lines:
                response_clean = '\n'.join(json_lines)
                error_type = 'parse_error_recoverable'
        
        # Priority 3: Find last complete JSON object (most reliable for contaminated output)
        json_start = response_clean.rfind('{')
        json_end = response_clean.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response_clean[json_start:json_end]
            if json_str and json_str != '{}':  # Ensure non-empty and not just braces
                if json_start > 0 or json_end < len(response_clean):
                    error_type = 'parse_error_recoverable'
                return json_str, error_type
        
        # Fallback: try first { to last }
        json_start = response_clean.find('{')
        json_end = response_clean.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response_clean[json_start:json_end]
            if json_str and json_str != '{}':
                return json_str, 'parse_error_recoverable'
        
        # Last resort: return as-is
        return response_clean, 'parse_error'
    
    def validate_response(self, response: str) -> Dict[str, Any]:
        """
        Validate LLM response with recoverable error handling.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Dict with 'valid', 'data', 'error', 'error_type' keys
        """
        try:
            # Extract JSON from response
            json_str, error_type = self._extract_json_from_response(response)
            
            # Parse JSON
            parsed = json.loads(json_str)
            
            # Validate required fields (rationale is optional)
            required_fields = ["label", "confidence"]
            for field in required_fields:
                if field not in parsed:
                    return {"valid": False, "error": f"Missing required field: {field}", 
                           "error_type": "schema_violation"}
            
            # Normalize label
            parsed["label"] = self._normalize_label(parsed["label"])
            
            # Validate confidence values
            valid_confidence = ["low", "medium", "high"]
            if parsed["confidence"] not in valid_confidence:
                return {"valid": False, "error": f"Invalid confidence: {parsed['confidence']}", 
                       "error_type": "schema_violation"}
            
            # Validate rationale length (optional field)
            if "rationale" in parsed and parsed["rationale"]:
                if len(parsed["rationale"]) < 10:
                    return {"valid": False, "error": "Rationale too short (min 10 chars)", 
                           "error_type": "content_violation"}
            else:
                # Rationale not provided (valid for phase 1)
                parsed["rationale"] = ""
            
            # Validate label against task-specific schema
            if self.json_schema and "properties" in self.json_schema:
                label_enum = self.json_schema["properties"]["label"].get("enum")
                if label_enum and parsed["label"] not in label_enum:
                    return {"valid": False, "error": f"Invalid label '{parsed['label']}'", 
                           "error_type": "schema_violation"}
            
            return {"valid": True, "data": parsed, "error_type": error_type}
            
        except json.JSONDecodeError as e:
            # If we already tried extraction, this is recoverable
            error_msg = f"JSON parse error: {str(e)}"
            return {"valid": False, "error": error_msg, 
                   "error_type": error_type if error_type else "parse_error"}
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}", 
                   "error_type": "unknown_error"}
    
    def get_generation_config(self) -> Dict[str, Any]:
        """
        Get generation configuration following Qwen3 best practices.
        
        Returns:
            Dictionary of generation parameters
            
        Note: For non-thinking mode classification, Qwen3 recommends:
              - Temperature=0.7 (we use 0.3-0.5 for more deterministic results)
              - TopP=0.8, TopK=20
              - do_sample=True (DO NOT use greedy decoding)
        """
        return {
            "temperature": self.temperature,  # 0.3-0.5 for deterministic but not greedy
            "top_p": 0.8,
            "top_k": 20,
            "do_sample": True,  # Must be True, greedy decoding causes issues
            "max_new_tokens": 512,
            "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer else None,
            "eos_token_id": self.tokenizer.eos_token_id if self.tokenizer else None
        }


def create_annotator(task_type: str, model_name: str = "Qwen/Qwen3-14B", 
                     temperature: float = 0.3, seed: int = 42, load_model: bool = True) -> QwenAnnotator:
    """
    Factory function to create annotator for specific task.
    
    Args:
        task_type: Type of annotation task (sentiment, toxicity, fact_verification)
        model_name: Name of model to use (default: Qwen/Qwen3-14B)
        temperature: Generation temperature (0.3-0.5 recommended)
        seed: Random seed for reproducibility (default: 42)
        load_model: Whether to load the model immediately
        
    Returns:
        Configured QwenAnnotator instance
    """
    task_enum = TaskType(task_type)
    annotator = QwenAnnotator(model_name, task_enum, temperature=temperature, seed=seed)
    
    if load_model:
        annotator.load_model()
    
    return annotator


if __name__ == "__main__":
    # Test the annotator
    print("Testing QwenAnnotator...")
    
    # Create test annotator
    annotator = create_annotator("sentiment")
    
    # Test single annotation
    test_text = "This movie was absolutely fantastic!"
    result = annotator.annotate_single(test_text)
    print(f"Test annotation result: {result.to_dict()}")
    
    print("Annotation logic implemented in QwenAnnotator class")
