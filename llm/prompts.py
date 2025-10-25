"""
Prompt templates for LLM annotation tasks.

This module contains structured prompt templates for different annotation tasks
with few-shot examples and JSON output format instructions.
"""

from typing import Dict, List, Any
from enum import Enum


class TaskType(Enum):
    """Enumeration of supported annotation tasks."""
    SENTIMENT = "sentiment"
    TOXICITY = "toxicity"
    CRISIS = "crisis_classification"
    FACT_VERIFICATION = "fact_verification"


class PromptTemplate:
    """Base class for prompt templates (zero-shot with strict constraints)."""
    
    def __init__(self, task_type: TaskType, system_prompt: str):
        """
        Initialize prompt template.
        
        Args:
            task_type: Type of annotation task
            system_prompt: System instruction for the LLM (zero-shot)
        """
        self.task_type = task_type
        self.system_prompt = system_prompt
    
    def format_prompt(self, text: str) -> str:
        """
        Format the complete prompt for a given input text.
        
        Args:
            text: Input text to annotate
            
        Returns:
            Formatted prompt string
        """
        # Build prompt with system instruction and input
        prompt_parts = [self.system_prompt]
        prompt_parts.append("\nNow classify this text:")
        prompt_parts.append(f'Text: "{text}"')
        prompt_parts.append("\nOutput (JSON only, no other text):")
        return "\n".join(prompt_parts)
    
    def format_messages(self, text: str) -> List[Dict[str, str]]:
        """
        Format messages for chat template (recommended for Qwen).
        
        Args:
            text: Input text to annotate
            
        Returns:
            List of message dicts with role and content
        """
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Text: {text}"}
        ]


def get_prompt_template(task_type: TaskType, require_rationale: bool = True) -> PromptTemplate:
    """
    Get prompt template for a specific task type (zero-shot with strict constraints).
    
    Args:
        task_type: Type of annotation task
        require_rationale: If True, require rationale; if False, only label+confidence
        
    Returns:
        PromptTemplate instance for the task
    """
    # Common output requirements (reduce repetition)
    if require_rationale:
        _OUTPUT_RULES = """
Output format:
<json>{"label": "...", "confidence": "low/medium/high", "rationale": "10-200 chars from text"}</json>

Requirements:
- "confidence" must be: low, medium, or high
- "rationale" must cite evidence from input text (10-200 characters)
- If uncertain, use "confidence": "low"
- Output ONLY the JSON, nothing else"""
    else:
        _OUTPUT_RULES = """
Output format:
<json>{"label": "...", "confidence": "low/medium/high"}</json>

Requirements:
- "confidence" must be: low, medium, or high
- If uncertain, use "confidence": "low"
- Output ONLY the JSON, nothing else"""
    
    if task_type == TaskType.SENTIMENT:
        system_prompt = f"""Task: Classify the sentiment of text as positive or negative.

Label must be one of: ["positive", "negative"]{_OUTPUT_RULES}"""
    
    elif task_type == TaskType.TOXICITY:
        system_prompt = f"""Task: Classify whether text is toxic or non-toxic.

Label must be one of: ["toxic", "non-toxic"]{_OUTPUT_RULES}"""
    
    elif task_type == TaskType.CRISIS:
        system_prompt = f"""Task: Classify whether text reports a crisis or emergency situation.

Label must be one of: ["crisis", "not_crisis"]{_OUTPUT_RULES}"""
    
    elif task_type == TaskType.FACT_VERIFICATION:
        system_prompt = f"""Task: Verify whether a claim is supported, refuted, or has not enough information.

Label must be one of: ["supports", "refutes", "not_enough_info"]{_OUTPUT_RULES}"""
    
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    return PromptTemplate(task_type, system_prompt)


def get_json_schema(task_type: TaskType) -> Dict[str, Any]:
    """
    Get JSON schema for LLM output validation.
    
    Args:
        task_type: Type of annotation task
        
    Returns:
        JSON schema for validating LLM responses
    """
    base_schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "rationale": {"type": "string", "minLength": 10}
        },
        "required": ["label", "confidence", "rationale"]
    }
    
    # Add task-specific label constraints
    if task_type == TaskType.SENTIMENT:
        base_schema["properties"]["label"]["enum"] = ["positive", "negative"]
    elif task_type == TaskType.TOXICITY:
        base_schema["properties"]["label"]["enum"] = ["toxic", "non-toxic"]
    elif task_type == TaskType.CRISIS:
        base_schema["properties"]["label"]["enum"] = ["crisis", "not_crisis"]
    elif task_type == TaskType.FACT_VERIFICATION:
        base_schema["properties"]["label"]["enum"] = ["supports", "refutes", "not_enough_info"]
    
    return base_schema


def get_verbal_confidence_mapping() -> Dict[str, float]:
    """
    Get mapping from verbal confidence to numeric scores.
    
    Returns:
        Dictionary mapping verbal confidence to numeric values
    """
    return {
        "low": 0.3,
        "medium": 0.6, 
        "high": 0.9
    }
