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
    """Base class for prompt templates."""
    
    def __init__(self, task_type: TaskType, system_prompt: str, examples: List[Dict[str, Any]]):
        """
        Initialize prompt template.
        
        Args:
            task_type: Type of annotation task
            system_prompt: System instruction for the LLM
            examples: Few-shot examples for the task
        """
        self.task_type = task_type
        self.system_prompt = system_prompt
        self.examples = examples
    
    def format_prompt(self, text: str) -> str:
        """
        Format the complete prompt for a given input text.
        
        Args:
            text: Input text to annotate
            
        Returns:
            Formatted prompt string
            
        TODO: Implement prompt formatting with examples and JSON instructions
        """
        # TODO: Implement prompt formatting
        # 1. Add system prompt
        # 2. Add few-shot examples
        # 3. Add JSON output format instructions
        # 4. Add the input text to annotate
        # 5. Ensure deterministic formatting
        return f"TODO: Format prompt for {self.task_type.value} task with text: {text[:50]}..."


def get_prompt_template(task_type: TaskType) -> PromptTemplate:
    """
    Get prompt template for a specific task type.
    
    Args:
        task_type: Type of annotation task
        
    Returns:
        PromptTemplate instance for the task
        
    TODO: Implement task-specific prompt templates
    """
    # TODO: Implement task-specific templates
    # 1. Sentiment: positive/negative classification with confidence
    # 2. Toxicity: toxic/non-toxic classification with confidence  
    # 3. Crisis: crisis/not_crisis classification with confidence
    # 4. Fact verification: supports/refutes classification with confidence
    
    if task_type == TaskType.SENTIMENT:
        # TODO: Create sentiment-specific prompt
        system_prompt = "TODO: Sentiment analysis system prompt"
        examples = [
            # TODO: Add few-shot examples for sentiment
            {"text": "This movie was amazing!", "label": "positive", "confidence": "high", "rationale": "Clear positive sentiment"}
        ]
    elif task_type == TaskType.TOXICITY:
        # TODO: Create toxicity-specific prompt
        system_prompt = "TODO: Toxicity detection system prompt"
        examples = [
            # TODO: Add few-shot examples for toxicity
            {"text": "You are an idiot!", "label": "toxic", "confidence": "high", "rationale": "Contains personal insult"}
        ]
    elif task_type == TaskType.CRISIS:
        # TODO: Create crisis-specific prompt
        system_prompt = "TODO: Crisis detection system prompt"
        examples = [
            # TODO: Add few-shot examples for crisis
            {"text": "Earthquake hit our city!", "label": "crisis", "confidence": "high", "rationale": "Reports natural disaster"}
        ]
    elif task_type == TaskType.FACT_VERIFICATION:
        # TODO: Create fact verification prompt
        system_prompt = "TODO: Fact verification system prompt"
        examples = [
            # TODO: Add few-shot examples for fact verification
            {"text": "The sky is blue", "label": "supports", "confidence": "high", "rationale": "Commonly known fact"}
        ]
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    return PromptTemplate(task_type, system_prompt, examples)


def get_json_schema(task_type: TaskType) -> Dict[str, Any]:
    """
    Get JSON schema for LLM output validation.
    
    Args:
        task_type: Type of annotation task
        
    Returns:
        JSON schema for validating LLM responses
        
    TODO: Implement JSON schema validation
    """
    # TODO: Implement JSON schema for each task type
    # Required fields: label, confidence, rationale
    # Optional fields: additional_metadata
    
    base_schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "rationale": {"type": "string", "minLength": 10}
        },
        "required": ["label", "confidence", "rationale"]
    }
    
    # TODO: Add task-specific label constraints
    if task_type == TaskType.SENTIMENT:
        base_schema["properties"]["label"]["enum"] = ["positive", "negative"]
    elif task_type == TaskType.TOXICITY:
        base_schema["properties"]["label"]["enum"] = ["toxic", "non-toxic"]
    elif task_type == TaskType.CRISIS:
        base_schema["properties"]["label"]["enum"] = ["crisis", "not_crisis"]
    elif task_type == TaskType.FACT_VERIFICATION:
        base_schema["properties"]["label"]["enum"] = ["supports", "refutes"]
    
    return base_schema


def get_verbal_confidence_mapping() -> Dict[str, float]:
    """
    Get mapping from verbal confidence to numeric scores.
    
    Returns:
        Dictionary mapping verbal confidence to numeric values
        
    TODO: Implement confidence score mapping
    """
    # TODO: Implement verbal to numeric confidence mapping
    # This will be used by the trust scorer for feature extraction
    return {
        "low": 0.3,
        "medium": 0.6, 
        "high": 0.9
    }
