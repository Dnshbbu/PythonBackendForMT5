from typing import Dict, Type, List
from model_base import BaseModel

class ModelFactory:
    _registry: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel]):
        """Register a new model type"""
        cls._registry[name] = model_class
        
    @classmethod
    def create(cls, name: str, **kwargs) -> BaseModel:
        """Create a model instance"""
        if name not in cls._registry:
            raise ValueError(f"Unknown model type: {name}")
        return cls._registry[name](**kwargs)
        
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of registered model types"""
        return list(cls._registry.keys()) 