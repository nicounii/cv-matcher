import os
import google.generativeai as genai
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class GeminiModelManager:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.preferred_model = os.getenv("GEMINI_MODEL", "auto")  # New env variable
        
        # Latest models in order of preference (newest first)
        self.available_models = [
            # Gemini 2.5 series (latest)
            "gemini-2.5-flash",
            "gemini-2.5-pro", 
            "gemini-2.5-flash-lite",
            
            # Gemini 2.0 series
            "gemini-2.0-flash-001",
            "gemini-2.0-flash-lite-001",
            
            # Gemini 1.5 series (fallback)
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash-8b",
            
            # Older models (last resort)
            "gemini-1.0-pro",
            "gemini-pro",
            "models/gemini-pro"
        ]
        
        self._working_model = None
        self._model_instance = None
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
    
    def get_model(self, generation_config: Optional[Dict[str, Any]] = None) -> Optional[genai.GenerativeModel]:
        """Get a working Gemini model with automatic fallback"""
        
        if generation_config is None:
            generation_config = {
                "temperature": float(os.getenv("GEMINI_TEMPERATURE", "0.0")),
                "top_p": float(os.getenv("GEMINI_TOP_P", "1.0")),
                "top_k": int(os.getenv("GEMINI_TOP_K", "1")),
            }
        
        # If we already have a working model, return it
        if self._model_instance and self._working_model:
            return self._model_instance
        
        # If user specified a specific model, try that first
        models_to_try = []
        
        if self.preferred_model != "auto":
            models_to_try.append(self.preferred_model)
            models_to_try.extend(self.available_models)
        else:
            models_to_try = self.available_models
        
        # Try each model until one works
        for model_name in models_to_try:
            try:
                logger.info(f"Trying Gemini model: {model_name}")
                
                model = genai.GenerativeModel(
                    model_name,
                    generation_config=generation_config
                )
                
                # Test the model with a simple request
                test_response = model.generate_content(
                    "Respond with exactly: MODEL_TEST_OK",
                    generation_config={"temperature": 0}
                )
                
                if "MODEL_TEST_OK" in test_response.text:
                    logger.info(f"✅ Successfully connected to: {model_name}")
                    self._working_model = model_name
                    self._model_instance = model
                    
                    # Save working model to env for future use
                    os.environ["GEMINI_WORKING_MODEL"] = model_name
                    
                    return model
                
            except Exception as e:
                logger.debug(f"❌ {model_name} failed: {str(e)[:100]}")
                continue
        
        logger.error("❌ No working Gemini model found")
        return None
    
    def get_working_model_name(self) -> Optional[str]:
        """Get the name of the currently working model"""
        if not self._working_model:
            model = self.get_model()
            if not model:
                return None
        return self._working_model
    
    def list_available_models(self) -> list:
        """List all models available in your Google Cloud project"""
        if not self.api_key:
            return []
        
        try:
            models = []
            for model in genai.list_models():
                if 'generateContent' in model.supported_generation_methods:
                    models.append(model.name)
            return models
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def test_all_models(self) -> Dict[str, bool]:
        """Test all models and return their status"""
        results = {}
        
        for model_name in self.available_models:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content("Test")
                results[model_name] = True
            except Exception:
                results[model_name] = False
        
        return results

# Global instance
gemini_manager = GeminiModelManager()
