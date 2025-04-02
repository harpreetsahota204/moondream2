"""
Florence-2 model wrapper for the FiftyOne Model Zoo.
"""
import logging
import os
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
import torch
from PIL import Image

from fiftyone import Model, SamplesMixin
from fiftyone.core.labels import Detection, Detections, Keypoint, Keypoints
from transformers import AutoModelForCausalLM, AutoTokenizer

MOONDREAM_OPERATIONS = {
    "caption": {
        "params": {"length": ["short", "normal", "long"]},
    },
    "query": {
        "params": {},
    },
    "detect": {
        "params": {},
    },
    "point": {
        "params": {},
    }
}

logger = logging.getLogger(__name__)

# Utility functions
def get_device():
    """Get the appropriate device for model inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class Moondream2(SamplesMixin, Model):
    """A FiftyOne model for running the Moondream2 model on images.
    
    Args:
        model_path (str): Path to model or HuggingFace model name
        operation (str, optional): Type of operation to perform
        prompt (str, optional): Prompt text to use
        **kwargs: Additional parameters
    """

    def __init__(
        self, 
        model_path: str,
        operation: str = None,
        prompt: str = None,
        **kwargs
    ):
        if not model_path:
            raise ValueError("model_path is required")
            
        self.model_path = model_path
        self._operation = None
        self._prompt = prompt
        self.params = {}
        self._fields = {}
        
        # Set operation if provided
        if operation:
            self.operation = operation
            
        # Store any additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
               
        # Set device
        self.device = get_device()
        logger.info(f"Using device: {self.device}")

        logger.info(f"Loading model from local path: {model_path}")

        print("\n" + "="*80)
        print("NOTICE: Creating necessary symbolic links for custom model code")
        print("When loading Moondream2 from a local directory,")
        print("the Transformers library expects to find Python modules in:")
        print(f"  ~/.cache/huggingface/modules/transformers_modules/moondream2/")
        print("rather than in your downloaded model directory.")
        print("Creating symbolic links to connect these locations...")
        print("="*80 + "\n")
        cache_dir = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/moondream2")
        os.makedirs(cache_dir, exist_ok=True)
        # Find all Python files in the model directory and create symlinks
        for file in os.listdir(model_path):
            if file.endswith('.py'):
                src = os.path.join(model_path, file)
                dst = os.path.join(cache_dir, file)
                # Create a symlink instead of copying
                if not os.path.exists(dst):
                    print(f"Creating symlink for {file}")
                    os.symlink(src, dst)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            revision=kwargs.get("revision"),
            trust_remote_code=True,
            local_files_only=True,
            device_map=self.device
        )

    @property
    def media_type(self):
        return "image"
    
    def _get_field(self):
        """Get the field name to use for prompt extraction."""
        if "prompt_field" in self.needs_fields:
            prompt_field = self.needs_fields["prompt_field"]
        else:
            prompt_field = next(iter(self.needs_fields.values()), None)
        return prompt_field

    @property
    def operation(self):
        """Get the current operation."""
        return self._operation

    @operation.setter
    def operation(self, value):
        """Set the operation with validation."""
        if value not in MOONDREAM_OPERATIONS:
            raise ValueError(f"Invalid operation: {value}. Must be one of {list(MOONDREAM_OPERATIONS.keys())}")
        self._operation = value

    @property
    def prompt(self):
        """Get the current prompt text."""
        return self._prompt

    @prompt.setter
    def prompt(self, value):
        """Set the prompt text."""
        self._prompt = value
        
    @property
    def length(self):
        """Get the caption length."""
        return self.params.get("length", "normal")

    @length.setter
    def length(self, value):
        """Set the caption length with validation."""
        valid_lengths = MOONDREAM_OPERATIONS["caption"]["params"]["length"]
        if value not in valid_lengths:
            raise ValueError(f"Invalid length: {value}. Must be one of {valid_lengths}")
        self.params["length"] = value


    def _convert_to_detections(self, boxes: List[Dict[str, float]], label: str) -> Detections:
        """Convert Moondream2 detection output to FiftyOne Detections.
        
        Args:
            boxes: List of bounding box dictionaries
            label: Object type label
            
        Returns:
            FiftyOne Detections object
        """
        detections = []

        for box in boxes:
            detection = Detection(
                label=label,
                bounding_box=[
                    box["x_min"],
                    box["y_min"],
                    box["x_max"] - box["x_min"],  # width
                    box["y_max"] - box["y_min"]   # height
                ]
            )

            detections.append(detection)
        
        return Detections(detections=detections)

    def _convert_to_keypoints(self, points: List[Dict[str, float]], label: str) -> Keypoints:
        """Convert Moondream2 point output to FiftyOne Keypoints.
        
        Args:
            points: List of point dictionaries
            label: Object type label
            
        Returns:
            FiftyOne Keypoints object
        """
        keypoints = []

        for idx, point in enumerate(points):

            keypoint = Keypoint(
                label=f"{label}",
                points=[[point["x"], point["y"]]]
            )

            keypoints.append(keypoint)
        
        return Keypoints(keypoints=keypoints)

    def _predict_caption(self, image: Image.Image, sample=None) -> Dict[str, str]:
        """Generate a caption for an image.
        
        Args:
            image: PIL image
            sample: Optional FiftyOne sample
            
        Returns:
            dict: Caption result
        """
        length = self.params.get("length", "normal")
        result = self.model.caption(image, length=length)["caption"]

        return result.strip()

    def _predict_query(self, image: Image.Image, sample=None) -> Dict[str, str]:
        """Answer a visual query about an image.
        
        Args:
            image: PIL image
            sample: Optional FiftyOne sample
            
        Returns:
            dict: Query answer
        """
        if not self.prompt:
            raise ValueError("No prompt provided for query operation")
            
        result = self.model.query(image, self.prompt)["answer"]

        return result.strip()

    def _predict_detect(self, image: Image.Image, sample=None) -> Dict[str, Detections]:
        """Detect objects in an image.
        
        Args:
            image: PIL image
            sample: Optional FiftyOne sample
            
        Returns:
            dict: Detection results
        """
        if not self.prompt:
            raise ValueError("No prompt provided for detect operation")

        result = self.model.detect(image, self.prompt)["objects"]

        detections = self._convert_to_detections(result, self.prompt)

        return detections

    def _predict_point(self, image: Image.Image, sample=None) -> Dict[str, Keypoints]:
        """Identify point locations of objects in an image.
        
        Args:
            image: PIL image
            sample: Optional FiftyOne sample
            
        Returns:
            dict: Keypoint results
        """
        if not self.prompt:
            raise ValueError("No prompt provided for point operation")

        result = self.model.point(image, self.prompt)["points"]

        keypoints = self._convert_to_keypoints(result, self.prompt)

        return keypoints

    def _predict(self, image: Image.Image, sample=None) -> Dict[str, Any]:
        """Process a single image with Moondream2.
        
        Args:
            image: PIL image
            sample: Optional FiftyOne sample
            
        Returns:
            dict: Operation results
        """
        # Centralized field handling
        if sample is not None and self._get_field() is not None:
            field_value = sample.get_field(self._get_field())
            if field_value is not None:
                self._prompt = str(field_value)
                
        if not self.operation:
            raise ValueError("No operation has been set")
                
        prediction_methods = {
            "caption": self._predict_caption,
            "query": self._predict_query,
            "detect": self._predict_detect,
            "point": self._predict_point
        }
        
        predict_method = prediction_methods.get(self.operation)

        if predict_method is None:
            raise ValueError(f"Unknown operation: {self.operation}")
            
        return predict_method(image, sample)

    def predict(self, image: np.ndarray, sample=None) -> Dict[str, Any]:
        """Process an image array with Moondream2.
        
        Args:
            image: numpy array image
            sample: Optional FiftyOne sample
            
        Returns:
            dict: Operation results
        """
        logger.info(f"Running {self.operation} operation")
        pil_image = Image.fromarray(image)
        return self._predict(pil_image, sample)