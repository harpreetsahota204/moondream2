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
        "required": ["length"],
    },
    "query": {
        "params": {"query_text": str, "query_field": str},
         "required_one_of": ["query_text", "query_field"]
    },
    "detect": {
        "params": {"object_type": str},
        "required": ["object_type"],
    },
    "point": {
        "params": {"object_type": str},
        "required": ["object_type"],
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

class Moondream2(Model, SamplesMixin):
    """A FiftyOne model for running the Moondream2 model on images.
    
    Args:
        operation (str): Type of operation to perform
        revision (str, optional): Model revision/tag to use
        **kwargs: Operation-specific parameters
    """

    def __init__(
        self, 
        model_path: str,
        **kwargs
    ):
        if not model_path:
            raise ValueError("model_path is required")
            
        self.model_path = model_path
        
        # Operation and parameters will be set at apply time by default
        self.operation = None
        self.params = {}
        self.needs_fields = {}
        
        # If operation is provided in kwargs, set it now
        if "operation" in kwargs:
            operation = kwargs.pop("operation")
            self.set_operation(operation, **kwargs)

        
        
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

    def set_operation(self, operation: str, **kwargs):
        """Set the current operation and parameters.
        
        Args:
            operation (str): Operation type ('caption', 'query', 'detect', or 'point')
            **kwargs: Operation-specific parameters:
                - caption: requires 'length' ('short', 'normal', or 'long')
                - query: requires either 'query_text' (str) or 'query_field' (str)
                - detect: requires 'object_type' (str)
                - point: requires 'object_type' (str)
                
        Returns:
            self: For method chaining
            
        Raises:
            ValueError: If operation is invalid or required parameters are missing/invalid
        """
        if operation not in MOONDREAM_OPERATIONS:
            raise ValueError(
                f"Invalid operation: {operation}. Must be one of {list(MOONDREAM_OPERATIONS.keys())}"
            )

        # Get operation specifications
        op_spec = MOONDREAM_OPERATIONS[operation]
        param_types = op_spec["params"]

        # Reset needs_fields
        self.needs_fields = {}

        # Handle required parameters based on operation
        if operation == "query":
            # Check if at least one of required_one_of parameters is provided
            required_one_of = op_spec["required_one_of"]
            if not any(param in kwargs for param in required_one_of):
                raise ValueError(
                    f"Operation '{operation}' requires at least one of: {required_one_of}"
                )
                
            # If query_field is provided, set up needs_fields
            if "query_field" in kwargs:
                field_name = kwargs["query_field"]
                self.needs_fields = {field_name: field_name}
                
        else:
            # For other operations, check required parameters
            required_params = op_spec["required"]
            missing_params = [p for p in required_params if p not in kwargs]
            if missing_params:
                raise ValueError(
                    f"Operation '{operation}' requires parameters: {missing_params}"
                )

        # Validate parameter types
        for param, value in kwargs.items():
            if param in param_types:
                expected_type = param_types[param]
                
                # Handle special case for caption length
                if param == "length" and operation == "caption":
                    if value not in expected_type:
                        raise ValueError(
                            f"Invalid value for 'length': {value}. Must be one of {expected_type}"
                        )
                # Handle other parameters
                elif not isinstance(value, expected_type):
                    raise ValueError(
                        f"Parameter '{param}' must be of type {expected_type.__name__}"
                    )

        # Store operation and validated parameters
        self.operation = operation
        self.params = kwargs

        return self


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
        result = self.model.caption(image, length=self.params["length"])["caption"]

        return result

    def _predict_query(self, image: Image.Image, sample=None) -> Dict[str, str]:
        """Answer a visual query about an image.
        
        Args:
            image: PIL image
            sample: Optional FiftyOne sample
            
        Returns:
            dict: Query answer
        """
        # Get query text either directly or from sample field
        query_text = None
        
        if "query_field" in self.params and sample is not None:
            field_name = self.params["query_field"]
            query_text = sample[field_name]
        elif "query_text" in self.params:
            query_text = self.params["query_text"]
        else:
            raise ValueError("Either query_text parameter or query_field with sample must be provided")
            
        result = self.model.query(image, query_text)["answer"]

        return result.strip()

    def _predict_detect(self, image: Image.Image, sample=None) -> Dict[str, Detections]:
        """Detect objects in an image.
        
        Args:
            image: PIL image
            sample: Optional FiftyOne sample
            
        Returns:
            dict: Detection results
        """
        result = self.model.detect(image, self.params["object_type"])["objects"]

        detections = self._convert_to_detections(result, self.params["object_type"])

        return detections

    def _predict_point(self, image: Image.Image, sample=None) -> Dict[str, Keypoints]:
        """Identify point locations of objects in an image.
        
        Args:
            image: PIL image
            sample: Optional FiftyOne sample
            
        Returns:
            dict: Keypoint results
        """
        result = self.model.point(image, self.params["object_type"])["points"]

        keypoints = self._convert_to_keypoints(result, self.params["object_type"])

        return keypoints

    def _predict(self, image: Image.Image, sample=None) -> Dict[str, Any]:
        """Process a single image with Moondream2.
        
        Args:
            image: PIL image
            sample: Optional FiftyOne sample
            
        Returns:
            dict: Operation results
        """
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
        pil_image = Image.fromarray(image)
        return self._predict(pil_image, sample)