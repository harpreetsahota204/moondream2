from huggingface_hub import snapshot_download

"""
Moondream2 model from https://huggingface.co/vikhyatk/moondream2
"""
import logging
import os

from huggingface_hub import snapshot_download
from fiftyone.operators import types

# Import constants from zoo.py to ensure consistency
from .zoo import MOONDREAM_OPERATIONS, Moondream2

MOONDREAM_MODES = {
    "caption": "Caption images", 
    "query": "Visual question answering",
    "detect": "Object detection",
    "point": "Apply point on object",
}


logger = logging.getLogger(__name__)

def download_model(model_name, model_path):
    """Downloads the model.

    Args:
        model_name: the name of the model to download, as declared by the
            ``base_name`` and optional ``version`` fields of the manifest
        model_path: the absolute filename or directory to which to download the
            model, as declared by the ``base_filename`` field of the manifest
    """
    
    snapshot_download(repo_id=model_name, local_dir=model_path, revision='2025-03-27')

def load_model(model_name, model_path, **kwargs):
    """Loads the model.

    Args:
        model_name: the name of the model to load, as declared by the
            ``base_name`` and optional ``version`` fields of the manifest
        model_path: the absolute filename or directory to which the model was
            donwloaded, as declared by the ``base_filename`` field of the
            manifest
        **kwargs: optional keyword arguments that configure how the model
            is loaded

    Returns:
        a :class:`fiftyone.core.models.Model`
    """
    
    if not model_path or not os.path.isdir(model_path):
        raise ValueError(
            f"Invalid model_path: '{model_path}'. Please ensure the model has been downloaded "
            "using fiftyone.zoo.download_zoo_model('voxel51/moondream')"
        )
    
    logger.info(f"Loading moondream2 model from {model_path}")

    # Create and return the model - operations specified at apply time
    return Moondream2(model_path=model_path, **kwargs)


def resolve_input(self, ctx):
        """Implement this method to collect user inputs as parameters
        that are stored in `ctx.params`.

        Returns:
            a `types.Property` defining the form's components
        """
        inputs = types.Object()

        mode_dropdown = types.Dropdown(label="What would you like to use Moondream2 for?")
        
        for k, v in MOONDREAM_MODES.items():
            mode_dropdown.add_choice(k, label=v)

        inputs.enum(
            "operation",
            values=mode_dropdown.values(),
            label="Moondream2 Tasks",
            description="Select from one of the supported tasks.",
            view=mode_dropdown,
            required=True
        )

        length_radio = types.RadioGroup()
        length_radio.add_choice("short", label="A short caption")
        length_radio.add_choice("normal", label="A more descriptive caption")        
        
        chosen_task = ctx.params.get("operation")

        if chosen_task == "caption":
            inputs.enum(
                "length",
                label="Caption Length",
                description="Which caption type would you like?",
                required=True,
                values=length_radio.values(),
                view=length_radio
            )

        if chosen_task == "query":
            inputs.str(
                "query_text",
                label="Query",
                description="What's your query?",
                required=True,
            )

        if chosen_task == "detect":
            inputs.str(
                "object_type",
                label="Detect",
                description="What do you want to detect? Currently this model only supports passing one object.",
                required=True,
            )

        if chosen_task == "point":
            inputs.str(
                "object_type",
                label="Point",
                description="What do you want to place a point on? Currently this model only supports passing one object",
                required=True,
            )
       

        inputs.str(
            "output_field",            
            required=True,
            label="Output Field",
            description="Name of the field to store the results in."
            )
        
        inputs.bool(
            "delegate",
            default=False,
            required=True,
            label="Delegate execution?",
            description=("If you choose to delegate this operation you must first have a delegated service running. "
            "You can launch a delegated service by running `fiftyone delegated launch` in your terminal"),
            view=types.CheckboxView(),
        )

        inputs.view_target(ctx)

        return types.Property(inputs)