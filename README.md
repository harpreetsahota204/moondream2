# Moondream2 FiftyOne Remote Zoo Model Implementation 

Moondream2 is a powerful vision-language model that can be used with FiftyOne for various image understanding tasks. This implementation allows you to easily integrate Moondream2 into your FiftyOne workflows.

> NOTE: Due to recent changes in Transformers 4.50.0 (which are to be patched by Hugging Face) please ensure you have transformers<=4.49.0 installed before running the model


## Installation

```bash
pip install transformers<=4.49.0
# Add other dependencies as needed
```

## Features

The model supports four main operations:

1. **Caption Generation** (`caption`)
   - Generates image descriptions with adjustable length (short, normal, long)
   - No prompt required

2. **Visual Question Answering** (`query`)
   - Answers specific questions about image content
   - Requires a text prompt/question

3. **Object Detection** (`detect`)
   - Locates objects in images with bounding boxes
   - Requires a prompt specifying what to detect

4. **Point Identification** (`point`)
   - Identifies specific points of interest in images
   - Requires a prompt specifying what points to identify

## Technical Details

The model automatically selects the best available device:
- CUDA (GPU) if available
- Apple Metal (MPS) if available
- CPU as fallback
- The model requires local installation of model files
- Symbolic links are automatically created for custom model code
- Make sure to set appropriate operations and prompts before running inference

# Usage

# Moondream2 - FiftyOne Zoo Model Integration

This repository provides an implementation of Moondream2 as a remotely sourced model for the FiftyOne computer vision toolkit.

## Installation

To use Moondream2 with FiftyOne, follow these steps:

1. Register the model source:
```python
import fiftyone as fo
import fiftyone.zoo as foz

foz.register_zoo_model_source("https://github.com/harpreetsahota204/moondream2", overwrite=True)
```

2. Download the model:
```python
foz.download_zoo_model(
    "https://github.com/harpreetsahota204/moondream2",
    model_name="vikhyatk/moondream2"
)
```

3. Load the model:
```python
model = foz.load_zoo_model(
    "vikhyatk/moondream2",
    revision="2025-03-27"
)
```

## Usage Examples

| Operation | Description | Required Parameters |
|-----------|-------------|---------------------|
| `caption` | Generate image captions | `length`: "short", "normal", or "long" |
| `classify` | Zero shot classification | `prompt`: the classes you want the model to choose from |
| `detect` | Detect objects in images | `prompt`: object type to detect |
| `point` | Identify keypoints | `prompt`: type of object to locate points for, can be a Python list or a string list |
| `query` | Visual question answering | `prompt`: question about the image |

### Image Captioning

Moondream2 supports three caption length options: `short`, `normal`, and `long`.

#### Short Captions

```python
model.operation = "caption"
model.length = "short"

dataset.apply_model(
    model, 
    label_field="short_captions",
)

# Access the results
print(dataset.first()['short_captions'])
```

#### Long Captions

```python
model.length = "long"

dataset.apply_model(
    model, 
    label_field="long_captions",
)

# Access the results
print(dataset.first()['long_captions'])
```

### Switching Between Operations

The same model instance can be used for different operations by simply changing its properties:

### Zero-shot Classification

Classify images in a zero-shot manner

```python
model.operation="classify"
model.prompt= "Pick one of the animals the image: horse, giraffe, elephant, shark"

dataset.apply_model(
    model, 
    label_field="classification",
)
```

### Object Detection

Detect specific objects in images by providing a prompt:

```python
model.operation = "detect" 
model.prompt = "surfer, wave, bird" # you can also pass a Python list: ["surfer", "wave", "bird"]

dataset.apply_model(model, label_field="detections")

# Access the results - returns Detections objects with bounding boxes
print(dataset.first()['detections'])
```

### Keypoint Detection

Identify keypoints for specific object types:

```python
model.operation = "point"
model.prompt = "surfer, wave, bird" # you can also pass a Python list: ["surfer", "wave", "bird"]

dataset.apply_model(model, label_field="pointings")

# Access the results
print(dataset.first()['pointings'])
```

### Visual Question Answering (VQA)

Ask questions about the content of images. This can be used in a variety of ways, for example you can ask it to perfom OCR.

```python
model.operation = "query"
model.prompt = "What is in the background of the image"

dataset.apply_model(model, label_field="vqa_response")

# Access the results
print(dataset.first()['vqa_response'])
```

### Using Sample Fields for Prompts

You can also use fields from your dataset as prompts:

```python
# Set a field with questions for each sample
dataset.set_values("questions", ["Where is the general location of this scene?"] * len(dataset))

# Use that field as the prompt source
dataset.apply_model(
    model,
    label_field="query_field_response",
    prompt_field="questions"
)

# Access the results
print(dataset.first()['query_field_response'])
```

## Output Formats

Moondream2 returns different types of output depending on the operation:

* Captions: Returns string
* Classify: Returns `fiftyone.core.labels.Classifications`
* Detection: Returns `fiftyone.core.labels.Detections`
* Phrase Grounding: Returns `fiftyone.core.labels.Detections`
* Segmentation: Returns `fiftyone.core.labels.Polylines`


# Citation

```bibtex
@misc{moondream2024,
    author = {Korrapati, Vikhyat and others},
    title = {Moondream: A Tiny Vision Language Model},
    year = {2024},
    publisher = {GitHub},
    journal = {GitHub repository},
    url = {https://github.com/vikhyat/moondream},
    commit = {main}
}
```