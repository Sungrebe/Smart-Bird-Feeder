from transformers import CLIPProcessor, CLIPModel
from PIL import Image, UnidentifiedImageError

def load_model_and_processor(cache_dirpath):
    """
        Loads the pretrained CLIP model and processor from Hugging Face
    """
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=cache_dirpath)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=cache_dirpath)

    return model, processor

def process_image(image_path, model_path):
    """
        Generate embeddings for a given image using the specified model (finetuned or pretrained)
        Args: image_path (string), model_path (string)
        Returns: a tensor representation of the image embedding
    """
    pretrained_model, pretrained_processor = load_model_and_processor(model_path)

    try:
        input = pretrained_processor(
            images=Image.open(image_path),
            return_tensors="pt",
        )

        print(f"input={input}")
        
        return pretrained_model.vision_model(**input).pooler_output
    except UnidentifiedImageError:
        return None