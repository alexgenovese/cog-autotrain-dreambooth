import re
import tempfile
import zipfile
import os
import time
import subprocess
import argparse
import json
from cog import BasePredictor, Input, Path
from autotrain.trainers.dreambooth.__main__ import train as train_dreambooth
from autotrain.trainers.dreambooth.params import DreamBoothTrainingParams


BASE_MODEL_CACHE = "./base-model-cache"
BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


def parse_args():
    # get training_config.json from the end user
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", type=str, required=True)
    return parser.parse_args()

class Predictor(BasePredictor):

    def setup(self):
        if not os.path.exists(BASE_MODEL_CACHE):
            # self.download_weights(BASE_MODEL_ID, BASE_MODEL_CACHE)
            subprocess.check_call(["python", "script/download_weights.py"])

    def download_weights(self, url, dest):
        start = time.time()
        print("downloading url: ", url)
        print("downloading to: ", dest)
        subprocess.check_call(["pqget", "-x", url, dest])
        print("downloading took: ", time.time() - start)
    
    def predict(
        self,
        model: str = Input(description="HF base model path", default="stabilityai/stable-diffusion-xl-base-1.0"),
        prompt: str = Input(description="Instance prompt", default="a photo of aisudhca bag"),
        train_data_zip: Path = Input(description="Upload image dataset in zip format"),
        mixed_precision: str = Input(description="FP16 or BF16", default="bf16", choices=["bf16", "fp16"]),
        train_class_data_zip: Path = Input(description="Upload image dataset in zip format - For Regularization"),
        class_prompt: str = Input(default=None, description="Class prompt"),
        seed: int = Input(default=42, description="Seed"),
        resolution: int = Input(default=1024, description="Resolution", choices=["512", "768", "1024"]),
        center_crop: bool = Input(default=False, description="Center crop"),
        train_text_encoder: bool = Input(default=True, description="Train text encoder"),
        batch_size: int = Input(default=1, description="Train batch size"),
        # sample_batch_size: int = Input(default=4, description="Sample batch size"),
        # epochs: int = Input(default=10, description="Number of training epochs"),
        num_steps: int = Input(default=2000, description="Max train steps"),
        checkpointing_steps: int = Input(default=500, description="Checkpointing steps"),
        # resume_from_checkpoint: str = Input(default=None, description="Resume from checkpoint"),

        gradient_accumulation: int = Input(default=1, description="Gradient accumulation steps"),

        lr: float = Input(default=4e-4, description="Learning rate"),
        #scale_lr: bool = Input(default=False, description="Scale learning rate"),
        scheduler: str = Input(default="constant", description="Learning rate scheduler", choices=[
                "linear", "cosine", "cosine_with_restarts", "polynomial",
                "constant", "constant_with_warmup"
        ]),
        warmup_steps: int = Input(default=0, description="Learning rate warmup steps"),
        num_cycles: int = Input(default=1, description="Learning rate num cycles"),
        # lr_power: float = Input(default=1.0, description="Learning rate power"),

        # dataloader_num_workers: int = Input(default=0, description="Dataloader num workers"),
        use_8bit_adam: bool = Input(default=False, description="Use 8bit adam"),
        # adam_weight_decay: float = Input(default=1e-2, description="Adam weight decay"),
        # adam_epsilon: float = Input(default=1e-8, description="Adam epsilon"),
        # max_grad_norm: float = Input(default=1.0, description="Max grad norm"),

        xformers: bool = Input(default=False, description="Enable xformers memory efficient attention"),
        xl: bool = Input(default=True, description="XL")
    ) -> Path:
        # Check all required params
        if model is None: 
            raise Exception("model is required.")
        if train_data_zip is None:
            raise Exception("train_data_zip is required.")
        if prompt is None: 
            raise Exception("prompt is required.")

        if num_cycles is None or num_cycles == 0:
            raise Exception("num_cycles can't be none or 0.")
        
        
        # Setup the args
        args = parse_args()

        args.model = model
        args.prompt = prompt
        if class_prompt is not None: args.class_prompt = class_prompt
        args.seed = seed
        args.resolution = resolution
        args.center_crop = center_crop
        args.train_text_encoder = train_text_encoder
        args.batch_size = batch_size
        args.num_steps = num_steps
        args.checkpointing_steps = checkpointing_steps
        args.gradient_accumulation = gradient_accumulation
        args.lr = lr
        args.scheduler = scheduler
        args.warmup_steps = warmup_steps
        args.num_cycles = num_cycles
        args.use_8bit_adam = use_8bit_adam
        args.xformers = xformers
        args.xl = xl


        # Unzip the training dataset
        train_data_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(train_data_zip, 'r') as zip_ref:
            zip_ref.extractall(train_data_dir)

        args.image_path = train_data_dir

        # Setup the parameters
        if mixed_precision == "bf16":
            args.bf16 = True
        
        if mixed_precision == "fp16":
            args.fp16 = True

        output_dir = Path(tempfile.mkdtemp())
        if not output_name:
            output_name = Path(re.sub("[^-a-zA-Z0-9_]", "", train_data_zip.name)).name
        args.project_name = output_dir

        # OPTIONAL PARAMS
        # Unzip the regularization dataset 
        if train_class_data_zip is not None:
            train_class_data_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(train_class_data_zip, 'r') as zip_ref:
                zip_ref.extractall(train_class_data_dir)

            args.class_image_path = train_class_data_dir

        
        params = DreamBoothTrainingParams(args)
        train_dreambooth(params)
    
        return output_dir