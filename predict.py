import re
import tempfile
import zipfile
import os
import time
import subprocess
import mimetypes
import json
from cog import BasePredictor, Input, Path
from autotrain.trainers.dreambooth.__main__ import train as train_dreambooth
from autotrain.trainers.dreambooth.params import DreamBoothTrainingParams

OUTPUT_DIR = './output_dir/'
BASE_MODEL_CACHE = "./base-model-cache"
BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
TEMP_TRAIN = './temp_training_dataset'
TEMP_CLASS = './temp_class_dataset'

class Predictor(BasePredictor):

    def setup(self):
        print("Started setup")
        if not os.path.exists(OUTPUT_DIR):
            print("Creating output folder")
            os.makedirs(OUTPUT_DIR)
            # self.\(BASE_MODEL_ID, BASE_MODEL_CACHE)
            # subprocess.check_call(["python", "script/download_weights.py"])

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
        args = {
            "model" : model,
            "prompt" : prompt,
            "seed" : seed,
            "resolution" : resolution,
            "center_crop" : center_crop,
            "train_text_encoder" : train_text_encoder,
            "batch_size" : batch_size,
            "num_steps" : num_steps,
            "checkpointing_steps" : checkpointing_steps,
            "gradient_accumulation" : gradient_accumulation,
            "lr" : lr,
            "scheduler" : scheduler,
            "warmup_steps" : warmup_steps,
            "num_cycles" : num_cycles,
            "use_8bit_adam" : use_8bit_adam,
            "xformers" : xformers,
            "xl" : xl
        }
        
        if class_prompt is not None: args['class_prompt'] = class_prompt

        # Unzip the training dataset
        with zipfile.ZipFile(str(train_data_zip), "r") as zip_ref:
            for zip_info in zip_ref.infolist():
                if zip_info.filename[-1] == "/" or zip_info.filename.startswith(
                    "__MACOSX"
                ):
                    continue
                mt = mimetypes.guess_type(zip_info.filename)
                if mt and mt[0] and mt[0].startswith("image/"):
                    zip_info.filename = os.path.basename(zip_info.filename)
                    zip_ref.extract(zip_info, TEMP_TRAIN)

        args['image_path'] = TEMP_TRAIN

        # Setup the parameters
        if mixed_precision == "bf16":
            args['bf16'] = True
        
        if mixed_precision == "fp16":
            args['fp16'] = True

        # output_dir: str = Path('lora_weights')
        args['project_name'] = 'project_name'

        if not os.path.exists(OUTPUT_DIR):
            print(f"------------- Creating output folder {OUTPUT_DIR}")
            os.makedirs(OUTPUT_DIR)

        # OPTIONAL PARAMS
        # Unzip the regularization dataset 
        if train_class_data_zip is not None:
            with zipfile.ZipFile(str(train_class_data_zip), "r") as zip_ref:
                for zip_info in zip_ref.infolist():
                    if zip_info.filename[-1] == "/" or zip_info.filename.startswith(
                        "__MACOSX"
                    ):
                        continue
                    mt = mimetypes.guess_type(zip_info.filename)
                    if mt and mt[0] and mt[0].startswith("image/"):
                        zip_info.filename = os.path.basename(zip_info.filename)
                        zip_ref.extract(zip_info, TEMP_CLASS)
            
            args['class_image_path'] = TEMP_CLASS


        dir_list = os.listdir(OUTPUT_DIR)
        print(dir_list)
    
        params = DreamBoothTrainingParams(**args)
        train_dreambooth(params)

        dir_list = os.listdir(OUTPUT_DIR)
        print(dir_list)    
    
        return Path(OUTPUT_DIR)
    

    # Only for testing
    def predict_mps(
        self,
        model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        prompt: str = "a photo of aisudhca shoes",
        train_data_zip: str = "./input/geox_captions.zip",
        mixed_precision: str = "fp16",
        train_class_data_zip: str = None,
        class_prompt: str = None,
        seed: int = 123456,
        resolution: int = "1024",
        center_crop: bool = False,
        train_text_encoder: bool = True,
        batch_size: int = 1,
        # sample_batch_size: int = Input(default=4, description="Sample batch size"),
        # epochs: int = Input(default=10, description="Number of training epochs"),
        num_steps: int = 2000,
        checkpointing_steps: int = 500,
        # resume_from_checkpoint: str = Input(default=None, description="Resume from checkpoint"),
        gradient_accumulation: int = 1,
        lr: float = 4e-4,
        #scale_lr: bool = Input(default=False, description="Scale learning rate"),
        scheduler: str = "polynomial",
        warmup_steps: int = 0,
        num_cycles: int = 1,
        # lr_power: float = 1.0,
        # dataloader_num_workers: int = 0,
        use_8bit_adam: bool = False,
        # adam_weight_decay: float = 1e-2,
        # adam_epsilon: float = Input(default=1e-8, description="Adam epsilon"),
        # max_grad_norm: float = Input(default=1.0, description="Max grad norm"),

        xformers: bool = False,
        xl: bool = True

    ):
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
        args = {
            "model" : model,
            "prompt" : prompt,
            "seed" : seed,
            "resolution" : resolution,
            "center_crop" : center_crop,
            "train_text_encoder" : train_text_encoder,
            "batch_size" : batch_size,
            "num_steps" : num_steps,
            "checkpointing_steps" : checkpointing_steps,
            "gradient_accumulation" : gradient_accumulation,
            "lr" : lr,
            "scheduler" : scheduler,
            "warmup_steps" : warmup_steps,
            "num_cycles" : num_cycles,
            "use_8bit_adam" : use_8bit_adam,
            "xformers" : xformers,
            "xl" : xl
        }
        
        if class_prompt is not None: args['class_prompt'] = class_prompt

        # Unzip the training dataset
        with zipfile.ZipFile(str(train_data_zip), "r") as zip_ref:
            for zip_info in zip_ref.infolist():
                if zip_info.filename[-1] == "/" or zip_info.filename.startswith(
                    "__MACOSX"
                ):
                    continue
                mt = mimetypes.guess_type(zip_info.filename)
                if mt and mt[0] and mt[0].startswith("image/"):
                    zip_info.filename = os.path.basename(zip_info.filename)
                    zip_ref.extract(zip_info, TEMP_TRAIN)

        args['image_path'] = TEMP_TRAIN

        # Setup the parameters
        if mixed_precision == "bf16":
            args['bf16'] = True
        
        if mixed_precision == "fp16":
            args['fp16'] = True

        # output_dir: str = Path('lora_weights')
        args['project_name'] = 'project_name'

        if not os.path.exists(OUTPUT_DIR):
            print(f"------------- Creating output folder {OUTPUT_DIR}")
            os.makedirs(OUTPUT_DIR)

        # OPTIONAL PARAMS
        # Unzip the regularization dataset 
        if train_class_data_zip is not None:
            with zipfile.ZipFile(str(train_class_data_zip), "r") as zip_ref:
                for zip_info in zip_ref.infolist():
                    if zip_info.filename[-1] == "/" or zip_info.filename.startswith(
                        "__MACOSX"
                    ):
                        continue
                    mt = mimetypes.guess_type(zip_info.filename)
                    if mt and mt[0] and mt[0].startswith("image/"):
                        zip_info.filename = os.path.basename(zip_info.filename)
                        zip_ref.extract(zip_info, TEMP_CLASS)
            
            args['class_image_path'] = TEMP_CLASS


        dir_list = os.listdir(OUTPUT_DIR)
        print(dir_list)
    
        params = DreamBoothTrainingParams(**args)
        train_dreambooth(params)

        dir_list = os.listdir(OUTPUT_DIR)
        print(dir_list)    
    
        return Path(OUTPUT_DIR)
    



def main():
    pred = Predictor()
    pred.setup()
    pred.predict_mps()


if __name__ == "__main__":
    main()


