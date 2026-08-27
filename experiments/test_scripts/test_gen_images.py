import argparse
import base64
import json
import pandas as pd
import gc
import os
from pathlib import Path
import torch
import traceback
from dotenv import load_dotenv

# Initialize the seed 
SEED = 42

def load_environment() -> str:
    """
    Loads the environment from the .env file in os system variables.
    
    return: (str) Access token. 
    
    """

    # Load the environment files
    load_dotenv("../../.env")
    # Load the access token
    access_token = os.environ["HF_TOKEN"] if os.environ["HF_TOKEN"] else False

    return access_token

def extract_code_and_robust(code_file,file_name:str):
    """
    Extracts the skill code from the file name
    """

    # Check if robust in string or not
    robust = "robust" in file_name
    skills = file_name[:-12] if robust else file_name[:-5]
    # Retrieve the skill code
    code = code_file.loc[code_file["skill"]==skills, "code"]
    # Check if the code is not empty
    if not code.empty: 
        code = code.iloc[0]
    else: 
        code = -1

    return code,robust

def extract_prompt_info(prompt:dict):
    """
    """

    # Get the id 
    prompt_number = prompt["id"].split("_")[1]
    # Get the level 
    prompt_level = prompt["level"]
    # Get the synthetic prompts
    synthetic_prompts = prompt["synthetic_prompts"]

    return prompt_number,prompt_level,synthetic_prompts

def set_output_name(model_name:str,skill_code,prompt_level:str,prompt_number:str, synthetic_prompt_number:int, output_folder_name:str, robust:bool):
    """
    """

    # Initialize the variable
    output_file_name = ""

    if robust:
        # Add the robust tag
        output_file_name = output_folder_name + "/" +  model_name + "_" + str(skill_code) + "_" + prompt_level + "_" + str(prompt_number) + "_" + str(synthetic_prompt_number) + "_robust"
    else: 
        output_file_name = output_folder_name + "/" +  model_name + "_" + str(skill_code) + "_" + prompt_level + "_" + str(prompt_number) + "_" + str(synthetic_prompt_number)

    return output_file_name


def generate_image_animagine(code_file,GPU="0"):
    """
    """
    # Set the json outputs directory (.json)
    json_dir = Path("../../outputs/prompts")
    # Get the list of json files
    json_files = [p for p in json_dir.glob("*") if p.is_file()]
    # Set the image output directory
    output_dir_name = "../../data/images_refactored/animagine"
    output_dir = Path(output_dir_name)
    # Get the list of files to generate
    output_files = [p._str for p in output_dir.glob("*") if p.is_file()]

    try: 
        # Import relevant packages
        from diffusers import StableDiffusionXLPipeline
        # Setup the pipe
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "cagliostrolab/animagine-xl-4.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            custom_pipeline="lpw_stable_diffusion_xl",
            add_watermarker=False
        ).to(f"cuda:{GPU}")
        # Set the elements 
        model_name = "animagine"
        # Display a message
        print("\nGenerating images with animagine.\n")
        # Loop through the files
        for i,json_file in enumerate(json_files):
            print(json_file)
            # Extract the skill code
            skill_code, robust = extract_code_and_robust(code_file,os.path.basename(json_file))
            # Initialize the collection
            elements = dict()
            # Open the file
            with open(json_file, "rb") as file:
                # Load the elements into a dict
                elements = json.load(file)
                # Loop through the prompts
                for prompt in elements["prompts"]:
                    # Get the prompt infos
                    prompt_number, prompt_level, synthetic_prompts = extract_prompt_info(prompt)
                    # Loop through the synthetic prompts
                    for j,gen_prompt in enumerate(synthetic_prompts):
                        # Set the output file name
                        output_file_name = set_output_name(model_name, skill_code, 
                                                           prompt_level,prompt_number,j,output_dir_name, robust)
                        # Check if the file doesn't exist
                        if output_file_name+".png" not in output_files:
                            # Display the output file name
                            print(f"\nOutput file name: {output_file_name}")
                            # Run inference ; generate the image
                            image = pipe(
                                gen_prompt, 
                                width=1024,
                                height=1024, 
                                guidance_scale=5.0, 
                                num_inference_steps=50, 
                                generator=torch.Generator(f"cuda:{GPU}").manual_seed(SEED) #Setting the seed to be more deterministic
                            ).images[0]
                            # Save image
                            image.save(output_file_name+".png")
                            # Display success message
                            print(f"\nImage generated for : {output_file_name}. Prompt: {gen_prompt}")
        # Free memory
        del image, pipe
        # Collect garbage
        gc.collect()
        # Empty cuda cache
        torch.cuda.empty_cache()
        # Collect garbage
        torch.cuda.ipc_collect()
        
        return 1
    except Exception:
        # Display exception
        traceback.print_exc()

def generate_image_stable_diffusion(code_file,GPU="0"):
    """
    Generates images using the Stable Diffusion model. 
    Doesn't generate if the output file already exists.
    Inference parameters are the same for most models : 
    - Inference steps : 50
    - Guidance scale : 5.0
    - Width and Height : 1024x1024
    We use textual inversion from SD embed for the model to support "longer" prompts. 
    """

    # Set the json outputs directory (.json)
    json_dir = Path("../../outputs/prompts")
    # Get the list of json files
    json_files = [p for p in json_dir.glob("*") if p.is_file()]
    json_files.sort()
    # Set the image output directory
    output_dir_name = "../../data/images_refactored/stable_diffusion"
    output_dir = Path(output_dir_name)
    # Get the list of files to generate
    output_files = [p._str for p in output_dir.glob("*") if p.is_file()]
    output_files.sort()

    try: 
        # Import relevant packages
        from diffusers import StableDiffusionXLPipeline
        from sd_embed.embedding_funcs import get_weighted_text_embeddings_sdxl
        # Setup the pipe
        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", 
                                                 torch_dtype=torch.float16, 
                                                 use_safetensors=True,
                                                 variant="fp16").to(f"cuda:{GPU}")
        
        # Set the elements 
        model_name = "stable_diffusion"
        # Display a message
        print("\nGenerating images with stable diffusion.\n")
        # Loop through the files
        for i,json_file in enumerate(json_files):
            print(json_file)
            # Extract the skill code
            skill_code, robust = extract_code_and_robust(code_file,os.path.basename(json_file))
            # Initialize the collection
            elements = dict()
            # Open the file
            with open(json_file, "rb") as file:
                # Load the elements into a dict
                elements = json.load(file)
                # Loop through the prompts
                for prompt in elements["prompts"]:
                    # Get the prompt infos
                    prompt_number, prompt_level, synthetic_prompts = extract_prompt_info(prompt)
                    # Loop through the synthetic prompts
                    for j,gen_prompt in enumerate(synthetic_prompts):
                        # Set the output file name
                        output_file_name = set_output_name(model_name=model_name, 
                                                           skill_code=skill_code, 
                                                           prompt_level=prompt_level,
                                                           prompt_number=prompt_number,
                                                           synthetic_prompt_number=j,
                                                           output_folder_name=output_dir_name, 
                                                           robust=robust)
                        # Check if the file doesn't exist
                        if output_file_name+".png" not in output_files:
                            # Display the output file name
                            print(f"\nOutput file name: {output_file_name}")
                            # Run inference ; generate the image
                            with torch.no_grad():
                                # Adding support for long prompts
                                (prompt_embeds, 
                                 prompt_neg_embeds,
                                 pooled_prompt_embeds,
                                 negative_pooled_prompt_embeds) = get_weighted_text_embeddings_sdxl(
                                     pipe,
                                     prompt = gen_prompt,
                                )
                                image = pipe(
                                    prompt_embeds=prompt_embeds,
                                    negative_prompt_embeds=prompt_neg_embeds,
                                    pooled_prompt_embeds=pooled_prompt_embeds,
                                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                                    width=1024,
                                    height=1024, 
                                    guidance_scale=5, 
                                    num_inference_steps=50, 
                                    generator=torch.Generator(f"cuda:{GPU}").manual_seed(SEED) #Setting the seed to be more deterministic
                                ).images[0]
                                # Save image
                                image.save(output_file_name+".png")
                            # Display success message
                            print(f"\nImage generated for : {output_file_name}. Prompt: {gen_prompt}")
        # Free memory
        del image, pipe
        # Collect garbage
        gc.collect()
        # Empty cuda cache
        torch.cuda.empty_cache()
        # Collect garbage
        torch.cuda.ipc_collect()
        
        return 1
    except Exception:
        # Display exception
        traceback.print_exc()

def generate_image_z_image_turbo(code_file,GPU="0"):
    """
    Generates images using the Z-image Turbo model. 
    Doesn't generate if the output file already exists.
    Inference parameters : 
    - Inference steps : 9
    - Guidance scale : 0.0
    - Width and Height : 1024x1024
    We use do not use textual inversion from SD embed since the model already supports longer prompts. 
    """

    # Set the json outputs directory (.json)
    json_dir = Path("../../outputs/prompts")
    # Get the list of json files
    json_files = [p for p in json_dir.glob("*") if p.is_file()]
    json_files.sort()
    # Set the image output directory
    output_dir_name = "../../data/images_refactored/z_image_turbo"
    output_dir = Path(output_dir_name)
    # Get the list of files to generate
    output_files = [p._str for p in output_dir.glob("*") if p.is_file()]
    output_files.sort()

    try: 
        # Import relevant packages
        from diffusers import DiffusionPipeline
        # Setup the pipe
        pipe = DiffusionPipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", 
                                                 torch_dtype=torch.bfloat16, 
                                                 use_safetensors=True).to(f"cuda:{GPU}")
        
        # Set the elements 
        model_name = "z_image_turbo"
        # Display a message
        print("\nGenerating images with z_image_turbo.\n")
        # Loop through the files
        for i,json_file in enumerate(json_files):
            print(json_file)
            # Extract the skill code
            skill_code, robust = extract_code_and_robust(code_file,os.path.basename(json_file))
            # Initialize the collection
            elements = dict()
            # Open the file
            with open(json_file, "rb") as file:
                # Load the elements into a dict
                elements = json.load(file)
                # Loop through the prompts
                for prompt in elements["prompts"]:
                    # Get the prompt infos
                    prompt_number, prompt_level, synthetic_prompts = extract_prompt_info(prompt)
                    # Loop through the synthetic prompts
                    for j,gen_prompt in enumerate(synthetic_prompts):
                        # Set the output file name
                        output_file_name = set_output_name(model_name=model_name, 
                                                           skill_code=skill_code, 
                                                           prompt_level=prompt_level,
                                                           prompt_number=prompt_number,
                                                           synthetic_prompt_number=j,
                                                           output_folder_name=output_dir_name, 
                                                           robust=robust)
                        # Check if the file doesn't exist
                        if output_file_name+".png" not in output_files:
                            # Display the output file name
                            print(f"\nOutput file name: {output_file_name}")
                            # Run inference ; generate the image
                            with torch.no_grad():
                                image = pipe(
                                    prompt=gen_prompt,
                                    width=1024,
                                    height=1024, 
                                    guidance_scale=0.0, 
                                    num_inference_steps=9, 
                                    generator=torch.Generator(f"cuda:{GPU}").manual_seed(SEED) #Setting the seed to be more deterministic
                                ).images[0]
                                # Save image
                                image.save(output_file_name+".png")
                            # Display success message
                            print(f"\nImage generated for : {output_file_name}. Prompt: {gen_prompt}")
        # Free memory
        del image, pipe
        # Collect garbage
        gc.collect()
        # Empty cuda cache
        torch.cuda.empty_cache()
        # Collect garbage
        torch.cuda.ipc_collect()
        
        return 1
    except Exception:
        # Display exception
        traceback.print_exc()


def generate_image_flux(code_file,access_token,GPU="0"):
    """
    Generates images using the Flux 1.0 model. 
    Doesn't generate if the output file already exists.
    Inference parameters are the same for most models : 
    - Inference steps : 4
    - Guidance scale : 3.5
    - Width and Height : 1024x1024
    We use do not use textual inversion from SD embed since the model already supports longer prompts. 
    """

    # Set the json outputs directory (.json)
    json_dir = Path("../../outputs/prompts")
    # Get the list of json files
    json_files = [p for p in json_dir.glob("*") if p.is_file()]
    json_files.sort()
    # Set the image output directory
    output_dir_name = "../../data/images_refactored/flux"
    output_dir = Path(output_dir_name)
    # Get the list of files to generate
    output_files = [p._str for p in output_dir.glob("*") if p.is_file()]
    output_files.sort()

    try: 
        # Import relevant packages
        from diffusers import DiffusionPipeline
        #from accelerate import Accelerator
        # Setup the pipe
        pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", 
                                            dtype=torch.bfloat16, 
                                            use_safetensors=True, 
                                            token=access_token, 
                                            device_map="balanced",
                                            max_memory={
                                                0: "40GiB", 
                                                1 : "40GiB"
                                            })
        # Setting up the accelerator
        #accelerator = Accelerator(mixed_precision="bf16")
        # Send the pipe to accelerator
        #pipe = accelerator.prepare(pipe)
        
        from sd_embed.embedding_funcs import get_weighted_text_embeddings_flux1
        # Optimize memory
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        #pipe.enable_sequential_cpu_offload()
        # Setup the generator
        device = pipe._execution_device
        generator = torch.Generator(device).manual_seed(SEED)
        
        # Set the elements 
        model_name = "flux"
        # Display a message
        print("\nGenerating images with flux 1.0.\n")
        # Loop through the files
        for i,json_file in enumerate(json_files):
            print(json_file)
            # Extract the skill code
            skill_code, robust = extract_code_and_robust(code_file,os.path.basename(json_file))
            # Initialize the collection
            elements = dict()
            # Open the file
            with open(json_file, "rb") as file:
                # Load the elements into a dict
                elements = json.load(file)
                # Loop through the prompts
                for prompt in elements["prompts"]:
                    # Get the prompt infos
                    prompt_number, prompt_level, synthetic_prompts = extract_prompt_info(prompt)
                    # Loop through the synthetic prompts
                    for j,gen_prompt in enumerate(synthetic_prompts):
                        # Set the output file name
                        output_file_name = set_output_name(model_name=model_name, 
                                                           skill_code=skill_code, 
                                                           prompt_level=prompt_level,
                                                           prompt_number=prompt_number,
                                                           synthetic_prompt_number=j,
                                                           output_folder_name=output_dir_name, 
                                                           robust=robust)
                        # Check if the file doesn't exist
                        if output_file_name+".png" not in output_files:
                            # Display the output file name
                            print(f"\nOutput file name: {output_file_name}")
                            # Run inference ; generate the image
                            with torch.inference_mode():
                                # Adding support for longer prompts
                                prompt_embeds, pooled_prompt_embeds = get_weighted_text_embeddings_flux1(
                                    pipe=pipe, 
                                    prompt=gen_prompt, 
                                    device=pipe._execution_device
                                )
                                image = pipe(
                                    prompt_embeds=prompt_embeds,
                                    pooled_prompt_embeds=pooled_prompt_embeds,
                                    guidance_scale=3.5, 
                                    num_inference_steps=4, 
                                    generator=generator,#Setting the seed to be more deterministic
                                ).images[0]
                                # Save image
                                image.save(output_file_name+".png")
                            # Display success message
                            print(f"\nImage generated for : {output_file_name}. Prompt: {gen_prompt}")
        # Free memory
        del image, pipe
        # Collect garbage
        gc.collect()
        # Empty cuda cache
        torch.cuda.empty_cache()
        # Collect garbage
        torch.cuda.ipc_collect()
        
        return 1
    except Exception:
        # Display exception
        traceback.print_exc()


def generate_image_qwen(code_file,access_token,GPU="0"):
    """
    Generates images using the Qwen-Image model. 
    Doesn't generate if the output file already exists.
    Inference parameters are the same for most models : 
    - Inference steps : 50
    - Guidance scale : 5.0
    - Width and Height : 1664 X 928
    We use do not use textual inversion from SD embed since the model already supports longer prompts. 
    """

    # Set the json outputs directory (.json)
    json_dir = Path("../../outputs/prompts")
    # Get the list of json files
    json_files = [p for p in json_dir.glob("*") if p.is_file()]
    json_files.sort()
    # Set the image output directory
    output_dir_name = "../../data/images_refactored/qwen_image"
    output_dir = Path(output_dir_name)
    # Get the list of files to generate
    output_files = [p._str for p in output_dir.glob("*") if p.is_file()]
    output_files.sort()

    try: 
        # Get the Comfy UI path
        comfy_path = os.environ["MOUNTED_COMFY_UI_PATH"]
        # Import relevant packages
        from diffusers import QwenImagePipeline
        # Setup the transformer
        torch_dtype = torch.bfloat16
        # Setup the pipe
        pipe = QwenImagePipeline.from_pretrained("Qwen/Qwen-Image", 
                                                 use_safetensors=True, 
                                                 token=access_token,
                                                 torch_dtype=torch_dtype,
                                                 low_cpu_mem_usage=True,
                                                 max_memory={
                                                0: "45GiB", 
                                                1 : "40GiB", 
                                                2: "40 GiB",
                                                })
        # Setting the generator
        generator = torch.Generator("cuda").manual_seed(SEED)
        # Set the elements 
        model_name = "qwen_image"
        # Display a message
        print("\nGenerating images with Qwen-Image.\n")
        # Loop through the files
        for i,json_file in enumerate(json_files):
            print(json_file)
            # Extract the skill code
            skill_code, robust = extract_code_and_robust(code_file,os.path.basename(json_file))
            # Initialize the collection
            elements = dict()
            # Open the file
            with open(json_file, "rb") as file:
                # Load the elements into a dict
                elements = json.load(file)
                # Loop through the prompts
                for prompt in elements["prompts"]:
                    # Get the prompt infos
                    prompt_number, prompt_level, synthetic_prompts = extract_prompt_info(prompt)
                    # Loop through the synthetic prompts
                    for j,gen_prompt in enumerate(synthetic_prompts):
                        # Set the output file name
                        output_file_name = set_output_name(model_name=model_name, 
                                                           skill_code=skill_code, 
                                                           prompt_level=prompt_level,
                                                           prompt_number=prompt_number,
                                                           synthetic_prompt_number=j,
                                                           output_folder_name=output_dir_name, 
                                                           robust=robust)
                        # Check if the file doesn't exist
                        if output_file_name+".png" not in output_files:
                            # Display the output file name
                            print(f"\nOutput file name: {output_file_name}")
                            # Run inference ; generate the image
                            with torch.inference_mode():
                                # Generate the image
                                image = pipe(
                                    prompt=gen_prompt,
                                    width=1664,
                                    height=928, 
                                    guidance_scale=4.0, 
                                    num_inference_steps=20, 
                                    generator=generator
                                    )                            
                            # Handle both possible return types 
                            if hasattr(image, "images"): 
                                image = image.images[0] 
                            else: 
                                image = image[0]
                            # Save image
                            image.save(output_file_name+".png")
                            # Display success message
                            print(f"\nImage generated for : {output_file_name}. Prompt: {gen_prompt}")
            # Free memory inside the loop, fro QWEN
            del image, pipe
            # Collect garbage
            gc.collect()
            # Empty cuda cache
            torch.cuda.empty_cache()
            # Collect garbage
            torch.cuda.ipc_collect()
        
        return 1
    except Exception:
        # Display exception
        traceback.print_exc()

def generate_image_stable_cascade(code_file, access_token):
    """
    Generates images using the Stable Cascade model. 
    Doesn't generate if the output file already exists.
    Inference parameters are the same for most models : 
    - Inference steps : 30
    - Guidance scale : 3.0
    - Width and Height : 1024x1024
    We use textual inversion from SD-embed to add support for longer prompts. 
    """

    # Set the json outputs directory (.json)
    json_dir = Path("../../outputs/prompts")
    # Get the list of json files
    json_files = [p for p in json_dir.glob("*") if p.is_file()]
    json_files.sort()
    # Set the image output directory
    output_dir_name = "../../data/images_refactored/stable_cascade"
    output_dir = Path(output_dir_name)
    # Get the list of files to generate
    output_files = [p._str for p in output_dir.glob("*") if p.is_file()]
    output_files.sort()

    try: 
        # Import relevant packages
        from sd_embed.embedding_funcs import get_weighted_text_embeddings_s_cascade
        from diffusers import StableCascadePriorPipeline, StableCascadeDecoderPipeline, StableCascadeUNet
        # Display a message 
        print("Loading the prior (Unet)...")
        # Loading the unet
        prior_unet = StableCascadeUNet.from_single_file(
            "../../../ComfyUI/models/stable_cascade/stage_c_bf16.safetensors",
            torch_dtype=torch.bfloat16, 
            local_files_only=True,
        ).to("cuda")
        # Display a message 
        print("Loading the decoder (Unet)...")
        decoder_unet = StableCascadeUNet.from_single_file(
            "../../../ComfyUI/models/stable_cascade/stage_b_bf16.safetensors",
            torch_dtype=torch.bfloat16,
            local_files_only=True
        ).to("cuda")
        # Compile the unet
        prior_unet = torch.compile(prior_unet, mode="reduce-overhead") 
        decoder_unet = torch.compile(decoder_unet, mode="reduce-overhead")
        # Set the prior
        prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", 
                                                           prior=prior_unet, 
                                                           torch_dtype=torch.bfloat16).to("cuda")
        # Set the decoder
        decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", 
                                                       decoder=decoder_unet, 
                                                       torch_dtype=torch.bfloat16).to("cuda")
        # Enable flash attention
        prior.enable_xformers_memory_efficient_attention() 
        decoder.enable_xformers_memory_efficient_attention()
        # Setting the generator
        generator = torch.Generator("cuda").manual_seed(SEED)
        # Set the elements 
        model_name = "stable_cascade"
        # Display a message
        print("\nGenerating images with Stable Cascade.\n")
        # Loop through the files
        for i,json_file in enumerate(json_files):
            print(json_file)
            # Extract the skill code
            skill_code, robust = extract_code_and_robust(code_file,os.path.basename(json_file))
            # Initialize the collection
            elements = dict()
            # Open the file
            with open(json_file, "rb") as file:
                # Load the elements into a dict
                elements = json.load(file)
                # Loop through the prompts
                for prompt in elements["prompts"]:
                    # Get the prompt infos
                    prompt_number, prompt_level, synthetic_prompts = extract_prompt_info(prompt)
                    # Loop through the synthetic prompts
                    for j,gen_prompt in enumerate(synthetic_prompts):
                        # Set the output file name
                        output_file_name = set_output_name(model_name=model_name, 
                                                           skill_code=skill_code, 
                                                           prompt_level=prompt_level,
                                                           prompt_number=prompt_number,
                                                           synthetic_prompt_number=j,
                                                           output_folder_name=output_dir_name, 
                                                           robust=robust)
                        # Check if the file doesn't exist
                        if output_file_name+".png" not in output_files:
                            # Display the output file name
                            print(f"\nOutput file name: {output_file_name}")
                            # Run inference ; generate the image
                            with torch.inference_mode():
                                # Display a message
                                print("Generating prompt embeddings...")
                                # Get the prompt embeddings
                                (
                                    prompt_embeds
                                    , negative_prompt_embeds
                                    , pooled_prompt_embeds
                                    , negative_prompt_embeds_pooled
                                ) = get_weighted_text_embeddings_s_cascade(prior, gen_prompt)
                                # Display a message
                                print("Generating prior output...")
                                # Set the prior output
                                prior_output = prior(
                                    prompt_embeds                   = prompt_embeds
                                    , negative_prompt_embeds        = negative_prompt_embeds
                                    , prompt_embeds_pooled          = pooled_prompt_embeds
                                    , negative_prompt_embeds_pooled = negative_prompt_embeds_pooled
                                    , num_inference_steps           = 30
                                    , guidance_scale                = 3.0
                                    , height                        = 1024
                                    , width                         = 1024
                                    , generator                     = generator
                                )
                                del prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_prompt_embeds_pooled
                                # Display a message
                                print("Generating encoder embeddings...")
                                # Get the encoders embeddings    
                                (
                                    prompt_embeds
                                    , negative_prompt_embeds
                                    , pooled_prompt_embeds
                                    , negative_prompt_embeds_pooled
                                ) = get_weighted_text_embeddings_s_cascade(decoder, gen_prompt)  
                                # Display a message
                                print("Generating decoder output...")
                                # Retrieve the image 
                                image = decoder(
                                    prompt_embeds                   = prompt_embeds
                                    , negative_prompt_embeds        = negative_prompt_embeds
                                    , prompt_embeds_pooled          = pooled_prompt_embeds
                                    , negative_prompt_embeds_pooled = negative_prompt_embeds_pooled
                                    , image_embeddings              = prior_output.image_embeddings
                                    , num_inference_steps           = 10
                                    , guidance_scale                = 0
                                    , generator                     = generator, 
                                    output_type = "pil",
                                ).images[0]
                                # Delete the objects
                                del prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_prompt_embeds_pooled
                            # Save image
                            image.save(output_file_name+".png")
                            # Display success message
                            print(f"\nImage generated for : {output_file_name}. Prompt: {gen_prompt}")
            # Free memory inside the loop, fro QWEN
            del image
            # Collect garbage
            gc.collect()
            # Empty cuda cache
            torch.cuda.empty_cache()
            # Collect garbage
            torch.cuda.ipc_collect()
        
        return 1
    except Exception:
        # Display exception
        traceback.print_exc()


def generate_image_qwen_cpp(code_file):
    """
    Generates images using the Qwen-Image model. 
    Doesn't generate if the output file already exists.
    Inference parameters are the same for most models : 
    - Inference steps : 50
    - Guidance scale : 5.0
    - Width and Height : default
    We use do not use textual inversion from SD embed since the model already supports longer prompts. 
    """

    # Set the json outputs directory (.json)
    json_dir = Path("../../outputs/prompts")
    # Get the list of json files
    json_files = [p for p in json_dir.glob("*") if p.is_file()]
    json_files.sort()
    # Set the image output directory
    output_dir_name = "../../data/images_refactored/qwen_image"
    output_dir = Path(output_dir_name)
    # Get the list of files to generate
    output_files = [p._str for p in output_dir.glob("*") if p.is_file()]
    output_files.sort()

    try: 
        # Import relevant packages
        from stable_diffusion_cpp import StableDiffusion
        # Retrieve the mounted Comfy UI path
        comfy_path = os.environ["MOUNTED_COMFY_UI_PATH"]
        # Setup the model
        model = StableDiffusion(
            llm_path=f"{comfy_path}/models/text_encoders/qwen_2.5_vl_7b.safetensors", 
            diffusion_model_path=f"{comfy_path}/models/diffusion_models/qwen_image_bf16.safetensors", 
            vae_path=f"{comfy_path}/models/vae/qwen_image_vae.safetensors",
            lora_model_dir=f"{comfy_path}/models/loras/",
            offload_params_to_cpu=True,
            flash_attn=True, 
            diffusion_flash_attn=True,          
            diffusion_conv_direct=True,         
            vae_conv_direct=True,  
            n_threads=32,     
            qwen_image_zero_cond_t=True,                  
            rng_type="cuda",
            sampler_rng_type="cuda",
            flow_shift=3,
        )
        # Set the elements 
        model_name = "qwen_image"
        # Display a message
        print("\nGenerating images with Qwen-Image.\n")
        # Loop through the files
        for i,json_file in enumerate(json_files):
            print(json_file)
            # Extract the skill code
            skill_code, robust = extract_code_and_robust(code_file,os.path.basename(json_file))
            # Initialize the collection
            elements = dict()
            # Open the file
            with open(json_file, "rb") as file:
                # Load the elements into a dict
                elements = json.load(file)
                # Loop through the prompts
                for prompt in elements["prompts"]:
                    # Get the prompt infos
                    prompt_number, prompt_level, synthetic_prompts = extract_prompt_info(prompt)
                    # Loop through the synthetic prompts
                    for j,gen_prompt in enumerate(synthetic_prompts):
                        # Set the output file name
                        output_file_name = set_output_name(model_name=model_name, 
                                                           skill_code=skill_code, 
                                                           prompt_level=prompt_level,
                                                           prompt_number=prompt_number,
                                                           synthetic_prompt_number=j,
                                                           output_folder_name=output_dir_name, 
                                                           robust=robust)
                        # Check if the file doesn't exist
                        if output_file_name+".png" not in output_files:
                            # Display the output file name
                            print(f"\nOutput file name: {output_file_name}")
                            # Run inference ; generate the image                         
                            image = model.generate_image(
                                prompt=gen_prompt,
                                cfg_scale=4.0,
                                sample_method="euler_a", 
                                seed=SEED, 
                                sample_steps=8,
                                width=1664,
                                height=928
                            )[0]
                            # Save image
                            image.save(output_file_name+".png")
                            # Display success message
                            print(f"\nImage generated for : {output_file_name}. Prompt: {gen_prompt}")
            # Free memory inside the loop, for QWEN
            del image
            # Collect garbage
            gc.collect()
            # Empty cuda cache
            torch.cuda.empty_cache()
            # Collect garbage
            torch.cuda.ipc_collect()
        
        return 1
    except Exception:
        # Display exception
        traceback.print_exc()


def generate_image_dalle(code_file):
    """
    Generates images using the dall-e API. 
    Doesn't generate if the output file already exists.
    - Width and Height : 1024x1024
    """

    # Set the json outputs directory (.json)
    json_dir = Path("../../outputs/prompts")
    # Get the list of json files
    json_files = [p for p in json_dir.glob("*") if p.is_file()]
    json_files.sort()
    # Set the image output directory
    output_dir_name = "../../data/images_refactored/dalle"
    output_dir = Path(output_dir_name)
    # Get the list of files to generate
    output_files = [p._str for p in output_dir.glob("*") if p.is_file()]
    output_files.sort()

    # Import relevant packages
    import requests
    import time
    # Set the API parameters
    azure_endpoint = os.environ["DALLE_ENDPOINT"]
    api_key = os.environ["DALLE_KEY"]
    # Set the elements 
    model_name = "dalle"
    # Display a message
    print("\nGenerating images with dalle.\n")
    # Loop through the files
    for i,json_file in enumerate(json_files):
        print(json_file)
        # Extract the skill code
        skill_code, robust = extract_code_and_robust(code_file,os.path.basename(json_file))
        # Initialize the collection
        elements = dict()
        # Open the file
        with open(json_file, "rb") as file:
            # Load the elements into a dict
            elements = json.load(file)
            # Loop through the prompts
            for prompt in elements["prompts"]:
                # Get the prompt infos
                prompt_number, prompt_level, synthetic_prompts = extract_prompt_info(prompt)
                # Loop through the synthetic prompts
                for j,gen_prompt in enumerate(synthetic_prompts):
                    # Set the output file name
                    output_file_name = set_output_name(model_name=model_name, 
                                                           skill_code=skill_code, 
                                                           prompt_level=prompt_level,
                                                           prompt_number=prompt_number,
                                                           synthetic_prompt_number=j,
                                                           output_folder_name=output_dir_name, 
                                                           robust=robust)
                    # Check if the file doesn't exist
                    if output_file_name+".png" not in output_files:
                        # Display the output file name
                        print(f"\nOutput file name: {output_file_name}")
                        # Set the headers
                        headers = {
                            "Content-Type": "application/json", 
                            "Authorization": api_key
                        }
                        # Set the payload
                        payload = { 
                            "model": "dall-e-3", 
                            "prompt": gen_prompt, 
                            "size": "1024x1024" 
                        }
                        try:
                            # Run inference ; generate the image                         
                            response = requests.post(azure_endpoint, 
                                headers=headers, 
                                json=payload
                                ) 
                            # Get response data
                            data = response.json()
                            # Retrieve the generated image
                            image_url = data["data"][0]["url"]
                            generated_image = requests.get(image_url).content
                            # Write the image 
                            with open(output_file_name+".png", "wb") as file:
                                file.write(generated_image)
                            # Display success message
                            print(f"\nImage generated for : {output_file_name}. Prompt: {gen_prompt}")
                            # Waiting 15 seconds (request quota)
                            time.sleep(15.0)        
                        except Exception:
                            # Display a message
                            print(f"Error for image {output_file_name}")
                            # Display exception
                            traceback.print_exc()

def generate_image_flux_cpp(code_file):
    """
    Generates images using the FLUX model. 
    Doesn't generate if the output file already exists.
    Inference parameters are the same for most models : 
    - Inference steps : 50
    - Guidance scale : 1.0, since it's recommended for flux models.
    - Width and Height : 1024
    """

    # Set the json outputs directory (.json)
    json_dir = Path("../../outputs/prompts")
    # Get the list of json files
    json_files = [p for p in json_dir.glob("*") if p.is_file()]
    json_files.sort()
    # Set the image output directory
    output_dir_name = "../../data/images_refactored/flux"
    output_dir = Path(output_dir_name)
    # Get the list of files to generate
    output_files = [p._str for p in output_dir.glob("*") if p.is_file()]
    output_files.sort()

    try: 
        # Import relevant packages
        from stable_diffusion_cpp import StableDiffusion
        # Setup the model
        model = StableDiffusion(
            diffusion_model_path="../../../ComfyUI/models/diffusion_models/flux1-dev-q8_0.gguf", 
            llm_path="../../../ComfyUI/models/text_encoders/Qwen2.5-VL-7B-Instruct.Q8_0.gguf", 
            t5xxl_path="../../../ComfyUI/models/text_encoders/t5xxl_fp16.safetensors",
            vae_path="../../../ComfyUI/models/vae/ae.safetensors",
            #keep_clip_on_cpu=True,
            vae_decode_only=True,
            flash_attn=True, 
            diffusion_flash_attn=True,          
            diffusion_conv_direct=True,         
            vae_conv_direct=True,  
            n_threads=32,     
            rng_type="cuda",
            sampler_rng_type="cuda",
            flow_shift=3,
        )
        # Set the elements 
        model_name = "flux"
        # Display a message
        print("\nGenerating images with FLUX.\n")
        # Loop through the files
        for i,json_file in enumerate(json_files):
            print(json_file)
            # Extract the skill code
            skill_code, robust = extract_code_and_robust(code_file,os.path.basename(json_file))
            # Initialize the collection
            elements = dict()
            # Open the file
            with open(json_file, "rb") as file:
                # Load the elements into a dict
                elements = json.load(file)
                # Loop through the prompts
                for prompt in elements["prompts"]:
                    # Get the prompt infos
                    prompt_number, prompt_level, synthetic_prompts = extract_prompt_info(prompt)
                    # Loop through the synthetic prompts
                    for j,gen_prompt in enumerate(synthetic_prompts):
                        # Set the output file name
                        output_file_name = set_output_name(model_name=model_name, 
                                                           skill_code=skill_code, 
                                                           prompt_level=prompt_level,
                                                           prompt_number=prompt_number,
                                                           synthetic_prompt_number=j,
                                                           output_folder_name=output_dir_name, 
                                                           robust=robust)
                        # Check if the file doesn't exist
                        if output_file_name+".png" not in output_files:
                            # Display the output file name
                            print(f"\nOutput file name: {output_file_name}")
                            # Run inference ; generate the image                         
                            image = model.generate_image(
                                prompt=gen_prompt,
                                sample_steps=50,
                                width=1024,
                                height=1024,
                                cfg_scale=1.0, # Using this because it's recommended for FLUX
                                sample_method="euler", # Same for this one
                                seed=SEED, 
                            )[0]
                            # Save image
                            image.save(output_file_name+".png")
                            # Display success message
                            print(f"\nImage generated for : {output_file_name}. Prompt: {gen_prompt}")
            # Free memory inside the loop, fro QWEN
            del image
            # Collect garbage
            gc.collect()
            # Empty cuda cache
            torch.cuda.empty_cache()
            # Collect garbage
            torch.cuda.ipc_collect()
        
        return 1
    except Exception:
        # Display exception
        traceback.print_exc()

def generate_image_gpt(code_file):
    """
    Generates images using the gpt-image API. 
    Doesn't generate if the output file already exists.
    - Width and Height : 1024x1024
    """

    # Set the json outputs directory (.json)
    json_dir = Path("../../outputs/prompts")
    # Get the list of json files
    json_files = [p for p in json_dir.glob("*") if p.is_file()]
    json_files.sort()
    # Set the image output directory
    output_dir_name = "../../data/images_refactored/gpt-image-1.5"
    output_dir = Path(output_dir_name)
    # Get the list of files to generate
    output_files = [p._str for p in output_dir.glob("*") if p.is_file()]
    output_files.sort()

    # Import relevant packages
    import requests
    import time
    # Set the API parameters
    azure_endpoint = os.environ["GPT_IMAGE_ENDPOINT"]
    api_key = os.environ["GPT_IMAGE_KEY"]
    # Set the elements 
    model_name = "gpt-image"
    # Display a message
    print("\nGenerating images with gpt-image-1.5.\n")
    # Loop through the files
    for i,json_file in enumerate(json_files):
        print(json_file)
        # Extract the skill code
        skill_code, robust = extract_code_and_robust(code_file,os.path.basename(json_file))
        # Exit if skill code <> 0 for now
        #if skill_code!=0:
        #    return 0
        # Initialize the collection
        elements = dict()
        # Open the file
        with open(json_file, "rb") as file:
            # Load the elements into a dict
            elements = json.load(file)
            # Loop through the prompts
            for prompt in elements["prompts"]:
                # Get the prompt infos
                prompt_number, prompt_level, synthetic_prompts = extract_prompt_info(prompt)
                # Loop through the synthetic prompts
                for j,gen_prompt in enumerate(synthetic_prompts):
                    # Set the output file name
                    output_file_name = set_output_name(model_name=model_name, 
                                                           skill_code=skill_code, 
                                                           prompt_level=prompt_level,
                                                           prompt_number=prompt_number,
                                                           synthetic_prompt_number=j,
                                                           output_folder_name=output_dir_name, 
                                                           robust=robust)
                    # Check if the file doesn't exist
                    if output_file_name+".png" not in output_files:
                        # Display the output file name
                        print(f"\nOutput file name: {output_file_name}")
                        # Set the headers
                        headers = {
                            "Content-Type": "application/json", 
                            "Authorization": api_key
                        }
                        # Set the payload
                        payload = { 
                            "model": "gpt-image-1.5", 
                            "prompt": gen_prompt, 
                            "size": "1024x1024", 
                            "quality": "medium", 
                            "output_format": "png"
                        }
                        try:
                            # Run inference ; generate the image                         
                            response = requests.post(azure_endpoint, 
                                headers=headers, 
                                json=payload
                                ) 
                            # Get response data
                            data = response.json()
                            # Retrieve the generated image
                            image_base64 = data["data"][0]["b64_json"]
                            generated_image = base64.b64decode(image_base64)
                            # Write the image 
                            with open(f"{output_file_name}.png", "wb") as file:
                                file.write(generated_image)
                            # Display success message
                            print(f"\nImage generated for : {output_file_name}. Prompt: {gen_prompt}")
                            # Waiting 15 seconds (request quota)
                            time.sleep(15.0)        
                        except Exception:
                            # Display a message
                            print(f"Error for image {output_file_name}")
                            # Display exception
                            traceback.print_exc()


def initialize_parser():
    """
    Initializes the argument parser. 
    """

    # Initializing the parser 
    parser = argparse.ArgumentParser(description="Test script for generating images. Default or (inferred) best paramaters for each model are used for each generator.")

    # Add arguments 
    parser.add_argument("--gpu", choices=["0", "1", "2"], default="0", help="Preferred GPU number.")
    parser.add_argument("--model", choices=["all", 
                                            "firelfy", 
                                            "dalle", 
                                            "kandinsky", 
                                            "runway", 
                                            "stable_cascade", 
                                            "stable_diffusion", 
                                            "gpt-image",
                                            "z_image", 
                                            "qwen", 
                                            "flux"], default="qwen", help="Generator to use.")
    parser.add_argument("--seed", type=int, default=42, help="The generator seed.")

    return parser


def main():
    """
    Main script
    """
    # Free memory
    gc.collect()

    if torch.cuda.is_available() and torch.version.cuda is not None:
        # Empty cuda cache
        torch.cuda.empty_cache()
        # Collect garbage
        torch.cuda.ipc_collect()
    
    # Parse arguments 
    parser = initialize_parser()
    # Set the GPU
    GPU = parser.parse_args().gpu
    # Set the model 
    MODEL = parser.parse_args().model
    # Set the generator seed
    SEED = parser.parse_args().seed
    # Load the code file
    code_file = pd.read_csv("../../skills_code.csv", encoding="utf-8")
    # Load the access token
    access_token = load_environment()

    if MODEL=="animagine":
        # Generate images with animagine
        generate_image_animagine(code_file,GPU)
    elif MODEL=="stable_diffusion":
        # Generate images with stable diffusion
        generate_image_stable_diffusion(code_file,GPU)
    elif MODEL=="z_image":
        # Generat images with z-image turbo
        generate_image_z_image_turbo(code_file,GPU)
    elif MODEL=="qwen":
        # Generate images with QWEN
        generate_image_qwen_cpp(code_file)
    elif MODEL=="flux":
        # Generate images with FLUX
        generate_image_flux_cpp(code_file)
    elif MODEL=="dalle":
        # Generate images with Dall-E 3
        generate_image_dalle(code_file)
    elif MODEL=="stable_cascade":
        # Stable cascade optimization
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        # Generate images with Stable Cascade
        generate_image_stable_cascade(code_file,access_token)
    elif MODEL=="gpt-image":
        # Generate images with gpt-image-1.5
        generate_image_gpt(code_file)



if __name__=="__main__":
    # Load the main
    main()