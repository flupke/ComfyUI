# ruff: noqa: T201
#
import os
import random
import sys
import pathlib
import shutil
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]  # type: ignore


def find_path(name: str, path: str | None = None) -> str | None:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS


class Models:
    def __init__(self):
        dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        self.dual_clip_loader = dualcliploader.load_clip(
            clip_name1="clip_l.safetensors",
            clip_name2="t5xxl_fp16.safetensors",
            type="flux",
            device="default",
        )
        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        self.vae_loader = vaeloader.load_vae(vae_name="ae.safetensors")
        unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        self.unet_loader = unetloader.load_unet(
            unet_name="flux1-fill-dev.safetensors", weight_dtype="bf16"
        )
        loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
        self.lora_loader_model_only = loraloadermodelonly.load_lora_model_only(
            lora_name="iceedit.safetensors",
            strength_model=1,
            model=get_value_at_index(self.unet_loader, 0),
        )


def edit(
    models: Models, input_image: str, output_prefix: str, prompt: str, turbo: bool
):
    cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
    cliptextencode_7 = cliptextencode.encode(
        text="", clip=get_value_at_index(models.dual_clip_loader, 0)
    )

    cliptextencode_114 = cliptextencode.encode(
        text=prompt,
        clip=get_value_at_index(models.dual_clip_loader, 0),
    )

    fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
    fluxguidance_26 = fluxguidance.append(
        guidance=50, conditioning=get_value_at_index(cliptextencode_114, 0)
    )

    loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
    loadimage_411 = loadimage.load_image(image=input_image)

    getimagesizeandcount = NODE_CLASS_MAPPINGS["GetImageSizeAndCount"]()
    getimagesizeandcount_395 = getimagesizeandcount.getsize(
        image=get_value_at_index(loadimage_411, 0)
    )

    solidmask = NODE_CLASS_MAPPINGS["SolidMask"]()
    solidmask_394 = solidmask.solid(
        value=1,
        width=get_value_at_index(getimagesizeandcount_395, 1),
        height=get_value_at_index(getimagesizeandcount_395, 2),
    )

    easy_makeimageforiclora = NODE_CLASS_MAPPINGS["easy makeImageForICLora"]()
    easy_makeimageforiclora_389 = easy_makeimageforiclora.make(
        direction="left-right",
        pixels=0,
        image_1=get_value_at_index(loadimage_411, 0),
        image_2=get_value_at_index(loadimage_411, 0),
        mask_2=get_value_at_index(solidmask_394, 0),
    )

    inpaintmodelconditioning = NODE_CLASS_MAPPINGS["InpaintModelConditioning"]()
    inpaintmodelconditioning_38 = inpaintmodelconditioning.encode(
        noise_mask=True,
        positive=get_value_at_index(fluxguidance_26, 0),
        negative=get_value_at_index(cliptextencode_7, 0),
        vae=get_value_at_index(models.vae_loader, 0),
        pixels=get_value_at_index(easy_makeimageforiclora_389, 0),
        mask=get_value_at_index(easy_makeimageforiclora_389, 1),
    )

    easy_seed = NODE_CLASS_MAPPINGS["easy seed"]()
    easy_seed.doit(seed=random.randint(1, 2**64))

    unloadmodel = NODE_CLASS_MAPPINGS["UnloadModel"]()
    ksampleradvanced = NODE_CLASS_MAPPINGS["KSamplerAdvanced"]()
    vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
    imagecrop = NODE_CLASS_MAPPINGS["ImageCrop"]()
    saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

    if turbo:
        ksampler_positive = get_value_at_index(inpaintmodelconditioning_38, 0)
    else:
        unload_node = unloadmodel.route(
            value=get_value_at_index(inpaintmodelconditioning_38, 0),
            model=get_value_at_index(models.dual_clip_loader, 0),
        )
        ksampler_positive = get_value_at_index(unload_node, 0)

    ksampleradvanced_116 = ksampleradvanced.sample(
        add_noise="enable",
        noise_seed=random.randint(1, 2**64),
        steps=20,
        cfg=1,
        sampler_name="euler",
        scheduler="beta",
        start_at_step=0,
        end_at_step=10,
        return_with_leftover_noise="disable",
        model=get_value_at_index(models.lora_loader_model_only, 0),
        positive=ksampler_positive,
        negative=get_value_at_index(inpaintmodelconditioning_38, 1),
        latent_image=get_value_at_index(inpaintmodelconditioning_38, 2),
    )

    ksampleradvanced_115 = ksampleradvanced.sample(
        add_noise="enable",
        noise_seed=random.randint(1, 2**64),
        steps=20,
        cfg=1,
        sampler_name="euler",
        scheduler="beta",
        start_at_step=10,
        end_at_step=20,
        return_with_leftover_noise="disable",
        model=get_value_at_index(models.lora_loader_model_only, 0),
        positive=ksampler_positive,
        negative=get_value_at_index(inpaintmodelconditioning_38, 1),
        latent_image=get_value_at_index(ksampleradvanced_116, 0),
    )

    vaedecode_8 = vaedecode.decode(
        samples=get_value_at_index(ksampleradvanced_115, 0),
        vae=get_value_at_index(models.vae_loader, 0),
    )

    imagecrop_399 = imagecrop.crop(
        width=512,
        height=512,
        x=512,
        y=0,
        image=get_value_at_index(vaedecode_8, 0),
    )

    saveimage.save_images(
        filename_prefix=output_prefix,
        images=get_value_at_index(imagecrop_399, 0),
    )


def main():
    for file in pathlib.Path("output").glob("*.png"):
        file.unlink()
    source_image = "/home/flupke/Downloads/yuli.png"
    shutil.copy(
        source_image, "/home/flupke/src/ext/ComfyUI/output/edit_output_0_00001_.png"
    )

    import_custom_nodes()
    models = Models()
    num_iterations = 128

    with torch.inference_mode():
        for i in range(1, num_iterations):
            print(f"Iteration {i}/{num_iterations}")
            edit(
                models=models,
                input_image=f"/home/flupke/src/ext/ComfyUI/output/edit_output_{i - 1}_00001_.png",
                output_prefix=f"edit_output_{i}",
                prompt="Make the photo look like a model studio shoot",
                turbo=False,
            )


if __name__ == "__main__":
    main()
