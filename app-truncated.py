import os
import gc
import gradio as gr
import numpy as np
import random
import spaces
import torch
from diffusers import Flux2KleinPipeline, AutoencoderKLFlux2
from PIL import Image
from pathlib import Path
import concurrent.futures
import threading
from typing import Iterable

from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

colors.orange_red = colors.Color(
    name="orange_red",
    c50="#FFF0E5",
    c100="#FFE0CC",
    c200="#FFC299",
    c300="#FFA366",
    c400="#FF8533",
    c500="#FF4500",
    c600="#E63E00",
    c700="#CC3700",
    c800="#B33000",
    c900="#992900",
    c950="#802200",
)

class OrangeRedTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.orange_red,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

orange_red_theme = OrangeRedTheme()

dtype  = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

MAX_SEED       = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024
EXAMPLES_DIR   = Path("examples")

print("Loading 4B Distilled model (Standard VAE)...")
pipe_standard = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=dtype,
)
pipe_standard.enable_model_cpu_offload()

print("Loading Small Decoder VAE...")
vae_small = AutoencoderKLFlux2.from_pretrained(
    "black-forest-labs/FLUX.2-small-decoder",
    torch_dtype=dtype,
)

print("Loading 4B Distilled model (Small Decoder VAE)...")
pipe_small_decoder = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    vae=vae_small,
    torch_dtype=dtype,
)
pipe_small_decoder.enable_model_cpu_offload()

pipe_lock_standard = threading.Lock()
pipe_lock_small    = threading.Lock()

def calc_dimensions(pil_img: Image.Image):
    """
    Given a PIL image return (width, height) snapped to multiples of 8,
    fitting within 1024 px on the long side, min 256 px on each side.
    Uses round() so we match the reference app exactly.
    """
    iw, ih = pil_img.size
    aspect = iw / ih

    if aspect >= 1:          # landscape / square
        new_width  = 1024
        new_height = int(round(1024 / aspect))
    else:                    # portrait
        new_height = 1024
        new_width  = int(round(1024 * aspect))

    # snap to 8-pixel grid with round(), clamp to [256, 1024]
    new_width  = max(256, min(1024, round(new_width  / 8) * 8))
    new_height = max(256, min(1024, round(new_height / 8) * 8))
    return new_width, new_height


def update_dimensions_from_image(image_list):
    """
    Called by the gallery .upload() event.
    Returns updated slider values for width and height.
    """
    if not image_list:
        return 1024, 1024

    # gallery items arrive as PIL images when type="pil"
    item = image_list[0]
    img  = item[0] if isinstance(item, tuple) else item

    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    elif not isinstance(img, Image.Image):
        return 1024, 1024

    return calc_dimensions(img)

def parse_and_resize_images(input_images, width: int, height: int):
    """
    Parse the gallery input and resize every frame to (width, height).
    Returns a list[PIL.Image] or None.
    """
    if input_images is None:
        return None

    raw_list = []

    if isinstance(input_images, str):
        if os.path.exists(input_images):
            raw_list = [Image.open(input_images).convert("RGB")]
    elif isinstance(input_images, Image.Image):
        raw_list = [input_images.convert("RGB")]
    elif isinstance(input_images, list):
        for item in input_images:
            try:
                src = item[0] if isinstance(item, tuple) else item
                if isinstance(src, str):
                    raw_list.append(Image.open(src).convert("RGB"))
                elif isinstance(src, Image.Image):
                    raw_list.append(src.convert("RGB"))
                elif hasattr(src, "name"):
                    raw_list.append(Image.open(src.name).convert("RGB"))
            except Exception as e:
                print(f"Skipping invalid image: {e}")

    if not raw_list:
        return None

    resized = [
        img.resize((width, height), Image.LANCZOS)
        for img in raw_list
    ]
    return resized

def run_pipeline(pipe, lock, kwargs, seed):
    with lock:
        gen    = torch.Generator(device="cpu").manual_seed(seed)
        result = pipe(**kwargs, generator=gen).images[0]
    return result

@spaces.GPU(duration=120)
def infer(
    prompt,
    input_images=None,
    seed=42,
    randomize_seed=False,
    width=1024,
    height=1024,
    num_inference_steps=4,
    guidance_scale=1.0,
    progress=gr.Progress(track_tqdm=True),
):
    gc.collect()
    torch.cuda.empty_cache()

    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    # ── width / height: derive from the first uploaded image if present ──
    image_list = None
    if input_images:
        # Re-derive dimensions from the actual first image so they are
        # always consistent with what the pipeline will receive.
        item = (
            input_images[0][0]
            if isinstance(input_images[0], tuple)
            else input_images[0]
        )
        if isinstance(item, str):
            first_pil = Image.open(item).convert("RGB")
        elif isinstance(item, Image.Image):
            first_pil = item.convert("RGB")
        else:
            first_pil = None

        if first_pil is not None:
            width, height = calc_dimensions(first_pil)

        # parse + resize all images to the final (width, height)
        image_list = parse_and_resize_images(input_images, width, height)

    # ensure dims are multiples of 8 even for text-only runs
    width  = max(256, min(MAX_IMAGE_SIZE, round(int(width)  / 8) * 8))
    height = max(256, min(MAX_IMAGE_SIZE, round(int(height) / 8) * 8))

    shared_kwargs = dict(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    if image_list is not None:
        shared_kwargs["image"] = image_list

    progress(0.30, desc="Launching both pipelines simultaneously...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_std   = executor.submit(
            run_pipeline, pipe_standard,     pipe_lock_standard, shared_kwargs, seed
        )
        future_small = executor.submit(
            run_pipeline, pipe_small_decoder, pipe_lock_small,    shared_kwargs, seed
        )
        concurrent.futures.wait(
            [future_std, future_small],
            return_when=concurrent.futures.ALL_COMPLETED,
        )

    progress(0.80, desc="✅ Both pipelines done!")

    out_standard = future_std.result()
    out_small    = future_small.result()

    gc.collect()
    torch.cuda.empty_cache()

    return out_standard, out_small, seed


@spaces.GPU(duration=120)
def infer_example(prompt):
    out_std, out_small, seed_used = infer(
        prompt=prompt,
        input_images=None,
        seed=0,
        randomize_seed=True,
        width=1024,
        height=1024,
        num_inference_steps=4,
        guidance_scale=1.0,
    )
    return out_std, out_small, seed_used


def get_example_items():
    example_prompts = {
        "1.jpg": "Change the weather to stormy.",
        "2.jpg": "Transform the scene into a snowy winter day while preserving the original subject identity, framing, and composition.",
        "3.jpg": "Relight the image with soft golden sunset lighting while keeping all structures and subject details consistent.",
        "4.jpg": "Make the texture high-resolution.",
    }
    items = []
    if EXAMPLES_DIR.exists():
        for name in sorted(os.listdir(EXAMPLES_DIR)):
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                items.append({
                    "file":   name,
                    "path":   str(EXAMPLES_DIR / name),
                    "prompt": example_prompts.get(
                        name, "Edit this image while preserving composition."
                    ),
                })
    return items

EXAMPLE_ITEMS = get_example_items()

css = """
#col-container {
    margin: 0 auto;
    max-width: 1100px;
}
#main-title h1 {
    font-size: 2.4em !important;
}
.vae-badge {
    font-weight: 700;
    font-size: 0.95em;
    text-align: center;
    padding: 4px 16px;
    border-radius: 20px;
    display: block;
    margin-bottom: 6px;
}
"""

with gr.Blocks() as demo:

    with gr.Column(elem_id="col-container"):

        gr.Markdown(
            "# **Flux.2-4B-Decoder-Comparator**",
            elem_id="main-title",
        )
        gr.Markdown(
            "Compare **FLUX.2-klein-4B** side-by-side with "
            "[small decoder](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder)."
        )

        with gr.Row(equal_height=True):

            with gr.Column():
                input_images = gr.Gallery(
                    label="Input Images",
                    type="pil",
                    columns=2,
                    rows=1,
                    height=300,
                    allow_preview=True,
                )

                prompt = gr.Text(
                    label="Prompt",
                    max_lines=1,
                    show_label=True,
                    placeholder="e.g., A black cat holding a sign that says hello world...",
                )

                run_button = gr.Button("Run Comparison", variant="primary")

            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        result_standard = gr.Image(
                            label="Standard Decoder",
                            show_label=True,
                            interactive=False,
                            format="png",
                            height=250,
                        )
                    with gr.Column():
                        result_small = gr.Image(
                            label="Small Decoder",
                            show_label=True,
                            interactive=False,
                            format="png",
                            height=250,
                        )

                seed_output = gr.Number(label="Seed Used", precision=0, visible=False)

                with gr.Accordion("Advanced Settings", open=False, visible=False):
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=0,
                    )
                    randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)

                    with gr.Row():
                        width = gr.Slider(
                            label="Width",
                            minimum=256,
                            maximum=MAX_IMAGE_SIZE,
                            step=8,
                            value=1024,
                        )
                        height_slider = gr.Slider(
                            label="Height",
                            minimum=256,
                            maximum=MAX_IMAGE_SIZE,
                            step=8,
                            value=1024,
                        )

                    with gr.Row():
                        num_inference_steps = gr.Slider(
                            label="Inference Steps",
                            minimum=1,
                            maximum=20,
                            step=1,
                            value=4,
                        )
                        guidance_scale = gr.Slider(
                            label="Guidance Scale",
                            minimum=0.0,
                            maximum=10.0,
                            step=0.1,
                            value=1.0,
                        )

        gr.Examples(
            examples=[
                [["examples/I1.jpg", "examples/I2.jpg"], "Make her wear these glasses in Image 2."],
                [["examples/1.jpg"], "Change the weather to stormy."],
                [["examples/2.jpg"], "Transform the scene into a snowy winter day while preserving the original subject identity, framing, and composition."],
                [["examples/3.jpg"], "Relight the image with soft golden sunset lighting while keeping all structures and subject details consistent."],
                [["examples/4.jpg"], "Make the texture high-resolution."],
            ],
            inputs=[input_images, prompt],
            outputs=[result_standard, result_small, seed_output],
            fn=infer_example,
            cache_examples=False,
            label="Examples",
        )

        gr.Markdown(
            "[*](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) "
            "Experimental Space — FLUX.2 [klein] 4B VAE Decoder Comparison."
        )

    input_images.upload(
        fn=update_dimensions_from_image,
        inputs=[input_images],
        outputs=[width, height_slider],
    )

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[
            prompt,
            input_images,
            seed,
            randomize_seed,
            width,
            height_slider,
            num_inference_steps,
            guidance_scale,
        ],
        outputs=[result_standard, result_small, seed_output],
    )

if __name__ == "__main__":
    demo.queue(max_size=20).launch(
        theme=orange_red_theme, css=css,
        mcp_server=True,
        ssr_mode=False,
        show_error=True,
    )
