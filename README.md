# **Flux.2-4B-Encoder-Comparator**

Flux.2-4B-Encoder-Comparator is an experimental, dual-pipeline application designed to perform direct, side-by-side visual evaluations of the FLUX.2-klein-4B model using two different Variational Autoencoders (VAEs). By executing the standard decoder and the newly introduced small-decoder concurrently, the tool allows developers and researchers to instantly observe differences in artifacting, detail retention, and color processing. The application features a deeply customized, Ubuntu-inspired web interface built with pure HTML, CSS, and JavaScript served via FastAPI. It supports both text-to-image generation and image-to-image editing, providing an interactive slider for real-time comparative analysis and batch ZIP downloading for offline review.

<img width="1538" height="810" alt="Screenshot 2026-04-13 at 08-48-36 Flux 2 4B Encoder Comparator - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/0232cc6c-6696-459c-8354-255109c1079c" />

### **Key Features**

* **Concurrent Dual Inference:** Simultaneously runs the standard FLUX.2-klein-4B pipeline and the FLUX.2-klein-4B pipeline equipped with the `FLUX.2-small-decoder` VAE using Python's ThreadPoolExecutor, maximizing GPU utility and minimizing wait times.
* **Interactive Comparison Slider:** Features a custom-built image slider stage that overlays the standard output and the small-decoder output, allowing users to drag back and forth to meticulously inspect pixel-level differences.
* **Custom Headless UI:** Abandons standard Gradio blocks in favor of a highly responsive, custom frontend design featuring a dark aubergine and orange theme. It includes a drag-and-drop uploader, dynamic aspect ratio calculation, and an integrated execution log.
* **Advanced Generation Controls:** Provides an expandable settings panel to fine-tune Generation Seed, Inference Steps, Width, Height, and Guidance Scale.
* **Batch Export:** Built-in functionality allows users to package and download both generated images into a single ZIP file with one click.

### **Repository Structure**

```text
├── examples/
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   ├── 4.jpg
│   ├── I1.jpg
│   └── I2.jpg
├── app-truncated.py
├── app.py
├── LICENSE.txt
├── pre-requirements.txt
├── README.md
└── requirements.txt
```

### **Installation and Requirements**

To run the Flux.2-4B-Encoder-Comparator locally, configure a Python environment with the following dependencies. Ensure you have a compatible CUDA-enabled GPU with sufficient VRAM to load both pipelines simultaneously.

**1. Install Pre-requirements**
Update pip to the required version before installing the main dependencies:
```bash
pip install pip>=26.0.0
```

**2. Install Core Requirements**
Install the necessary diffusion, machine learning, and web server libraries. Place these in a `requirements.txt` file and execute `pip install -r requirements.txt`.

```text
git+https://github.com/huggingface/diffusers.git
transformers==4.57.6
huggingface_hub
sentencepiece
bitsandbytes
torchvision
accelerate
torchao
spaces
hf_xet
gradio
numpy
torch
peft
av
```

### **Usage**

After setting up your environment and ensuring your dependencies are installed, launch the application by running the main Python script:

```bash
python app.py
```

The script will initialize both FLUX pipelines and the respective VAEs into memory. Once ready, it will expose a local web server (typically at `http://127.0.0.1:7860/`). Open this address in your browser to access the comparator interface. Upload reference images if desired, enter your generation or editing prompt, and click "Run Comparison" to view the side-by-side results.

### **License and Source**

* **License:** Apache License - Version 2.0 (Available at [LICENSE.txt](https://github.com/PRITHIVSAKTHIUR/Flux.2-4B-Encoder-Comparator/blob/main/LICENSE.txt))
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/Flux.2-4B-Encoder-Comparator.git](https://github.com/PRITHIVSAKTHIUR/Flux.2-4B-Encoder-Comparator.git)
