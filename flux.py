from mflux.models.flux2.variants import Flux2Klein

flux = Flux2Klein(
    model_path="RunPod/FLUX.2-klein-4B-mflux-4bit",
)

image = flux.generate_image(
    prompt="A cute robot standing in a field of flowers, digital art",
    width=768,
    height=1024,
    num_inference_steps=4,
    seed=42,
)

image.save("output.png")
