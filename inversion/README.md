# The Liar's Dividend: Measuring the Risk of Image Generators on Plausible Deniability

This repository contains implementation code for image manipulation using various Stable Diffusion models (SD2, SD3, SD3.5) with RF inversion techniques. The project focuses on exploring plausible deniability in the context of image generation and manipulation.

## Features

- Support for multiple Stable Diffusion models (SD2, SD3, SD3.5)
- RF inversion implementation
- Conditional and unconditional image generation
- Batch processing capabilities
- Various interpolation strategies

## Requirements

- Python 3.8+
- PyTorch
- diffusers
- transformers
- Pillow
- torchvision

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/The-Liar-s-Divident-Measuring-the-Risk-of-Image-Generators-on-the-Plausible-Deniability.git
cd The-Liar-s-Divident-Measuring-the-Risk-of-Image-Generators-on-the-Plausible-Deniability
```

2. Create and activate a virtual environment:
```bash
python -m venv fakeit
source fakeit/bin/activate  # On Windows, use: fakeit\Scripts\activate
```

3. Install the required packages:
```bash
pip install torch torchvision diffusers transformers Pillow
```

## Usage

The repository contains several scripts for different purposes:

### SD3.5 All-in-One Script
```bash
python sd3.5_conditioned_all_in.py \
    --model_path stabilityai/stable-diffusion-3.5-medium \
    --input_dir path/to/input/images \
    --output_dir path/to/output \
    --eta_base 0.95 \
    --eta_trend constant \
    --start_step 0 \
    --end_step 9
```

### RF Inversion Script
```bash
python sd3_5_rf_inversion.py \
    --model_path stabilityai/stable-diffusion-3.5-medium \
    --image_path path/to/image.jpg \
    --output_dir path/to/output \
    --gamma 0.5 \
    --prompt "your prompt here"
```

## Parameters

- `--model_path`: Path to the pretrained model
- `--input_dir`: Directory containing input images
- `--output_dir`: Directory to save output images
- `--eta_base`: Base eta value for interpolation (default: 0.95)
- `--eta_trend`: Interpolation trend ['constant', 'linear_increase', 'linear_decrease']
- `--start_step`: Starting step for interpolation
- `--end_step`: Ending step for interpolation
- `--gamma`: Control parameter for RF inversion (0.0-1.0)
- `--guidance_scale`: Guidance scale for stable diffusion
- `--num_steps`: Number of denoising steps
- `--seed`: Random seed for reproducibility
- `--dtype`: Data type for computation ['float16', 'bfloat16', 'float32']

## Directory Structure

```
├── Evaluation_code/
├── sd2_all_in.py
├── sd2_condtioned_all_in.py
├── sd2_rf_inversion.py
├── sd3.5_all_in.py
├── sd3.5_conditioned_all_in.py
├── sd3_5_rf_inversion.py
├── sd3_all_in.py
├── sd3_conditioned_all_in.py
└── sd3_rf_inversion.py
```

## Implementation Details

The implementation uses a novel approach to image manipulation through RF inversion and interpolated denoising. Key components include:

1. **RF Inversion**: Implements reverse diffusion process with flow matching
2. **Interpolated Denoising**: Combines target image velocity with predicted velocity
3. **Eta Scheduling**: Supports various interpolation strategies (constant, linear increase/decrease)



## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.