


import os
import json
import numpy as np
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from scipy.linalg import sqrtm
from tqdm.auto import tqdm
from torchvision.models import inception_v3
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
from aesthetics_scorer import preprocess, load_model
import lpips
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity as structural_similarity_index_measure,
    normalized_mutual_information,
)

class ImageMetricsComputer:
    def __init__(self, device='cuda'):
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        print("Loading LPIPS model...")
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        
        print("Loading CLIP and aesthetic models...")
        self._load_aesthetic_models()
        
    def _load_aesthetic_models(self):
        local_path = "./model_weights/clip_laion"
        if not os.path.exists(local_path):
            os.makedirs(local_path, exist_ok=True)
            model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=local_path)
            model.save_pretrained(local_path)
        else:
            model = CLIPModel.from_pretrained(local_path)
        
        self.vision_model = model.vision_model.to(self.device)
        del model
        
        self.clip_processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", 
                                                     cache_dir="./model_weights/clip_laion_processor")
        self.rating_model = load_model("aesthetics_scorer_rating_openclip_vit_h_14").to(self.device)
        self.artifacts_model = load_model("aesthetics_scorer_artifacts_openclip_vit_h_14").to(self.device)

    def compute_lpips(self, image1, image2):
        try:
            img1 = self.transform(image1).unsqueeze(0).to(self.device)
            img2 = self.transform(image2).unsqueeze(0).to(self.device)
            return self.lpips_model(img1, img2).item()
        except Exception as e:
            raise Exception(f"Failed to compute LPIPS: {str(e)}")

    def compute_aesthetics_and_artifacts_scores(self, images):
        try:
            inputs = self.clip_processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                vision_output = self.vision_model(**inputs)
            pooled_output = vision_output.pooler_output
            embedding = preprocess(pooled_output)
            with torch.no_grad():
                rating = self.rating_model(embedding)
                artifact = self.artifacts_model(embedding)
            return (
                rating.detach().cpu().numpy().flatten().tolist(),
                artifact.detach().cpu().numpy().flatten().tolist(),
            )
        except Exception as e:
            raise Exception(f"Failed to compute aesthetics scores: {str(e)}")

def compute_metrics(computer, image1, image2):
    """Compute all metrics for a pair of images"""
    try:
        # Convert to numpy for traditional metrics
        image1_np = np.array(image1)
        image2_np = np.array(image2)
        
        psnr = float(peak_signal_noise_ratio(image1_np, image2_np))
        ssim = float(structural_similarity_index_measure(image1_np, image2_np, channel_axis=2))
        nmi = float(normalized_mutual_information(image1_np, image2_np))
        lpips_score = computer.compute_lpips(image1, image2)
        
        x1, y1 = computer.compute_aesthetics_and_artifacts_scores(image1)
        x2, y2 = computer.compute_aesthetics_and_artifacts_scores(image2)
        
        delta_aesthetic = x2[0] - x1[0]
        delta_artifact = y2[0] - y1[0]
        
        return {
            "psnr": psnr,
            "ssim": ssim,
            "nmi": nmi,
            "lpips": lpips_score,
            "aesthetic_delta": delta_aesthetic,
            "artifact_delta": delta_artifact
        }
    except Exception as e:
        raise Exception(f"Failed to compute metrics: {str(e)}")

def compare_image_folders(fake_dir, inv_fake_dir, output_json, device='cuda'):
    """Compare images in two folders and compute various metrics"""
    if not os.path.exists(fake_dir) or not os.path.exists(inv_fake_dir):
        print(f"Directory check failed:")
        print(f"fake_dir exists: {os.path.exists(fake_dir)}")
        print(f"inv_fake_dir exists: {os.path.exists(inv_fake_dir)}")
        return {}

    # Initialize metrics computer once
    computer = ImageMetricsComputer(device=device)
    
    fake_files = sorted(os.listdir(fake_dir))
    results = {}
    errors = {}
    
    for filename in tqdm(fake_files, desc="Processing images"):
        fake_path = os.path.join(fake_dir, filename)
        inv_fake_path = os.path.join(inv_fake_dir, filename)
        
        # Skip if either image is missing
        if not os.path.exists(fake_path):
            errors[filename] = "Source image not found"
            continue
        if not os.path.exists(inv_fake_path):
            errors[filename] = "Target image not found"
            continue
            
        try:
            # Load and resize both images to 768x768
            fake_img = Image.open(fake_path).convert('RGB')
            fake_img = fake_img.resize((768, 768), Image.Resampling.LANCZOS)
            
            inv_fake_img = Image.open(inv_fake_path).convert('RGB')
            inv_fake_img = inv_fake_img.resize((768, 768), Image.Resampling.LANCZOS)
            
            # Compute metrics
            results[filename] = compute_metrics(computer, fake_img, inv_fake_img)
            
            fake_img.close()
            inv_fake_img.close()
            
        except Exception as e:
            errors[filename] = str(e)
            print(f"Error processing {filename}: {str(e)}")
            continue
        
        # Save results periodically
        if len(results) % 100 == 0:
            save_results(output_json, results, errors)
    
    # Save final results
    save_results(output_json, results, errors)
    
    return results, errors

def save_results(output_json, results, errors):
    """Save results and errors to JSON file"""
    try:
        with open(output_json, 'w') as f:
            json.dump({
                "results": results,
                "errors": errors,
                "total_processed": len(results),
                "total_errors": len(errors)
            }, f, indent=2)
    except Exception as e:
        print(f"Error saving results: {str(e)}")

if __name__ == "__main__":
    # Configuration
    base_dir = "/shared/shashmi/inversion/inversion/SD3.5"
    fake_dir = os.path.join(base_dir, "real")
    inv_fake_dir = os.path.join(base_dir, "inv_real")
    output_json = os.path.join("/ephemeral/shashmi/rf-inversion-sd3/evaluation/", "sd3.5_unconditioned_comparison_metrics_real_inv_real.json")
    
    # Print directory information
    print(f"Starting image comparison:")
    print(f"Source directory: {fake_dir}")
    print(f"Target directory: {inv_fake_dir}")
    print(f"Output file: {output_json}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Run comparison
    results, errors = compare_image_folders(fake_dir, inv_fake_dir, output_json, device)
    
    # Print summary
    print(f"\nProcessing complete:")
    print(f"Total images processed successfully: {len(results)}")
    print(f"Total errors: {len(errors)}")
    if errors:
        print("\nFirst few errors:")
        for filename, error in list(errors.items())[:5]:
            print(f"{filename}: {error}")