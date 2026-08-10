import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
from transformers import CLIPProcessor, CLIPModel
import cv2
import json
from tqdm import tqdm

def normalize_psnr(psnr_value):
    """Normalize PSNR to 0-1 range"""
    min_psnr = 20
    max_psnr = 50
    return np.clip((psnr_value - min_psnr) / (max_psnr - min_psnr), 0, 1)

def calculate_combined_score(psnr_value, ssim_value, lpips_value, clip_sim):
    """Calculate combined similarity score with weighted metrics"""
    norm_psnr = normalize_psnr(psnr_value)
    norm_ssim = (ssim_value + 1) / 2
    norm_lpips = 1 - lpips_value
    norm_clip = (clip_sim + 1) / 2
    
    weights = {
        'psnr': 0.15,
        'ssim': 0.30,
        'lpips': 0.35,
        'clip': 0.20
    }
    
    combined_score = (
        weights['psnr'] * norm_psnr +
        weights['ssim'] * norm_ssim +
        weights['lpips'] * norm_lpips +
        weights['clip'] * norm_clip
    )
    
    return combined_score

def load_and_resize_image(image_path, target_size=None):
    """Load and convert image to numpy array, optionally resizing it."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    return img

def get_image_dimensions(image_path):
    """Get dimensions of an image."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return img.shape[1], img.shape[0]  # width, height

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images."""
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions for PSNR calculation")
    return psnr(img1, img2)

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images."""
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions for SSIM calculation")
    return ssim(img1, img2, channel_axis=2, data_range=255)

def calculate_lpips(img1_path, img2_path, lpips_model, device):
    """Calculate LPIPS between two images."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    try:
        img1 = transform(Image.open(img1_path)).unsqueeze(0)
        img2 = transform(Image.open(img2_path)).unsqueeze(0)
    except Exception as e:
        raise ValueError(f"Failed to process images for LPIPS: {str(e)}")
    
    img1 = img1.to(device)
    img2 = img2.to(device)
    
    with torch.no_grad():
        lpips_score = lpips_model(img1, img2).item()
    return lpips_score

def calculate_clip_similarity(img1_path, img2_path, clip_model, clip_processor, device):
    """Calculate CLIP embedding cosine similarity between two images."""
    try:
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
    except Exception as e:
        raise ValueError(f"Failed to load images for CLIP similarity: {str(e)}")
    
    inputs1 = clip_processor(images=img1, return_tensors="pt")
    inputs2 = clip_processor(images=img2, return_tensors="pt")
    
    inputs1 = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs1.items()}
    inputs2 = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs2.items()}
    
    with torch.no_grad():
        img1_features = clip_model.get_image_features(**inputs1)
        img2_features = clip_model.get_image_features(**inputs2)
    
    img1_features = img1_features / img1_features.norm(dim=-1, keepdim=True)
    img2_features = img2_features / img2_features.norm(dim=-1, keepdim=True)
    
    similarity = (img1_features @ img2_features.T).item()
    return similarity

def main():
    # Set paths
    fake_dir = "/shared/shashmi/inversion/inversion/SD3.5/real"
    inverted_dir = "/shared/shashmi/inversion/prompt_conditioned_inversion/prompt_conditioned/SD3.5/inv_real"
    output_file = "SD3.5_condtioned_image_real_inv_real_metrics.jsonl"
    error_log = "processing_errors.log"
    
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Initializing models...")
    lpips_model = lpips.LPIPS(net='alex').to(device).eval()
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    image_files = [f for f in os.listdir(fake_dir) if f.endswith('.png')]
    total_images = len(image_files)
    print(f"\nProcessing {total_images} images...")
    
    results = []
    errors = []
    
    with open(output_file, 'w') as f, open(error_log, 'w') as error_f:
        for img_file in tqdm(image_files, desc="Processing images", unit="img"):
            try:
                fake_path = os.path.join(fake_dir, img_file)
                inverted_path = os.path.join(inverted_dir, img_file)
                
                # Get dimensions of both images
                fake_dims = get_image_dimensions(fake_path)
                inverted_dims = get_image_dimensions(inverted_path)
                
                # Use the dimensions of the fake image as target size
                target_size = fake_dims
                
                # Load and resize images if necessary
                fake_img = load_and_resize_image(fake_path)
                inverted_img = load_and_resize_image(inverted_path, target_size)
                
                # Calculate metrics
                psnr_value = calculate_psnr(fake_img, inverted_img)
                ssim_value = calculate_ssim(fake_img, inverted_img)
                lpips_value = calculate_lpips(fake_path, inverted_path, lpips_model, device)
                clip_sim = calculate_clip_similarity(fake_path, inverted_path, clip_model, clip_processor, device)
                
                combined_score = calculate_combined_score(psnr_value, ssim_value, lpips_value, clip_sim)
                
                result = {
                    'image_path': fake_path,
                    'original_dimensions': {
                        'fake': fake_dims,
                        'inverted': inverted_dims
                    },
                    'metrics': {
                        'psnr': float(psnr_value),
                        'ssim': float(ssim_value),
                        'lpips': float(lpips_value),
                        'clip_similarity': float(clip_sim),
                        'combined_score': float(combined_score)
                    }
                }
                
                json.dump(result, f)
                f.write('\n')
                
                results.append({
                    'image': img_file,
                    'psnr': psnr_value,
                    'ssim': ssim_value,
                    'lpips': lpips_value,
                    'clip_similarity': clip_sim,
                    'combined_score': combined_score
                })
                
            except Exception as e:
                error_msg = f"Error processing {img_file}: {str(e)}"
                print(f"\nWarning: {error_msg}")
                errors.append(error_msg)
                error_f.write(f"{error_msg}\n")
                continue
    
    if results:
        print("\nCalculating average metrics...")
        
        print("\nResults for each image pair:")
        print("-" * 80)
        for result in results:
            print(f"\nImage: {result['image']}")
            print(f"PSNR: {result['psnr']:.4f}")
            print(f"SSIM: {result['ssim']:.4f}")
            print(f"LPIPS: {result['lpips']:.4f}")
            print(f"CLIP Similarity: {result['clip_similarity']:.4f}")
            print(f"Combined Score: {result['combined_score']:.4f}")
        
        avg_psnr = sum(r['psnr'] for r in results) / len(results)
        avg_ssim = sum(r['ssim'] for r in results) / len(results)
        avg_lpips = sum(r['lpips'] for r in results) / len(results)
        avg_clip = sum(r['clip_similarity'] for r in results) / len(results)
        avg_combined = sum(r['combined_score'] for r in results) / len(results)
        
        print("\nAverage Metrics:")
        print("-" * 80)
        print(f"Average PSNR: {avg_psnr:.4f}")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average LPIPS: {avg_lpips:.4f}")
        print(f"Average CLIP Similarity: {avg_clip:.4f}")
        print(f"Average Combined Score: {avg_combined:.4f}")
    
    if errors:
        print(f"\nEncountered {len(errors)} errors. See {error_log} for details.")
    
    print(f"\nResults have been saved to {output_file}")

if __name__ == "__main__":
    main()