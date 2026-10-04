import math
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

def calculate_a_index(psnr_value, ssim_value, lpips_value, clip_sim, sigma=0.9):
    """A-index from the paper: s = a1*PSNR + a2*SSIM + a3*(1-LPIPS) + a4*CLIP, A = sigmoid(-sigma*s).

    Faithful reconstructions (generated content) receive low scores.
    """
    s = -0.0181 * psnr_value + 1.380 * ssim_value - 4.058 * (1 - lpips_value) + 8.066 * clip_sim
    z = -sigma * s
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)

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
    import argparse
    ap = argparse.ArgumentParser(description='Similarity metrics between images and their reconstructions.')
    ap.add_argument('--source_dir', required=True, help='folder of query images (.png)')
    ap.add_argument('--recon_dir', required=True, help='folder of reconstructions with the same file names')
    ap.add_argument('--out', required=True, help='output .jsonl')
    args = ap.parse_args()
    fake_dir = args.source_dir
    inverted_dir = args.recon_dir
    output_file = args.out
    error_log = os.path.splitext(args.out)[0] + '_errors.log'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
                
                a_index = calculate_a_index(psnr_value, ssim_value, lpips_value, clip_sim)
                
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
                        'a_index': float(a_index)
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
                    'a_index': a_index
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
            print(f"A-index: {result['a_index']:.4f}")
        
        avg_psnr = sum(r['psnr'] for r in results) / len(results)
        avg_ssim = sum(r['ssim'] for r in results) / len(results)
        avg_lpips = sum(r['lpips'] for r in results) / len(results)
        avg_clip = sum(r['clip_similarity'] for r in results) / len(results)
        avg_a_index = sum(r['a_index'] for r in results) / len(results)
        
        print("\nAverage Metrics:")
        print("-" * 80)
        print(f"Average PSNR: {avg_psnr:.4f}")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average LPIPS: {avg_lpips:.4f}")
        print(f"Average CLIP Similarity: {avg_clip:.4f}")
        print(f"Average A-index: {avg_a_index:.4f}")
    
    if errors:
        print(f"\nEncountered {len(errors)} errors. See {error_log} for details.")
    
    print(f"\nResults have been saved to {output_file}")

if __name__ == "__main__":
    main()