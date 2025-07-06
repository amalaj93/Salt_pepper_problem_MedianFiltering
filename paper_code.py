import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import time
from tqdm import tqdm

# Configuration
DATASET_PATH = "COVID-19_Radiography_Dataset"  # Your local dataset folder
OUTPUT_DIR = "covid_results"
NOISE_LEVELS = [0.1, 0.3, 0.5]  # 10%, 30%, 50% noise densities
REPORT_NAME = "covid_denoising_report.pdf"
MAX_IMAGES =  150 # Process only first 25 images

def setup_environment():
    """Create necessary directories"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "noisy"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "denoised_mf"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "denoised_amf"), exist_ok=True)

def load_local_images():
    """Load first 25 images from local dataset folder with verification"""
    image_files = []
    for root, _, files in os.walk(DATASET_PATH):
        for file in files[:MAX_IMAGES]:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(root, file)
                if os.path.exists(full_path):
                    image_files.append(full_path)
                    if len(image_files) >= MAX_IMAGES:
                        return image_files
                else:
                    print(f"Warning: File not found - {full_path}")
    return image_files

def add_sp_noise(image, prob):
    """Add salt-and-pepper noise to image"""
    noisy = np.copy(image)
    h, w = image.shape
    salt = np.random.rand(h, w) < prob/2
    pepper = np.random.rand(h, w) < prob/2
    noisy[salt] = 255
    noisy[pepper] = 0
    return noisy

def median_filter(image, size=3):
    """Standard median filter"""
    return cv2.medianBlur(image, size)

def adaptive_median_filter(image, max_size=7):
    """Adaptive median filter with dynamic window sizing"""
    h, w = image.shape
    padded = cv2.copyMakeBorder(image, max_size//2, max_size//2, 
                               max_size//2, max_size//2, 
                               cv2.BORDER_REFLECT)
    output = np.zeros_like(image)
    
    for i in range(h):
        for j in range(w):
            window_size = 3
            while window_size <= max_size:
                window = padded[i:i+window_size, j:j+window_size]
                med = np.median(window)
                min_val, max_val = np.min(window), np.max(window)
                
                center = padded[i + window_size//2, j + window_size//2]
                
                if min_val < med < max_val:
                    output[i,j] = center if min_val < center < max_val else med
                    break
                window_size += 2
            else:
                output[i,j] = med
    return output

def verify_save(path):
    """Verify an image was saved successfully"""
    if not os.path.exists(path):
        print(f"Warning: Failed to save {path}")
        return False
    img = cv2.imread(path)
    if img is None:
        print(f"Warning: Corrupted save at {path}")
        return False
    return True

def process_images(image_paths):
    """Main processing pipeline with robust error handling"""
    results = []
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            # Verify and load original
            if not os.path.exists(img_path):
                print(f"Warning: Missing original - {img_path}")
                continue
                
            original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if original is None:
                print(f"Warning: Cannot read {img_path}")
                continue
                
            original = cv2.resize(original, (256, 256))
            filename = os.path.basename(img_path)
            
            for noise_level in NOISE_LEVELS:
                # Add noise
                noisy = add_sp_noise(original, noise_level)
                noisy_filename = f"{noise_level*100:.0f}pct_{filename}"
                noisy_path = os.path.join(OUTPUT_DIR, "noisy", noisy_filename)
                cv2.imwrite(noisy_path, noisy)
                if not verify_save(noisy_path):
                    continue
                
                # Apply filters
                start_mf = time.time()
                denoised_mf = median_filter(noisy)
                time_mf = time.time() - start_mf
                
                start_amf = time.time()
                denoised_amf = adaptive_median_filter(noisy)
                time_amf = time.time() - start_amf
                
                # Save results with verification
                mf_path = os.path.join(OUTPUT_DIR, "denoised_mf", noisy_filename)
                amf_path = os.path.join(OUTPUT_DIR, "denoised_amf", noisy_filename)
                
                cv2.imwrite(mf_path, denoised_mf)
                cv2.imwrite(amf_path, denoised_amf)
                
                if not all([verify_save(mf_path), verify_save(amf_path)]):
                    continue
                
                # Calculate metrics
                metrics = {
                    'Image': filename,
                    'Noise Level': f"{noise_level*100:.0f}%",
                    'PSNR_MF': psnr(original, denoised_mf, data_range=255),
                    'SSIM_MF': ssim(original, denoised_mf, data_range=255),
                    'Time_MF': time_mf,
                    'PSNR_AMF': psnr(original, denoised_amf, data_range=255),
                    'SSIM_AMF': ssim(original, denoised_amf, data_range=255),
                    'Time_AMF': time_amf
                }
                results.append(metrics)
                
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue
    
    return pd.DataFrame(results)

def generate_report(results_df):
    """Generate comprehensive PDF report with robust error handling"""
    with PdfPages(os.path.join(OUTPUT_DIR, REPORT_NAME)) as pdf:
        # 1. Summary Statistics
        plt.figure(figsize=(12, 8))
        plt.axis('off')
        
        try:
            avg_results = results_df.groupby('Noise Level').mean(numeric_only=True)
            summary_text = [
                "COVID-19 X-ray Denoising Performance Summary\n",
                f"Total Images Processed: {len(results_df['Image'].unique())}",
                f"Noise Levels Tested: {', '.join([f'{nl}%' for nl in NOISE_LEVELS])}\n",
                "Average Performance Metrics:",
                "Median Filter:",
                f"  • PSNR: {avg_results['PSNR_MF'].mean():.2f} dB",
                f"  • SSIM: {avg_results['SSIM_MF'].mean():.3f}",
                f"  • Processing Time: {avg_results['Time_MF'].mean():.3f} sec\n",
                "Adaptive Median Filter:",
                f"  • PSNR: {avg_results['PSNR_AMF'].mean():.2f} dB",
                f"  • SSIM: {avg_results['SSIM_AMF'].mean():.3f}",
                f"  • Processing Time: {avg_results['Time_AMF'].mean():.3f} sec"
            ]
            
            plt.text(0.1, 0.5, "\n".join(summary_text), fontsize=11, linespacing=1.5)
            plt.title("COVID-19 X-ray Analysis Report", pad=20, fontsize=14)
            pdf.savefig()
            plt.close()
        except Exception as e:
            print(f"Error generating summary: {str(e)}")

        # 2. Visual Comparison
        try:
            sample_size = min(3, len(results_df))
            if sample_size > 0:
                sample = results_df.sample(sample_size)
                for _, row in sample.iterrows():
                    try:
                        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
                        
                        # Load original
                        original_path = os.path.join(DATASET_PATH, row['Image'])
                        if not os.path.exists(original_path):
                            print(f"Warning: Missing original - {original_path}")
                            continue
                            
                        original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
                        if original is None:
                            print(f"Warning: Cannot read {original_path}")
                            continue
                            
                        original = cv2.resize(original, (256, 256))
                        
                        # Load processed images
                        noisy_path = os.path.join(OUTPUT_DIR, "noisy", f"{row['Noise Level'].replace('%','')}pct_{row['Image']}")
                        mf_path = os.path.join(OUTPUT_DIR, "denoised_mf", f"{row['Noise Level'].replace('%','')}pct_{row['Image']}")
                        amf_path = os.path.join(OUTPUT_DIR, "denoised_amf", f"{row['Noise Level'].replace('%','')}pct_{row['Image']}")
                        
                        images = []
                        titles = []
                        
                        for path, title in zip(
                            [original_path, noisy_path, mf_path, amf_path],
                            ["Original", f"Noisy ({row['Noise Level']})", 
                             f"Median Filter\nPSNR: {row['PSNR_MF']:.2f} dB", 
                             f"Adaptive MF\nPSNR: {row['PSNR_AMF']:.2f} dB"]
                        ):
                            if os.path.exists(path):
                                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                                if img is not None:
                                    images.append(img)
                                    titles.append(title)
                                else:
                                    print(f"Warning: Cannot read {path}")
                            else:
                                print(f"Warning: Missing {path}")
                        
                        if len(images) == 4:
                            for img, ax, title in zip(images, axs, titles):
                                ax.imshow(img, cmap='gray')
                                ax.set_title(title, fontsize=10)
                                ax.axis('off')
                            
                            plt.suptitle(f"Sample Results: {row['Image']}", y=1.05, fontsize=12)
                            plt.tight_layout()
                            pdf.savefig()
                        
                        plt.close()
                        
                    except Exception as e:
                        print(f"Error generating sample {row['Image']}: {str(e)}")
                        plt.close()
                        continue
        except Exception as e:
            print(f"Error in visual comparison: {str(e)}")

        # 3. Performance Metrics
        try:
            metrics = ['PSNR', 'SSIM']
            for metric in metrics:
                plt.figure(figsize=(12, 6))
                for noise_level in results_df['Noise Level'].unique():
                    subset = results_df[results_df['Noise Level'] == noise_level]
                    plt.plot(subset[f'{metric}_MF'], 'o-', label=f'MF {noise_level}')
                    plt.plot(subset[f'{metric}_AMF'], 's--', label=f'AMF {noise_level}')
                
                plt.xlabel('Image Index')
                plt.ylabel(metric)
                plt.title(f'{metric} Comparison Across Noise Levels')
                plt.legend()
                plt.grid(True)
                pdf.savefig()
                plt.close()
        except Exception as e:
            print(f"Error in metrics plot: {str(e)}")

if __name__ == "__main__":
    print(f"COVID-19 X-ray Denoising Analysis (First {MAX_IMAGES} images only)")
    print("="*60)
    
    # Setup environment
    setup_environment()
    
    # Load local images
    print("\n1. Loading first 25 images...")
    image_paths = load_local_images()
    if not image_paths:
        print(f"Error: No valid images found in {DATASET_PATH}")
        print("Please verify:")
        print(f"1. The folder '{DATASET_PATH}' exists")
        print("2. It contains COVID-19 X-ray images (PNG/JPG format)")
        print("3. You have read permissions")
        exit()
    
    # Process images
    print(f"\n2. Processing {len(image_paths)} images...")
    results_df = process_images(image_paths)
    
    if results_df.empty:
        print("Error: No images processed successfully")
        exit()
    
    results_df.to_csv(os.path.join(OUTPUT_DIR, "results.csv"), index=False)
    
    # Generate report
    print("\n3. Generating report...")
    generate_report(results_df)
    
    print("\nAnalysis Complete!")
    print(f"Results saved to: {os.path.abspath(OUTPUT_DIR)}")
    print(f"- Processed images: {len(results_df['Image'].unique())}/{len(image_paths)}")
    print(f"- PDF Report: {REPORT_NAME}")
    print(f"- Raw Data: results.csv")