"""
Image Feature Extraction 
"""
import pandas as pd
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import warnings
import urllib3
warnings.filterwarnings('ignore')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class ImageFeatureExtractor:
    
    def __init__(self, feature_dim=2048):

        self.feature_dim = feature_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.model.to(self.device)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def load_image_from_url(self, url, timeout=10, retries=2):
        if any(skip in url.lower() for skip in ['drive.google.com', 'youtube.com', 'vimeo.com', '.html', '.pdf']):
            return None
        
        if 'commons.wikimedia.org/wiki/File:' in url:
            filename = url.split('/File:')[-1]
            url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{filename}"
        elif 'commons.wikimedia.org/wiki/Special:FilePath/' in url:
            pass
        elif 'upload.wikimedia.org' in url:
            pass
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        for attempt in range(retries + 1):
            try:
                response = requests.get(url, timeout=timeout, stream=True, headers=headers, verify=False)
                response.raise_for_status()
                
                content_type = response.headers.get('content-type', '').lower()
                if 'image' not in content_type and not url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
                    pass
                
                img = Image.open(BytesIO(response.content))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return img
            except requests.exceptions.SSLError:
                try:
                    import urllib3
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                    response = requests.get(url, timeout=timeout, stream=True, headers=headers, verify=False)
                    response.raise_for_status()
                    img = Image.open(BytesIO(response.content))
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    return img
                except:
                    if attempt == retries:
                        return None
                    continue
            except Exception as e:
                if attempt == retries:
                    if '403' not in str(e) and '404' not in str(e) and 'timeout' not in str(e).lower():
                        pass  
                    return None
                continue
        
        return None
    
    def extract_features(self, image):

        try:
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(img_tensor)
                features = features.squeeze().cpu().numpy()
                features = features.flatten()
            
            return features
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None
    
    def extract_features_from_url(self, url):

        image = self.load_image_from_url(url)
        if image is None:
            return None
        return self.extract_features(image)
    
    def extract_features_batch(self, urls, batch_size=32, verbose=True):

        features_list = []
        failed_indices = []
        successful_features = []
        
        for i, url in enumerate(urls):
            if verbose and (i + 1) % 50 == 0:
                print(f"Processing {i + 1}/{len(urls)} images...")
            
            features = self.extract_features_from_url(url)
            if features is not None:
                features_list.append(features)
                successful_features.append(features)
            else:
                if successful_features:
                    mean_features = np.mean(successful_features, axis=0)
                    features_list.append(mean_features)
                else:
                    features_list.append(np.zeros(self.feature_dim))
                failed_indices.append(i)
        
        if failed_indices and verbose:
            print(f"\nWarning: Failed to extract features from {len(failed_indices)} images")
            print(f"Using mean features for failed images to maintain data quality")
        
        return np.array(features_list), failed_indices
    
    def extract_features_from_dataframe(self, df, url_column='Image URL', 
                                       save_path=None, verbose=True):

        urls = df[url_column].tolist()
        features, failed_indices = self.extract_features_batch(urls, verbose=verbose)
        
        if save_path:
            np.save(save_path, features)
            if verbose:
                print(f"\nFeatures saved to {save_path}")
        
        return features, failed_indices


def main():
    df = pd.read_csv('cleaned_data.csv')
    print(f"Loaded {len(df)} samples")
    
    extractor = ImageFeatureExtractor()
    
    print("\nExtracting features from images...")
    features, failed_indices = extractor.extract_features_from_dataframe(
        df, 
        save_path='image_features.npy',
        verbose=True
    )
    
    print(f"\nFeature extraction complete!")
    print(f"Feature matrix shape: {features.shape}")
    print(f"Failed extractions: {len(failed_indices)}")


if __name__ == "__main__":
    main()
