import torch
import kornia.feature as KF
import numpy as np
import cv2

def test_kornia():
    print("Testing Kornia DISK and LightGlue...")
    device = torch.device('cpu')
    
    # Create dummy image
    img = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(img, (100, 100), 10, 255, -1)
    cv2.circle(img, (200, 200), 10, 255, -1)
    
    # Kornia expects (B, C, H, W) float tensors in [0, 1]
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)
    
    # DISK
    print("Initializing DISK...")
    try:
        disk = KF.DISK.from_pretrained('depth').to(device)
        disk_features = disk(img_tensor)
        # disk_features is a list of objects (one per batch element)
        # Each object has keypoints and descriptors? 
        # Actually DISK in kornia might return something else.
        # Let's inspect
        print("DISK output type:", type(disk_features))
        if isinstance(disk_features, list):
             print("Element type:", type(disk_features[0]))
             # Kornia DISK returns Keypoints and Descriptors
             # It returns a list of Features (named tuple or similar)
             kps = disk_features[0].keypoints
             descs = disk_features[0].descriptors
             print(f"DISK: found {len(kps)} keypoints")
    except Exception as e:
        print(f"DISK failed: {e}")

    # LightGlue
    print("Initializing LightGlue...")
    try:
        # LightGlue expects {'image0': ..., 'image1': ...} usually or separate features?
        # Kornia's LightGlue matcher
        matcher = KF.LightGlue(features='disk').to(device)
        
        # Create a second image (shifted)
        img2 = np.roll(img, 10, axis=1)
        img_tensor2 = torch.from_numpy(img2).float() / 255.0
        img_tensor2 = img_tensor2.unsqueeze(0).unsqueeze(0).to(device)
        
        # Detect features for both
        feats1 = disk(img_tensor)
        feats2 = disk(img_tensor2)
        
        # Match
        # LightGlue forward:
        # forward(data: dict) -> dict
        # data keys: image0, keypoints0, descriptors0, image1, keypoints1, descriptors1
        
        input_dict = {
            "image0": { "keypoints": feats1[0].keypoints.unsqueeze(0), "descriptors": feats1[0].descriptors.unsqueeze(0), "image_size": torch.tensor(img.shape).unsqueeze(0) },
            "image1": { "keypoints": feats2[0].keypoints.unsqueeze(0), "descriptors": feats2[0].descriptors.unsqueeze(0), "image_size": torch.tensor(img.shape).unsqueeze(0) }
        }
        # Wait, Kornia LightGlue API might differ.
        # It typically takes:
        # matcher(lafs1, desc1, lafs2, desc2) or similar?
        # Or matcher({"image0": ..., ...})
        
        # Let's try minimal call
        # Kornia LightGlue forward doc:
        # forward(data) where data is a dict with keys:
        # keypoints0, descriptors0, image0, keypoints1, descriptors1, image1 (optional images)
        
        data = {
            'keypoints0': feats1[0].keypoints.unsqueeze(0),
            'descriptors0': feats1[0].descriptors.unsqueeze(0),
            'keypoints1': feats2[0].keypoints.unsqueeze(0),
            'descriptors1': feats2[0].descriptors.unsqueeze(0),
        }
        
        out = matcher(data)
        matches = out['matches']
        print(f"LightGlue: found {len(matches[0])} matches")
        
    except Exception as e:
        print(f"LightGlue failed: {e}")

if __name__ == "__main__":
    test_kornia()

