import os
import cv2


class TUMDataset:
    def __init__(self, sequence_path):
        self.sequence_path = sequence_path
        rgb_txt = os.path.join(sequence_path, "rgb.txt")
        
        if not os.path.exists(rgb_txt):
            raise ValueError(f"rgb.txt not found in: {sequence_path}")
        
        # Parse rgb.txt file
        self.timestamps = []
        self.image_paths = []
        
        with open(rgb_txt, "r") as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    timestamp = float(parts[0])
                    filename = parts[1]
                    full_path = os.path.join(sequence_path, filename)
                    
                    if os.path.exists(full_path):
                        self.timestamps.append(timestamp)
                        self.image_paths.append(full_path)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in: {sequence_path}")
        
        print(f"Loaded {len(self.image_paths)} images from {sequence_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {self.image_paths[idx]}")
        return img
    
    def get_timestamp(self, idx):
        """Get timestamp for frame at index idx."""
        return self.timestamps[idx]

