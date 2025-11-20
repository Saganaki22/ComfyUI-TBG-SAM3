"""
SAM3 Utility Functions - Based on Official SAM3 API
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from PIL import Image

try:
    import folder_paths
except ImportError:
    # Fallback if folder_paths not available
    class folder_paths:
        models_dir = "models"


class SAM3ImageSegmenter:
    """
    SAM3 Image Segmentation using official API
    Based on: sam3.model_builder.build_sam3_image_model()
    """

    def __init__(self, device: str = "cuda"):
        """Initialize SAM3 image segmenter"""
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load SAM3 model using official API"""
        try:
            # Try different import paths based on SAM3 version
            try:
                from sam3.model_builder import build_sam3_image_model
                from sam3.model.sam3_image_processor import Sam3Processor
            except ImportError:
                # Fallback for different SAM3 structure
                from sam3 import build_sam3_image_model, Sam3Processor

            print(f"[SAM3] Loading SAM3 image model on {self.device}...")

            # Build model
            self.model = build_sam3_image_model(device=self.device)
            self.processor = Sam3Processor(self.model)

            print("[SAM3] Model loaded successfully")

        except ImportError as e:
            error_msg = f"SAM3 not installed or import failed: {e}\n"
            error_msg += "Please run:\n"
            error_msg += "pip install git+https://github.com/facebookresearch/sam3.git"
            raise RuntimeError(error_msg)
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM3 model: {str(e)}")

    def segment_with_text(
        self,
        image: torch.Tensor,
        text_prompt: str,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Segment image using text prompt"""
        from PIL import Image

        # Convert tensor to PIL Image
        if isinstance(image, torch.Tensor):
            img_np = (image.cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)
        else:
            pil_image = image

        # Use official API
        inference_state = self.processor.set_image(pil_image)
        output = self.processor.set_text_prompt(
            state=inference_state,
            prompt=text_prompt
        )

        # Extract and filter results
        masks = output.get("masks", [])
        boxes = output.get("boxes", [])
        scores = output.get("scores", [])

        filtered_results = {"masks": [], "boxes": [], "scores": []}

        for i, score in enumerate(scores):
            if score >= threshold:
                filtered_results["masks"].append(masks[i])
                filtered_results["boxes"].append(boxes[i])
                filtered_results["scores"].append(float(score))

        return filtered_results

    def segment_with_points(
        self,
        image: torch.Tensor,
        points: List[Tuple[int, int]],
        point_labels: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Segment using point prompts"""
        if point_labels is None:
            point_labels = [1] * len(points)

        # Convert to PIL
        img_np = (image.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)

        # Set image
        inference_state = self.processor.set_image(pil_image)

        # Add point prompts
        output = self.processor.set_point_prompt(
            state=inference_state,
            points=points,
            labels=point_labels
        )

        return {
            "masks": output.get("masks", []),
            "boxes": output.get("boxes", []),
            "scores": output.get("scores", [1.0] * len(output.get("masks", [])))
        }

    def extract_points_from_mask(
        self,
        mask: torch.Tensor,
        num_points: int = 5
    ) -> List[Tuple[int, int]]:
        """Extract point coordinates from binary mask"""
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)

        # Handle 3D masks
        if len(mask_np.shape) == 3:
            mask_np = mask_np.squeeze() if mask_np.shape[0] == 1 else mask_np[:, :, 0]

        # Find foreground pixels
        y_coords, x_coords = np.where(mask_np > 0.5)

        if len(y_coords) == 0:
            return []

        # Sample points uniformly
        total_points = len(y_coords)
        if total_points <= num_points:
            indices = range(total_points)
        else:
            indices = np.linspace(0, total_points - 1, num_points, dtype=int)

        points = [(int(x_coords[i]), int(y_coords[i])) for i in indices]
        return points


class DepthEstimator:
    """Depth map generation using MiDaS"""

    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load MiDaS depth model"""
        try:
            from transformers import DPTImageProcessor, DPTForDepthEstimation

            model_name = "Intel/dpt-hybrid-midas"
            print(f"[SAM3] Loading depth model: {model_name}")

            self.processor = DPTImageProcessor.from_pretrained(model_name)
            self.model = DPTForDepthEstimation.from_pretrained(model_name).to(self.device)
            self.model.eval()

            print("[SAM3] Depth model loaded")

        except Exception as e:
            print(f"[SAM3] Could not load depth model: {e}")
            print("[SAM3] Depth features will be disabled")
            self.model = None

    def estimate_depth(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Estimate depth map"""
        if self.model is None:
            # Return dummy depth
            h, w = image.shape[:2]
            return torch.zeros((h, w), dtype=torch.float32)

        # Convert to PIL
        img_np = (image.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)

        # Process
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            depth = outputs.predicted_depth

        # Resize to original size
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

        # Apply mask if provided
        if mask is not None:
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            depth = depth * mask.to(self.device)

        return depth.cpu()


def convert_to_segs(
    masks: List[torch.Tensor],
    boxes: List[torch.Tensor],
    scores: List[float],
    image_size: Tuple[int, int],
    label: str = "sam3"
) -> Tuple[Tuple[int, int], List[Tuple]]:
    """
    Convert SAM3 outputs to Impact Pack SEGS format

    SEGS format: ((width, height), [seg1, seg2, ...])
    Each seg: (cropped_mask, crop_region, bbox, label, confidence)
    """
    h, w = image_size
    segs = []

    for i, (mask, score) in enumerate(zip(masks, scores)):
        # Convert mask to numpy
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)

        # Ensure 2D
        if len(mask_np.shape) == 3:
            mask_np = mask_np.squeeze() if mask_np.shape[0] == 1 else mask_np[0]

        # Resize if needed
        if mask_np.shape != (h, w):
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float()
            mask_np = torch.nn.functional.interpolate(
                mask_tensor,
                size=(h, w),
                mode="nearest"
            ).squeeze().numpy()

        # Get bounding box
        if i < len(boxes):
            box = boxes[i]
            if isinstance(box, torch.Tensor):
                box = box.cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
        else:
            # Calculate from mask
            rows = np.any(mask_np > 0.5, axis=1)
            cols = np.any(mask_np > 0.5, axis=0)
            if not rows.any() or not cols.any():
                continue
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]

        # Ensure valid bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        # Crop mask
        cropped_mask = mask_np[y1:y2+1, x1:x2+1]
        cropped_mask_tensor = torch.from_numpy(cropped_mask).float()

        # Create SEG tuple
        seg = (
            cropped_mask_tensor,        # cropped mask
            (x1, y1, x2 - x1, y2 - y1), # crop_region (x, y, w, h)
            (x1, y1, x2, y2),           # bbox (x1, y1, x2, y2)
            label,                       # label
            score                        # confidence
        )
        segs.append(seg)

    return ((w, h), segs)
