# Copyright (c) 2025 Haian Jin. Modified for copy-paste augmentation.

import os
import cv2
import random
import numpy as np
import torch
from skimage.filters import gaussian
from pycocotools.coco import COCO


def image_copy_paste(img, paste_img, alpha, blend=True, sigma=1):
    """
    Paste one image onto another using alpha mask with optional blending.

    Args:
        img: Background image (numpy array, H x W x 3)
        paste_img: Image to paste (numpy array, H x W x 3)
        alpha: Alpha mask (numpy array, H x W), values in [0, 1]
        blend: Whether to apply Gaussian blur to alpha for smooth edges
        sigma: Gaussian blur sigma for blending

    Returns:
        Composited image (numpy array, H x W x 3)
    """
    if alpha is not None:
        if blend:
            alpha = gaussian(alpha, sigma=sigma, preserve_range=True)

        img_dtype = img.dtype
        alpha = alpha[..., None]  # Add channel dimension
        img = paste_img * alpha + img * (1 - alpha)
        img = img.astype(img_dtype)

    return img


class COCOAnimalExtractor:
    """Extract animals from COCO dataset for copy-paste augmentation."""

    def __init__(self, coco_root, coco_ann_file, animal_categories=None):
        """
        Initialize COCO dataset and filter for animal categories.

        Args:
            coco_root: Path to COCO images directory
            coco_ann_file: Path to COCO annotations JSON
            animal_categories: Set of COCO category IDs to use (default: all animals)
        """
        self.coco_root = coco_root
        self.coco = COCO(coco_ann_file)

        # Default animal categories
        if animal_categories is None:
            animal_categories = {16, 17, 18, 19, 20, 21, 22, 23, 24, 25}
        self.animal_categories = set(animal_categories)

        # Filter images that contain animals
        print("[COCOAnimalExtractor] Filtering images with animals...")
        self.valid_img_ids = []
        for img_id in self.coco.getImgIds():
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            annos = self.coco.loadAnns(ann_ids)
            has_animals = any(obj["category_id"] in self.animal_categories for obj in annos)
            if has_animals:
                self.valid_img_ids.append(img_id)

        print(f"[COCOAnimalExtractor] Found {len(self.valid_img_ids)} images with animals")

    def extract_random_animal(self):
        """
        Extract a random animal from COCO dataset.

        Returns:
            Dictionary with keys:
                - 'image': Full source image (numpy array, H x W x 3, RGB)
                - 'mask': Binary mask (numpy array, H x W, values 0 or 1)
                - 'bbox': Bounding box [x, y, width, height]
                - 'category_id': COCO category ID
                - 'category_name': Category name string
        """
        # Randomly sample an image with animals
        img_id = random.choice(self.valid_img_ids)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annos = self.coco.loadAnns(ann_ids)

        # Filter for animal annotations
        animal_annos = [obj for obj in annos if obj["category_id"] in self.animal_categories]

        if len(animal_annos) == 0:
            # Fallback: try another image
            return self.extract_random_animal()

        # Pick a random animal from this image
        anno = random.choice(animal_annos)

        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.coco_root, img_info["file_name"])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get mask
        mask = self.coco.annToMask(anno)

        return {
            "image": image,
            "mask": mask,
            "bbox": anno["bbox"],
            "category_id": anno["category_id"],
            "category_name": self.coco.cats[anno["category_id"]]["name"],
        }


def paste_animal_on_image_tensor(
    image_tensor,
    animal_data,
    position,
    scale,
    blend=True,
    sigma=3,
    overlay_mask=None,
):
    """
    Paste an animal onto a PyTorch image tensor.

    Args:
        image_tensor: (3, H, W) PyTorch tensor, values in [0, 1], RGB
        animal_data: Dictionary from COCOAnimalExtractor.extract_random_animal()
        position: (x, y) tuple for paste position in pixels
        scale: Fraction of image size (0.25 ⇒ largest object dimension ≈ 25% of image size)
        blend: Whether to blend edges
        sigma: Gaussian blur sigma
        overlay_mask: Optional torch tensor (1, H, W) to receive pasted region as binary mask

    Returns:
        Modified image tensor (3, H, W)
    """
    # Convert tensor to numpy for processing
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    img_np = (img_np * 255).astype(np.uint8)

    # Get animal image and mask
    source_img = animal_data["image"]  # (H_src, W_src, 3), RGB, uint8
    mask = animal_data["mask"]  # (H_src, W_src), values 0 or 1

    # Tight crop around the mask so scaling is based on the visible object
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        # Nothing to paste
        return image_tensor

    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    mask = mask[y1:y2, x1:x2]
    source_img = source_img[y1:y2, x1:x2]

    # Get target image dimensions
    tgt_h, tgt_w = img_np.shape[:2]

    # Desired maximum width/height as fraction of image dimensions
    target_w = max(1, int(tgt_w * scale))
    target_h = max(1, int(tgt_h * scale))

    # Compute scale factor so resized mask fits within target_w/target_h while
    # preserving aspect ratio
    src_h, src_w = mask.shape
    scale_factor = min(target_w / src_w, target_h / src_h)

    new_w = max(1, int(src_w * scale_factor))
    new_h = max(1, int(src_h * scale_factor))

    mask = cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    source_img = cv2.resize(source_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Get dimensions
    src_h, src_w = mask.shape
    tgt_h, tgt_w = img_np.shape[:2]

    # Calculate paste region
    x, y = int(position[0]), int(position[1])

    # Calculate valid paste region (handle boundaries)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(tgt_w, x + src_w), min(tgt_h, y + src_h)

    # Calculate source crop region
    src_x1, src_y1 = x1 - x, y1 - y
    src_x2, src_y2 = src_x1 + (x2 - x1), src_y1 + (y2 - y1)

    # Check if there's a valid region to paste
    if x2 <= x1 or y2 <= y1 or src_x2 <= src_x1 or src_y2 <= src_y1:
        return image_tensor

    # Extract regions
    paste_region = img_np[y1:y2, x1:x2].copy()
    source_region = source_img[src_y1:src_y2, src_x1:src_x2]
    mask_region = mask[src_y1:src_y2, src_x1:src_x2]

    # Blend
    result_region = image_copy_paste(
        paste_region, source_region, mask_region.astype(np.float32), blend=blend, sigma=sigma
    )

    # Update overlay mask if provided
    if overlay_mask is not None:
        mask_tensor = torch.from_numpy((mask_region > 0).astype(np.float32))
        overlay_slice = overlay_mask[:, y1:y2, x1:x2]
        overlay_mask[:, y1:y2, x1:x2] = torch.maximum(
            overlay_slice,
            mask_tensor.unsqueeze(0),
        )

    # Paste back
    img_np[y1:y2, x1:x2] = result_region

    # Convert back to tensor
    img_np = img_np.astype(np.float32) / 255.0
    result_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()

    return result_tensor


def apply_copy_paste_to_views(
    images,
    extractor,
    num_animals_range=(1, 2),
    scale_range=(0.1, 0.3),
    position_margin=0.1,
    blend=True,
    sigma=3,
    per_view_objects=False,
    generate_overlay_masks=False,
):
    """
    Apply copy-paste augmentation to a sequence of multi-view images.

    Args:
        images: (num_views, 3, H, W) PyTorch tensor
        extractor: COCOAnimalExtractor instance
        num_animals_range: (min, max) number of animals to paste per sequence/view
        scale_range: (min, max) scale range relative to image size
        position_margin: Margin from image edges (0.1 = 10% margin)
        blend: Whether to blend edges
        sigma: Gaussian blur sigma
        per_view_objects: If False, same objects across all views (different positions).
                          If True, sample NEW random objects for each view.
        generate_overlay_masks: Whether to produce binary masks indicating pasted regions

    Returns:
        Tuple (augmented_images, overlay_masks)
            augmented_images: tensor (num_views, 3, H, W)
            overlay_masks: tensor (num_views, 1, H, W) or None if not requested
    """
    num_views = images.shape[0]
    H, W = images.shape[2], images.shape[3]

    overlay_masks = None
    if generate_overlay_masks:
        overlay_masks = torch.zeros((num_views, 1, H, W), dtype=torch.float32)

    augmented_images = []

    if per_view_objects:
        # Mode: Different random objects for each view
        for view_idx in range(num_views):
            view_image = images[view_idx]  # (3, H, W)

            # Sample NEW objects for this view
            num_animals = random.randint(num_animals_range[0], num_animals_range[1])
            animals = [extractor.extract_random_animal() for _ in range(num_animals)]

            # Paste each object
            for animal in animals:
                x = random.uniform(position_margin * W, (1 - position_margin) * W)
                y = random.uniform(position_margin * H, (1 - position_margin) * H)
                scale = random.uniform(scale_range[0], scale_range[1])

                view_image = paste_animal_on_image_tensor(
                    view_image,
                    animal,
                    position=(x, y),
                    scale=scale,
                    blend=blend,
                    sigma=sigma,
                    overlay_mask=overlay_masks[view_idx] if overlay_masks is not None else None,
                )

            augmented_images.append(view_image)

    else:
        # Mode: Same objects across all views (different positions per view)
        num_animals = random.randint(num_animals_range[0], num_animals_range[1])
        animals = [extractor.extract_random_animal() for _ in range(num_animals)]

        for view_idx in range(num_views):
            view_image = images[view_idx]  # (3, H, W)

            # Paste same objects at different positions
            for animal in animals:
                x = random.uniform(position_margin * W, (1 - position_margin) * W)
                y = random.uniform(position_margin * H, (1 - position_margin) * H)
                scale = random.uniform(scale_range[0], scale_range[1])

                view_image = paste_animal_on_image_tensor(
                    view_image,
                    animal,
                    position=(x, y),
                    scale=scale,
                    blend=blend,
                    sigma=sigma,
                    overlay_mask=overlay_masks[view_idx] if overlay_masks is not None else None,
                )

            augmented_images.append(view_image)

    augmented_images = torch.stack(augmented_images)
    return augmented_images, overlay_masks
