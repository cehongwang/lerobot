# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NumPy camera geometry for exports whose backends cannot resize with antialiasing."""

import numpy as np


def _axis_weights(length: int, output: int) -> tuple[np.ndarray, np.ndarray]:
    scale = max(length / output, 1)
    center = (np.arange(output, dtype=np.float32) + 0.5) * np.float32(length / output)
    indices = np.floor(center - 2 * scale + 0.5).astype(np.int64)[:, None]
    indices = indices + np.arange(int(np.ceil(4 * scale)) + 1)
    distance = np.abs((indices.astype(np.float32) + 0.5 - center[:, None]) / np.float32(scale))
    # Antialiased bicubic uses a=-0.5 and renormalizes the taps that overlap the image.
    weights = np.where(
        distance < 1,
        ((1.5 * distance - 2.5) * distance) * distance + 1,
        np.where(distance < 2, ((-0.5 * distance + 2.5) * distance - 4) * distance + 2, 0),
    )
    weights *= (indices >= 0) & (indices < length)
    weights /= weights.sum(axis=1, keepdims=True)
    return indices.clip(0, length - 1), weights


def resize_bicubic(image: np.ndarray, size: tuple[int, int] | list[int]) -> np.ndarray:
    """Resize a uint8 HWC image with antialiased bicubic sampling and integer pixel rounding."""
    if image.dtype != np.uint8 or image.ndim != 3 or min(image.shape) == 0:
        raise ValueError("Expected a nonempty uint8 HWC image.")
    if len(size) != 2 or min(size) <= 0:
        raise ValueError("Expected a positive output height and width.")
    if image.shape[:2] == tuple(size):
        return image
    columns, x_weights = _axis_weights(image.shape[1], size[1])
    rows, y_weights = _axis_weights(image.shape[0], size[0])
    # Reuse private workspaces to avoid a second allocation for each multiply.
    horizontal = image[:, columns].astype(np.float32)
    horizontal *= x_weights[None, :, :, None]
    horizontal = horizontal.sum(axis=2)
    resized = horizontal[rows]
    resized *= y_weights[:, :, None, None]
    resized = resized.sum(axis=1)
    return np.ascontiguousarray(np.rint(resized.clip(0, 255)), dtype=np.uint8)


def resize_images(frame: dict[str, np.ndarray], geometry: dict) -> dict[str, np.ndarray]:
    """Apply the export's saved resize/crop geometry without changing the caller's frame."""
    result = dict(frame)
    for name in geometry["cameras"]:
        image = frame[name]
        if geometry["target_size"]:
            if geometry["letterbox"]:
                height, width = image.shape[:2]
                side = max(height, width)
                image = np.pad(
                    image,
                    (
                        ((side - height) // 2, (side - height + 1) // 2),
                        ((side - width) // 2, (side - width + 1) // 2),
                        (0, 0),
                    ),
                )
            edge = geometry["resize_edge"]
            image = resize_bicubic(image, (edge, edge))
            fraction = geometry["crop_fraction"]
            if fraction is not None and 0 < fraction < 1:
                crop = max(1, round(edge * fraction))
                offset = (edge - crop) // 2
                image = image[offset : offset + crop, offset : offset + crop]
            image = resize_bicubic(image, geometry["target_size"])
        result[name] = np.ascontiguousarray(resize_bicubic(image, geometry["image_size"]))
    return result
