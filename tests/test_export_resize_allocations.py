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

import tracemalloc

import numpy as np
import pytest

from lerobot.inference.export.images import _axis_weights, resize_bicubic


def _allocating_resize(image, size):
    columns, x_weights = _axis_weights(image.shape[1], size[1])
    rows, y_weights = _axis_weights(image.shape[0], size[0])
    horizontal = (image[:, columns].astype(np.float32) * x_weights[None, :, :, None]).sum(axis=2)
    resized = (horizontal[rows] * y_weights[:, :, None, None]).sum(axis=1)
    return np.ascontiguousarray(np.rint(resized.clip(0, 255)), dtype=np.uint8)


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("shape,size", [((96, 128), (64, 64)), ((23, 23), (32, 32)), ((7, 11), (3, 5))])
def test_resize_workspace_preserves_pixels_and_readonly_input(channels, shape, size):
    image = np.random.default_rng(123).integers(0, 256, (*shape, channels), dtype=np.uint8)
    image = image[::-1, ::-1]
    before = image.copy()
    image.flags.writeable = False
    expected = _allocating_resize(image, size)
    actual = resize_bicubic(image, size)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(image, before)
    assert actual.dtype == np.uint8
    assert actual.flags.c_contiguous


def test_resize_reuses_large_temporaries():
    image = np.random.default_rng(456).integers(0, 256, (96, 128, 3), dtype=np.uint8)

    def peak(function):
        tracemalloc.start()
        try:
            function(image, (64, 64))
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    resize_bicubic(image, (64, 64))
    _allocating_resize(image, (64, 64))
    assert peak(resize_bicubic) < 0.8 * peak(_allocating_resize)
