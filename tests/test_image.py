# -*- coding: utf-8 -*-
"""Image pre-processing and the vision-side inputs (qev_decide.image); needs Pillow only."""
import io

import numpy as np
import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from qev_decide.image import (IMAGE_TOKEN_ID, ImageConfig, bilinear_inputs, decode_image, expand_image_tokens,  # noqa: E402
                              fit_long_side, image_tokens_for_grid, position_ids, prepare_images, smart_resize,
                              vision_position_ids)


def test_smart_resize_rounds_to_the_factor_and_respects_bounds():
    assert smart_resize(256, 256, 32, 65536, 262144) == (256, 256)
    assert smart_resize(224, 224, 32, 65536, 262144) == (256, 256)             # a thumbnail is scaled UP to min_pixels
    h, w = smart_resize(5000, 5000, 32, 65536, 262144)
    assert h % 32 == 0 and w % 32 == 0 and h * w <= 262144
    with pytest.raises(ValueError):
        smart_resize(10, 3000, 32, 65536, 262144)


def test_prepare_images_shapes_and_token_counts():
    out = prepare_images([Image.new("RGB", (224, 224), (10, 200, 30))], ImageConfig())
    assert out.pixel_values.dtype == np.float32 and out.pixel_values.shape[1] == 3 * 2 * 16 * 16
    assert out.image_grid_thw.tolist() == [[1, 16, 16]] and out.token_counts == [64]   # 256 x 256 px: 64 visual tokens
    assert out.pixel_values.shape[0] == 256
    assert -1.0 <= out.pixel_values.min() and out.pixel_values.max() <= 1.0


def test_one_image_at_most():
    assert prepare_images([]).pixel_values.shape[0] == 0
    with pytest.raises(ValueError, match="at most 1"):
        prepare_images([Image.new("RGB", (64, 64))] * 2)


def test_decode_image_sources_and_errors(tmp_path):
    buf = io.BytesIO()
    Image.new("RGBA", (40, 30), (255, 0, 0, 0)).save(buf, "PNG")
    img = decode_image(buf.getvalue())                                          # QEV drops alpha: it does not composite
    assert img.mode == "RGB" and img.getpixel((0, 0)) == (255, 0, 0)
    path = tmp_path / "a.png"
    path.write_bytes(buf.getvalue())
    assert decode_image(str(path)).size == (40, 30)
    with pytest.raises(ValueError):
        decode_image(b"not an image")
    with pytest.raises(ValueError):
        decode_image(Image.new("RGB", (100, 100)), max_pixels=50)
    with pytest.raises(ValueError):
        decode_image(str(tmp_path / "missing.png"))


def test_placeholders_expand_to_the_token_counts():
    prompt = "a<|image_pad|>b"
    assert expand_image_tokens(prompt, [3]).count("<|image_pad|>") == 3
    with pytest.raises(ValueError):
        expand_image_tokens(prompt, [3, 5])
    assert image_tokens_for_grid([[1, 16, 16]], 2) == [64]


def test_fit_long_side_zero_means_no_cap():
    img = Image.new("RGB", (1000, 500))
    assert fit_long_side(img, 0).size == (1000, 500)
    assert fit_long_side(img, 500).size == (500, 250)


def test_bilinear_taps_are_a_partition_of_unity():
    idx, wts = bilinear_inputs(np.array([[1, 8, 6]]), 48)
    assert idx.shape == wts.shape == (4, 48) and idx.dtype == np.int64 and wts.dtype == np.float32
    np.testing.assert_allclose(wts.sum(0), 1.0, atol=1e-6)
    assert idx.min() >= 0 and idx.max() < 48 * 48


def test_vision_position_ids_cover_the_grid_in_merge_block_order():
    pos = vision_position_ids(np.array([[1, 4, 4]]))
    assert pos.shape == (16, 2)
    assert pos[:4].tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]                 # the first 2x2 merge block
    assert sorted(map(tuple, pos.tolist())) == [(r, c) for r in range(4) for c in range(4)]


def test_mrope_positions():
    pos = position_ids([5, 6, 7, 8], np.zeros((0, 3), dtype=np.int64))
    assert pos.shape == (3, 4) and (pos == np.arange(4)).all()
    ids = [1] + [IMAGE_TOKEN_ID] * 4 + [2]
    pos = position_ids(ids, np.array([[1, 4, 4]]))
    assert pos[0, 1:5].tolist() == [1, 1, 1, 1] and pos[1, 1:5].tolist() == [1, 1, 2, 2] and pos[2, 1:5].tolist() == [1, 2, 1, 2]
    assert pos[:, 5].tolist() == [3, 3, 3]
    with pytest.raises(ValueError):
        position_ids(ids, np.zeros((0, 3), dtype=np.int64))
