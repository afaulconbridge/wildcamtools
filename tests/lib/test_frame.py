import warnings

import numpy as np

from wildcamtools.lib import BBox, Frame


class TestFrameDataclass:
    def test_frame_creation_with_raw_only(self):
        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1)

        assert np.array_equal(frame.raw, raw)
        assert frame.frame_no == 1
        assert frame.crop is None
        assert frame.rescale is None
        assert frame.crop_bbox is None
        assert frame.motion_proportion == -1.0
        assert frame.filter_keep is True

    def test_frame_creation_with_crop(self):
        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        crop = np.zeros((50, 100, 3), dtype=np.uint8)
        bbox = BBox(50, 25, 150, 75)

        frame = Frame(raw=raw, frame_no=1, crop=crop, crop_bbox=bbox)

        assert np.array_equal(frame.crop, crop)
        assert frame.crop_bbox == bbox

    def test_frame_creation_with_rescale(self):
        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        rescale = np.zeros((25, 50, 3), dtype=np.uint8)

        frame = Frame(raw=raw, frame_no=1, rescale=rescale)

        assert np.array_equal(frame.rescale, rescale)

    def test_frame_creation_with_all_fields(self):
        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        crop = np.zeros((50, 100, 3), dtype=np.uint8)
        rescale = np.zeros((25, 50, 3), dtype=np.uint8)
        bbox = BBox(50, 25, 150, 75)

        frame = Frame(
            raw=raw,
            frame_no=1,
            crop=crop,
            rescale=rescale,
            crop_bbox=bbox,
            motion_proportion=0.75,
            filter_keep=False,
        )

        assert np.array_equal(frame.raw, raw)
        assert np.array_equal(frame.crop, crop)
        assert np.array_equal(frame.rescale, rescale)
        assert frame.crop_bbox == bbox
        assert frame.motion_proportion == 0.75
        assert frame.filter_keep is False


class TestFrameOutputProperty:
    def test_output_returns_rescale_when_present(self):
        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        rescale = np.zeros((25, 50, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1, rescale=rescale)

        assert frame.output is rescale

    def test_output_returns_crop_when_no_rescale(self):
        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        crop = np.zeros((50, 100, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1, crop=crop)

        assert frame.output is crop

    def test_output_returns_raw_when_no_crop_or_rescale(self):
        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1)

        assert frame.output is raw

    def test_output_prioritizes_rescale_over_crop(self):
        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        crop = np.zeros((50, 100, 3), dtype=np.uint8)
        rescale = np.zeros((25, 50, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1, crop=crop, rescale=rescale)

        assert frame.output is rescale


class TestFrameWidthHeightRaw:
    def test_width_raw_from_raw(self):
        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1)

        assert frame.width_raw == 200

    def test_height_raw_from_raw(self):
        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1)

        assert frame.height_raw == 100

    def test_width_raw_grayscale(self):
        raw = np.zeros((100, 200), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1)

        assert frame.width_raw == 200

    def test_height_raw_grayscale(self):
        raw = np.zeros((100, 200), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1)

        assert frame.height_raw == 100


class TestFrameWidthHeightRescaled:
    def test_width_rescaled_from_rescale(self):
        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        rescale = np.zeros((50, 100, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1, rescale=rescale)

        assert frame.width_rescaled == 100

    def test_height_rescaled_from_rescale(self):
        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        rescale = np.zeros((50, 100, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1, rescale=rescale)

        assert frame.height_rescaled == 50

    def test_width_rescaled_from_crop_when_no_rescale(self):
        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        crop = np.zeros((50, 100, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1, crop=crop)

        assert frame.width_rescaled == 100

    def test_height_rescaled_from_crop_when_no_rescale(self):
        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        crop = np.zeros((50, 100, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1, crop=crop)

        assert frame.height_rescaled == 50

    def test_width_rescaled_from_raw_when_no_crop_or_rescale(self):
        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1)

        assert frame.width_rescaled == 200

    def test_height_rescaled_from_raw_when_no_crop_or_rescale(self):
        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1)

        assert frame.height_rescaled == 100

    def test_width_rescaled_prioritizes_rescale_over_crop(self):
        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        crop = np.zeros((50, 100, 3), dtype=np.uint8)
        rescale = np.zeros((25, 50, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1, crop=crop, rescale=rescale)

        assert frame.width_rescaled == 50

    def test_height_rescaled_prioritizes_rescale_over_crop(self):
        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        crop = np.zeros((50, 100, 3), dtype=np.uint8)
        rescale = np.zeros((25, 50, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1, crop=crop, rescale=rescale)

        assert frame.height_rescaled == 25


class TestFrameWidthHeightDeprecated:
    def test_width_deprecated_returns_raw(self):
        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            width = frame.width

            assert width == 200.0
            assert len(w) == 1
            assert issubclass(w[0].category, FutureWarning)
            assert "Frame.width is deprecated" in str(w[0].message)

    def test_height_deprecated_returns_raw(self):
        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            height = frame.height

            assert height == 100.0
            assert len(w) == 1
            assert issubclass(w[0].category, FutureWarning)
            assert "Frame.height is deprecated" in str(w[0].message)

    def test_width_deprecated_with_crop_and_rescale(self):
        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        crop = np.zeros((50, 100, 3), dtype=np.uint8)
        rescale = np.zeros((25, 50, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1, crop=crop, rescale=rescale)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            width = frame.width

            assert width == 200.0
            assert len(w) == 1
            assert issubclass(w[0].category, FutureWarning)

    def test_height_deprecated_with_crop_and_rescale(self):
        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        crop = np.zeros((50, 100, 3), dtype=np.uint8)
        rescale = np.zeros((25, 50, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1, crop=crop, rescale=rescale)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            height = frame.height

            assert height == 100.0
            assert len(w) == 1
            assert issubclass(w[0].category, FutureWarning)


class TestCropBbox:
    def test_crop_bbox_stored_correctly(self):
        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        bbox = BBox(50, 25, 150, 75)
        frame = Frame(raw=raw, frame_no=1, crop_bbox=bbox)

        assert frame.crop_bbox == bbox
        assert frame.crop_bbox.x1 == 50
        assert frame.crop_bbox.y1 == 25
        assert frame.crop_bbox.x2 == 150
        assert frame.crop_bbox.y2 == 75

    def test_crop_bbox_maps_crop_to_raw(self):
        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        bbox = BBox(50, 25, 150, 75)
        crop = raw[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]
        frame = Frame(raw=raw, frame_no=1, crop=crop, crop_bbox=bbox)

        assert frame.crop is not None
        assert frame.crop.shape[0] == bbox.height
        assert frame.crop.shape[1] == bbox.width


class TestHandlerChainIntegration:
    def test_motion_handler_preserves_crop_rescale(self):
        from wildcamtools.lib.motion import MogMotion

        raw = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        crop = np.zeros((50, 100, 3), dtype=np.uint8)
        rescale = np.zeros((25, 50, 3), dtype=np.uint8)
        bbox = BBox(50, 25, 150, 75)

        frame = Frame(raw=raw, frame_no=10, crop=crop, rescale=rescale, crop_bbox=bbox)

        motion_handler = MogMotion(history=1, threshold=16, detect_shadows=False, kernel_size=3)
        result = motion_handler.handle(frame)

        assert result.crop is crop
        assert result.rescale is rescale
        assert result.crop_bbox == bbox
        assert result.frame_no == 10

    def test_filter_ssim_uses_output_property(self):
        from wildcamtools.lib.frames import FilterSSIM

        raw1 = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        crop1 = np.zeros((50, 100, 3), dtype=np.uint8)
        frame1 = Frame(raw=raw1, frame_no=1, crop=crop1)

        raw2 = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        crop2 = np.zeros((50, 100, 3), dtype=np.uint8)
        frame2 = Frame(raw=raw2, frame_no=2, crop=crop2)

        filter_ssim = FilterSSIM(similarity_minimum=0.5)
        filter_ssim.handle(frame1)
        filter_ssim.handle(frame2)

        assert frame1.filter_keep is True
        assert filter_ssim.frame_previous_interesting is not None

    def test_frame_image_writer_writes_output(self, tmp_path):
        from wildcamtools.lib.frames import FrameImageWriter

        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        crop = np.ones((50, 100, 3), dtype=np.uint8) * 128
        frame = Frame(raw=raw, frame_no=1, crop=crop)

        writer = FrameImageWriter(tmp_path)
        result = writer.handle(frame)

        assert result.filter_keep is True
        written_file = tmp_path / "frame_00001.jpg"
        assert written_file.exists()

        import cv2

        written_image = cv2.imread(str(written_file))
        assert written_image is not None
        assert written_image.shape[0] == 50
        assert written_image.shape[1] == 100

    def test_frame_image_writer_writes_tiles(self, tmp_path):
        from wildcamtools.lib.frames import FrameImageWriter, FrameTiler

        raw = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1)

        tiler = FrameTiler(cols=2, rows=2)
        frame = tiler.handle(frame)

        writer = FrameImageWriter(tmp_path)
        result = writer.handle(frame)

        assert result.filter_keep is True
        assert (tmp_path / "frame_00001_tile_0_0.jpg").exists()
        assert (tmp_path / "frame_00001_tile_0_1.jpg").exists()
        assert (tmp_path / "frame_00001_tile_1_0.jpg").exists()
        assert (tmp_path / "frame_00001_tile_1_1.jpg").exists()

        import cv2

        for row in range(2):
            for col in range(2):
                tile_file = tmp_path / f"frame_00001_tile_{row}_{col}.jpg"
                tile_image = cv2.imread(str(tile_file))
                assert tile_image is not None
                assert tile_image.shape[0] == 50
                assert tile_image.shape[1] == 100

    def test_rescaler_uses_output_property(self):
        from wildcamtools.lib.frames import Rescaler
        from wildcamtools.lib.stats import Colourspace, VideoStats

        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        crop = np.zeros((50, 100, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1, crop=crop)

        stats = VideoStats(x=200, y=100, fps=30.0, frame_count=30, colourspace=Colourspace.RGB)
        rescaler = Rescaler(stats=stats, x=50, y=25)
        result = rescaler.handle(frame)

        assert result.rescale is not None
        assert result.rescale.shape[0] == 25
        assert result.rescale.shape[1] == 50

    def test_motion_flow_highlighter_preserves_crop_rescale(self):
        from wildcamtools.lib.frames import MotionFlowHighlighter

        raw = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        crop = np.zeros((50, 100, 3), dtype=np.uint8)
        rescale = np.zeros((25, 50, 3), dtype=np.uint8)
        bbox = BBox(50, 25, 150, 75)

        frame1 = Frame(raw=raw, frame_no=1, crop=crop, rescale=rescale, crop_bbox=bbox)
        frame2 = Frame(raw=raw, frame_no=2, crop=crop, rescale=rescale, crop_bbox=bbox)

        highlighter = MotionFlowHighlighter(alpha=0.5, max_magnitude=10.0)
        highlighter.handle(frame1)
        result = highlighter.handle(frame2)

        assert result.crop is crop
        assert result.rescale is rescale
        assert result.crop_bbox == bbox

    def test_output_property_with_grayscale(self):
        raw = np.zeros((100, 200), dtype=np.uint8)
        crop = np.zeros((50, 100), dtype=np.uint8)
        rescale = np.zeros((25, 50), dtype=np.uint8)

        frame = Frame(raw=raw, frame_no=1, crop=crop, rescale=rescale)

        assert frame.output is rescale
        assert frame.width_rescaled == 50
        assert frame.height_rescaled == 25


class TestFrameTiling:
    def test_frame_tiling_fields_default_to_none(self):
        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1)

        assert frame.tiles is None
        assert frame.tiling_cols is None
        assert frame.tiling_rows is None
        assert frame.tiling_width is None
        assert frame.tiling_height is None

    def test_get_tile_returns_none_when_no_tiles(self):
        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1)

        assert frame.get_tile(0, 0) is None

    def test_get_tile_returns_none_for_invalid_coordinates(self):
        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        frame = Frame(
            raw=raw, frame_no=1, tiles=[raw], tiling_cols=1, tiling_rows=1, tiling_width=200, tiling_height=100
        )

        assert frame.get_tile(-1, 0) is None
        assert frame.get_tile(0, -1) is None
        assert frame.get_tile(1, 0) is None
        assert frame.get_tile(0, 1) is None
        assert frame.get_tile(5, 5) is None

    def test_get_tile_returns_correct_tile(self):
        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        tile1 = np.ones((50, 100, 3), dtype=np.uint8)
        tile2 = np.ones((50, 100, 3), dtype=np.uint8) * 2
        tiles = [tile1, tile2]

        frame = Frame(
            raw=raw, frame_no=1, tiles=tiles, tiling_cols=2, tiling_rows=1, tiling_width=100, tiling_height=50
        )

        assert np.array_equal(frame.get_tile(0, 0), tile1)
        assert np.array_equal(frame.get_tile(1, 0), tile2)


class TestFrameTiler:
    def test_frame_tiler_creation(self):
        from wildcamtools.lib.frames import FrameTiler

        tiler = FrameTiler(cols=3, rows=2)
        assert tiler.cols == 3
        assert tiler.rows == 2

    def test_frame_tiler_default_values(self):
        from wildcamtools.lib.frames import FrameTiler

        tiler = FrameTiler()
        assert tiler.cols == 2
        assert tiler.rows == 2

    def test_frame_tiler_invalid_cols(self):
        import pytest

        from wildcamtools.lib.frames import FrameTiler

        with pytest.raises(ValueError, match="cols must be at least 1"):
            FrameTiler(cols=0)

        with pytest.raises(ValueError, match="cols must be at least 1"):
            FrameTiler(cols=-1)

    def test_frame_tiler_invalid_rows(self):
        import pytest

        from wildcamtools.lib.frames import FrameTiler

        with pytest.raises(ValueError, match="rows must be at least 1"):
            FrameTiler(rows=0)

        with pytest.raises(ValueError, match="rows must be at least 1"):
            FrameTiler(rows=-1)

    def test_frame_tiler_splits_frame_into_tiles(self):
        from wildcamtools.lib.frames import FrameTiler

        raw = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1)

        tiler = FrameTiler(cols=2, rows=2)
        result = tiler.handle(frame)

        assert result.tiles is not None
        assert len(result.tiles) == 4
        assert result.tiling_cols == 2
        assert result.tiling_rows == 2
        assert result.tiling_width == 100
        assert result.tiling_height == 50

    def test_frame_tiler_tiles_cover_image(self):
        from wildcamtools.lib.frames import FrameTiler

        raw = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1)

        tiler = FrameTiler(cols=2, rows=2)
        result = tiler.handle(frame)

        assert result.tiles is not None
        for tile in result.tiles:
            assert tile.shape[0] == 50
            assert tile.shape[1] == 100
            assert tile.shape[2] == 3

    def test_frame_tiler_get_tile_access(self):
        from wildcamtools.lib.frames import FrameTiler

        raw = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1)

        tiler = FrameTiler(cols=2, rows=2)
        result = tiler.handle(frame)

        assert result.get_tile(0, 0) is not None
        assert result.get_tile(1, 0) is not None
        assert result.get_tile(0, 1) is not None
        assert result.get_tile(1, 1) is not None

    def test_frame_tiler_single_tile(self):
        from wildcamtools.lib.frames import FrameTiler

        raw = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1)

        tiler = FrameTiler(cols=1, rows=1)
        result = tiler.handle(frame)

        assert result.tiles is not None
        assert len(result.tiles) == 1
        assert result.tiling_cols == 1
        assert result.tiling_rows == 1
        assert result.tiling_width == 200
        assert result.tiling_height == 100
        assert np.array_equal(result.tiles[0], raw)

    def test_frame_tiler_uses_output_property(self):
        from wildcamtools.lib.frames import FrameTiler

        raw = np.zeros((100, 200, 3), dtype=np.uint8)
        crop = np.random.randint(0, 255, (50, 100, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1, crop=crop)

        tiler = FrameTiler(cols=2, rows=2)
        result = tiler.handle(frame)

        assert result.tiles is not None
        assert len(result.tiles) == 4
        for tile in result.tiles:
            assert tile.shape[2] == 3

    def test_frame_tiler_non_divisible_dimensions(self):
        from wildcamtools.lib.frames import FrameTiler

        raw = np.random.randint(0, 255, (101, 201, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1)

        tiler = FrameTiler(cols=2, rows=2)
        result = tiler.handle(frame)

        assert result.tiles is not None
        assert len(result.tiles) == 4
        assert all(tile.shape == result.tiles[0].shape for tile in result.tiles)
        assert result.tiles[0].shape == (50, 100, 3)
        assert result.tiling_width == 100
        assert result.tiling_height == 50

    def test_frame_tiler_covers_entire_image(self):
        from wildcamtools.lib.frames import FrameTiler

        raw = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1)

        tiler = FrameTiler(cols=3, rows=2)
        result = tiler.handle(frame)

        assert result.tiles is not None
        assert len(result.tiles) == 6
        assert all(tile.shape == result.tiles[0].shape for tile in result.tiles)

        tile_h, tile_w = result.tiles[0].shape[:2]
        step_y = int(tile_h * (1 - 0))
        step_x = int(tile_w * (1 - 0))

        last_tile_start_y = raw.shape[0] - tile_h
        last_tile_start_x = raw.shape[1] - tile_w
        assert last_tile_start_y >= step_y
        assert last_tile_start_x >= step_x * 2
        assert last_tile_start_y + tile_h == raw.shape[0]
        assert last_tile_start_x + tile_w == raw.shape[1]

    def test_frame_tiler_asymmetric_grid(self):
        from wildcamtools.lib.frames import FrameTiler

        raw = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1)

        tiler = FrameTiler(cols=3, rows=2)
        result = tiler.handle(frame)

        assert result.tiles is not None
        assert len(result.tiles) == 6
        assert result.tiling_cols == 3
        assert result.tiling_rows == 2
        assert result.tiling_width == 66
        assert result.tiling_height == 50

    def test_frame_tiler_overlap_default(self):
        from wildcamtools.lib.frames import FrameTiler

        raw = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1)

        tiler = FrameTiler(cols=2, rows=2)
        result = tiler.handle(frame)

        assert result.tiles is not None
        assert len(result.tiles) == 4
        assert result.tiles[0].shape == (50, 100, 3)

    def test_frame_tiler_overlap_fifty_percent(self):
        from wildcamtools.lib.frames import FrameTiler

        raw = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1)

        tiler = FrameTiler(cols=2, rows=2, overlap=0.5)
        result = tiler.handle(frame)

        assert result.tiles is not None
        assert len(result.tiles) == 4
        assert all(tile.shape == result.tiles[0].shape for tile in result.tiles)
        assert result.tiles[0].shape[0] > 50
        assert result.tiles[0].shape[1] > 100

    def test_frame_tiler_overlap_coverage(self):
        from wildcamtools.lib.frames import FrameTiler

        raw = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        frame = Frame(raw=raw, frame_no=1)

        tiler = FrameTiler(cols=3, rows=2, overlap=0.5)
        result = tiler.handle(frame)

        assert result.tiles is not None
        assert len(result.tiles) == 6
        assert all(tile.shape == result.tiles[0].shape for tile in result.tiles)

        tile_h, tile_w = result.tiles[0].shape[:2]
        last_tile_start_y = raw.shape[0] - tile_h
        last_tile_start_x = raw.shape[1] - tile_w

        assert last_tile_start_y + tile_h == raw.shape[0]
        assert last_tile_start_x + tile_w == raw.shape[1]

    def test_frame_tiler_overlap_invalid(self):
        import pytest

        from wildcamtools.lib.frames import FrameTiler

        with pytest.raises(ValueError, match=r"overlap must be between 0\.0 and 1\.0"):
            FrameTiler(cols=2, rows=2, overlap=-0.1)

        with pytest.raises(ValueError, match=r"overlap must be between 0\.0 and 1\.0"):
            FrameTiler(cols=2, rows=2, overlap=1.0)

        with pytest.raises(ValueError, match=r"overlap must be between 0\.0 and 1\.0"):
            FrameTiler(cols=2, rows=2, overlap=1.5)
