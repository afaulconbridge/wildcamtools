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
