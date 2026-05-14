import logging
import math
from collections.abc import Sequence
from pathlib import Path

from wildcamtools.lib import BBox, Frame
from wildcamtools.lib.ai import AbstractAnalyser, ResultList

logger = logging.getLogger(__name__)


class AICropFinder:
    analyser: AbstractAnalyser
    detections: ResultList | None = None
    expansion: float = 0.25

    def __init__(
        self,
        analyser: AbstractAnalyser,
        expansion: float,
    ):
        self.analyser = analyser
        self.expansion = expansion

    def run_detection(self, images: Sequence[Path]) -> None:
        """Run AI detection on low-resolution frames."""
        logger.info("Starting detection on %d low-res frames", len(images))

        self.detections = self.analyser.detect(images)

        logger.info("Detection complete: found %d species results", len(self.detections.results))

    def handle(self, frame: Frame) -> Frame:
        if self.detections is None:
            raise RuntimeError("detections must be set before handling")
        if len(self.detections.results) == 0:
            frame.filter_keep = False
            return frame
        # TODO handle multiple species
        # TODO handle multiple detections per frame
        if len(self.detections.results) > 1:
            raise NotImplementedError()
        for frameresult in self.detections.results[0].frames:
            if frameresult.frame_no == frame.frame_no:
                h, w, _ = frame.raw.shape
                x1 = math.floor(w * min(frameresult.left, frameresult.right))
                x2 = math.ceil(w * max(frameresult.left, frameresult.right))
                y1 = math.floor(h * min(frameresult.bottom, frameresult.top))
                y2 = math.ceil(h * max(frameresult.bottom, frameresult.top))
                # enlarge crop box within screen
                box_w = x2 - x1
                box_h = y2 - y1
                x1 = max(0, x1 - int(box_w * self.expansion))
                x2 = min(w, x2 + int(box_w * self.expansion))
                y1 = max(0, y1 - int(box_h * self.expansion))
                y2 = min(h, y2 + int(box_h * self.expansion))

                # double check for validity
                if x1 >= x2 or y1 >= y2:
                    logger.warning(
                        "Invalid bounding box detected on %s : (%s,%s)-(%s,%s)",
                        frame.frame_no,
                        x1,
                        y1,
                        x2,
                        y2,
                    )
                    continue

                frame.crop_bbox = BBox(x1, y1, x2, y2)
                logger.debug(
                    "Crop %s to (%s,%s)-(%s,%s)",
                    frame.frame_no,
                    frame.crop_bbox.x1,
                    frame.crop_bbox.y1,
                    frame.crop_bbox.x2,
                    frame.crop_bbox.y2,
                )
                frame.crop = frame.raw[frame.crop_bbox.y1 : frame.crop_bbox.y2, frame.crop_bbox.x1 : frame.crop_bbox.x2]
                return frame
        frame.filter_keep = False
        return frame
