import logging
import math

from wildcamtools.lib import BBox, Frame
from wildcamtools.lib.ai.llm.abstract import AbstractLlm
from wildcamtools.lib.ai.types import ResultList

logger = logging.getLogger(__name__)


class AICropFinder:
    analyser: AbstractLlm
    detections: ResultList | None = None
    expansion: float = 0.25

    def __init__(
        self,
        analyser: AbstractLlm,
        expansion: float,
    ):
        self.analyser = analyser
        self.expansion = expansion

    def handle(self, frame: Frame) -> Frame:
        if self.detections is None:
            raise RuntimeError("detections must be set before handling")
        if len(self.detections.results) == 0:
            frame.filter_keep = False
            return frame
        # TODO handle multiple detections per frame
        if len(self.detections.results) > 1:
            logger.warning("Multiple species detected (%d), using only the first result", len(self.detections.results))
        for frameresult in self.detections.results[0].frames:
            if frameresult.frame_no == frame.frame_no:
                h, w, _ = frame.raw.shape
                result_bbox = BBox(
                    min(frameresult.left, frameresult.right),
                    min(frameresult.bottom, frameresult.top),
                    max(frameresult.left, frameresult.right),
                    max(frameresult.bottom, frameresult.top),
                )
                frame.crop_bbox = self.expand_bbox(result_bbox, w, h)
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

    def expand_bbox(self, bbox: BBox, max_width: int, max_height: int) -> BBox:
        # convert proportional bounding box into pixels
        x1 = math.floor(max_width * bbox.x1)
        x2 = math.ceil(max_width * bbox.x2)
        y1 = math.floor(max_height * bbox.y1)
        y2 = math.ceil(max_height * bbox.y2)
        # enlarge crop box within screen
        box_w = x2 - x1
        box_h = y2 - y1
        x1 = max(0, x1 - int(box_w * self.expansion))
        x2 = min(max_width, x2 + int(box_w * self.expansion))
        y1 = max(0, y1 - int(box_h * self.expansion))
        y2 = min(max_height, y2 + int(box_h * self.expansion))

        return BBox(x1, y1, x2, y2)
