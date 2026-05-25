import logging
import math
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from wildcamtools.lib import BBox, Frame
from wildcamtools.lib.ai import ResultList
from wildcamtools.lib.ai.llm.abstract import AbstractLlm

logger = logging.getLogger(__name__)


class AICropFinder:
    DETECTION_PROMPT = """These are images from a video taken in a UK garden near a river.
Identify any animals you are highly confident of in the images.
Return JSON only with this exact structure:
{"results": [{"species_name": "string", "frames": [{"frame_no":0,"left": 0.0, "right": 1.0, "top": 1.0, "bottom": 0.0}]}]}
Note that the bounding box coordinates are proportional to the image dimensions and therefore must be between 0.0 and 1.0.
If no animals are detected, return {"results": []}."""

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

    def run_detection(self, images: Sequence[Path]) -> None:
        """Run AI detection on low-resolution frames."""
        logger.info("Starting detection on %d low-res frames", len(images))

        try:
            self.detections = self.analyser.message_with_schema(
                message=self.DETECTION_PROMPT,
                images=sorted(images),
                response_class=ResultList,
            )
        except ValidationError:
            logger.exception("Unable to validate")
            self.detections = ResultList(results=[])

        logger.info("Detection complete: found %d species results", len(self.detections.results))

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
                        "Invalid bounding box detected on frame %d : (%d,%d)-(%d,%d)",
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
