from enum import Enum
from typing import Tuple

class TagType(Enum):
    UofA = 'UofA Tag'
    StopSign = "Stop Sign"
    TIntersection = "T-Intersection"


class Tag:
    def __init__(self, id: int, type: TagType):
        self.id = id
        self.type = type
        self.color = TAG_TYPE_TO_COLOR[type]


TAG_TYPE_TO_COLOR = {
    TagType.StopSign: (255, 0, 0),
    TagType.TIntersection: (0, 0, 255),
    TagType.UofA: (0, 255, 0),
    None: (255, 0, 255),
}

TAG_ID_TO_TAG = {
    162: Tag(162, TagType.StopSign),
    169: Tag(169, TagType.StopSign),
    58: Tag(58, TagType.TIntersection),
    62: Tag(62, TagType.TIntersection),
    63: Tag(63, TagType.TIntersection),
    153: Tag(153, TagType.TIntersection),
    133: Tag(133, TagType.TIntersection),
    143: Tag(143, TagType.TIntersection),
    93: Tag(93, TagType.UofA),
    94: Tag(94, TagType.UofA),
    200: Tag(200, TagType.UofA),
    201: Tag(201, TagType.UofA),
}
