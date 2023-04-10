from enum import Enum
from typing import Tuple

class TagType(Enum):
    UofA = 'UofA Tag'
    StopSign = "Stop Sign"
    TIntersection = "T-Intersection"
    ForwardStop = "Stop sign indicating forward traffic"
    RightStop = "Stop sign indicating right-turning traffic"
    LeftStop = "Stop sign indicating left-turning traffic"
    CrossingStop = "Duckie crossing stop sign"
    ParkingLotEnteringStop = "Stop sign that leads into the parking lot"


class Tag:
    def __init__(self, id: int, type: TagType, label: int):
        self.id = id
        self.type = type
        self.label = label
        self.color = TAG_TYPE_TO_COLOR[type]


TAG_TYPE_TO_COLOR = {
    TagType.StopSign: (255, 0, 0),
    TagType.TIntersection: (0, 0, 255),
    TagType.UofA: (0, 255, 0),
    None: (255, 0, 255),
}

TAG_ID_TO_TAG = {
    162: Tag(162, TagType.StopSign, 0),
    169: Tag(169, TagType.StopSign, 0),
    58: Tag(58, TagType.TIntersection, 0),
    62: Tag(62, TagType.TIntersection, 0),
    63: Tag(63, TagType.TIntersection, 0),
    153: Tag(153, TagType.TIntersection, 0),
    133: Tag(133, TagType.TIntersection, 0),
    143: Tag(143, TagType.TIntersection, 0),
    93: Tag(93, TagType.UofA, 0),
    94: Tag(94, TagType.UofA, 0),
    200: Tag(200, TagType.UofA, 0),
    201: Tag(201, TagType.UofA, 0),
    # For final
    48: Tag(48, TagType.RightStop, 1),
    50: Tag(56, TagType.LeftStop, 2),
    56: Tag(56, TagType.ForwardStop, 3),
    163: Tag(163, TagType.CrossingStop, 4),
    38: Tag(38, TagType.ParkingLotEnteringStop, 5),
}
