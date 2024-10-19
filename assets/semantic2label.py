# Unity 数据集的语义类别
# SEMANTIC_TO_LABEL = {
#     "Wall":1,
#     "Chair":2,
#     "Table":3,
#     "Shelf":4,
#     "Bed":5,
#     "Window":6,
#     "Stove":7,
#     "Television":8,
#     "Picture":9,
#     "Bathtub":10,
#     "Toilet":11
# }

# 自己选择的语义物体类别和标签的对应关系
SEMANTIC_TO_LABEL = {
    'wall':1,
    'cabinet':2, # 橱柜
    'night stand':2, # 床头柜
    'shelves':2, # 架子
    'window':3,
    'curtain':3,
    'door':4,
    'chair':5,
    'picture':6,
    'sofa':7,
    'table':8,
    'desk':8,
    'bed':9,
    'pillow':9,
    'television':10,
    'lamp':11,
    'fridge':12,
    'mirror':13
}

LABEL_TO_COLOR = {
    0: (255, 255, 255),
    1: (0, 0, 0),
    2: [255, 125, 0],
    3: [251, 247, 0],
    4: [133, 255, 0],
    5: [7, 255, 0],
    6: [0, 255, 117],
    7: [0, 255, 243],
    8: [0, 141, 255],
    9: [0, 15, 255],
    10: [109, 0, 255],
    11: [235, 0, 255],
    12: [255, 0, 149],
    13: [255, 0, 23],
}

# Structured3D 中颜色和语义类别的对应关系
COLOR_TO_SEMANTIC = {
    (0, 0, 0): "unknown",
    (174, 199, 232): "wall",
    (152, 223, 138): "floor",
    (31, 119, 180): "cabinet",
    (255, 187, 120): "bed",
    (188, 189, 34): "chair",
    (140, 86, 75): "sofa",
    (255, 152, 150): "table",
    (214, 39, 40): "door",
    (197, 176, 213): "window",
    (148, 103, 189): "bookshelf",
    (196, 156, 148): "picture",
    (23, 190, 207): "counter",
    (178, 76, 76): "blinds",
    (247, 182, 210): "desk",
    (66, 188, 102): "shelves",
    (219, 219, 141): "curtain",
    (140, 57, 197): "dresser",
    (202, 185, 52): "pillow",
    (51, 176, 203): "mirror",
    (200, 54, 131): "floor mat",
    (92, 193, 61): "clothes",
    (78, 71, 183): "ceiling",
    (172, 114, 82): "books",
    (255, 127, 14): "fridge",
    (91, 163, 138): "television",
    (153, 98, 156): "paper",
    (140, 153, 101): "towel",
    (158, 218, 229): "shower curtain",
    (100, 125, 154): "box",
    (178, 127, 135): "whiteboard",
    (120, 185, 128): "person",
    (146, 111, 194): "night stand",
    (44, 160, 44): "toilet",
    (112, 128, 144): "sink",
    (96, 207, 209): "lamp",
    (227, 119, 194): "bathtub",
    (213, 92, 176): "bag",
    (94, 106, 211): "structure",
    (82, 84, 163): "furniture",
    (100, 85, 144): "prop"
}

SEMANTIC_TO_COLOR = {value: key for key, value in COLOR_TO_SEMANTIC.items()}
