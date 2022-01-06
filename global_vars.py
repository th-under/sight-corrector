# Parameters you might want to adjust

TWEAK_DENS = 0.05 # head orientation mesh densitiy (0:100 %) used to replace eyes. 
MAP_TOL = 0.2 # tolerance up to which eye mapping should be performed (eukl. distance)
DAMPING = 5 # Value representing positional changes below which tweak updates are reduced for reducing flickering
BTOL = 1.01 # rel. difference in brightness above which eyes should not be tweaked

DEVICE_IN = '/dev/video0' # For windows or macOS use DEVICE_IN = 0
MODE = 'demo' # or 'live'


# label numbers of anatomical landmarks (mediapipe specific, don't change)

# left / right always refers to anatomical side (not to visual side)
# EYE_CIRCLE_R = [133, 173, 157, 158, 159, 160, 161, 246,  33,   7, 163, 144, 145, 153, 154, 155] # first (most inner) ring
# EYE_CIRCLE_R = [243, 190,  56,  28,  27,  29,  30, 247, 130,  25, 110,  24,  23,  22,  26, 112] # 2nd ring
EYE_CIRCLE_R = [244, 189, 221, 222, 223, 224, 225, 113, 226,  31, 228, 229, 230, 231, 232, 233] # 3rd ring

# EYE_CIRCLE_L = [463, 414, 286, 258, 257, 259, 260, 467, 359, 255, 339, 254, 253, 252, 256, 341] # 2nd ring
# EYE_CIRCLE_L = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382] # first (most inner) ring
EYE_CIRCLE_L = [464, 413, 441, 442, 443, 444, 445, 342, 446, 261, 448, 449, 450, 451, 452, 453] # 3rd ring

# EYE_R = [130, 243, 27, 23] # = indices 8, 0, 4, 12, lateral, medial, upper, lower point
# EYE_L = [359, 463, 257, 253] 

FACE = [1, 10, 152, 234, 454] # nose, forehead, chin, right, left most point


