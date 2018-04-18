# Import libraries
import utility_functions_5 as util
from CellTracks5_1 import CellTracks
import numpy as np
import pickle

# Set paths to data and to trained model
trial = 7
model_path = "./prob_model.p"
color_path = "../normalized_frames/video_" + format(trial) + "/Raw/"
seg_path = "../normalized_frames/video_" + format(trial) + "/Binary/"
out_path = "./output_" + format(trial) + "/"

# Set parameters
search_dist = 60 # Max distance moved per frame
# A = image_width*image_height # Image area
# PM = .8
min_area = 15
std_dev_mult = 2*np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
# Features:
weights = np.array([500, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# Load color images and segmentations
video_img, bin_mask = util.load_testing_data(color_path, seg_path)

# Load trained model
load_data = pickle.load(open(model_path, "rb"))
feat_std_dev = np.multiply(std_dev_mult, load_data[0])
count_lr = load_data[1]

del model_path, color_path, seg_path, load_data

# Get image dimensions
image_height = video_img.shape[1]
image_width = video_img.shape[2]

# Generate graphical model of detections
cell_tracks = CellTracks(video_img=video_img, bin_mask=bin_mask, feat_std_dev=feat_std_dev, count_lr=count_lr,
                         weights=weights, search_dist=search_dist, min_area=min_area)

# VITERBI
cell_tracks.run_viterbi()

# DRAW GRAPH REPRESENTATION OF TRACKS
# cell_tracks.draw_graph(False)

# GENERATE TRACKING IMAGES/VIDEO
cell_tracks.gen_images_reverse(out_path)
# cell_tracks.gen_images()

# SAVE GRAPH OBJECT
pickle.dump(cell_tracks, open("tracks_" + format(trial) + ".p", "wb"))

print "fin."