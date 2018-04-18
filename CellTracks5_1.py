import utility_functions_5 as util
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import cv2


class CellTracks:

    # Constructor - calculates features for each detection and initializes graphical model
    def __init__(self, video_img, bin_mask, feat_std_dev, count_lr, weights, search_dist, min_area):
        self.feat_std_dev = feat_std_dev
        self.count_lr = count_lr
        self.video_img = video_img
        self.search_dist = search_dist
        self.weights = weights
        # Calculate vector of features for each image
        feats, self.contours, centroid_vect, self.mean_area = util.gen_features(video_img, bin_mask, min_area)
        print "Calculated feature vector for all nodes..."
        # Initialize nodes in graph
        self.G = nx.DiGraph()
        # Add a node for each detection
        self.img_nodes = []
        for i in range(0, len(feats)):
            curr_nodes = []
            for d in range(0, feats[i].shape[0]):
                # centroid = feats[i][0:2, d]
                feats_node = feats[i][d, :]
                centroid_node = centroid_vect[i][d]
                curr_nodes.append(self.G.number_of_nodes())
                self.G.add_node(self.G.number_of_nodes(), feats=feats_node, centroid=centroid_node, count=0)
            self.img_nodes.append(curr_nodes)
        self.num_frames = len(self.img_nodes)
        print "Initialized graphical model..."

    # Run the viterbi algorithm to generate optimal tracks in a graph
    def run_viterbi(self):
        # Loop through frames at which to start tracks
        for track_start_frame in range(0, self.num_frames - 1):
            print "Finding tracks starting in frame " + format(track_start_frame)
            score_increased = True
            # Add tracks from this frame while they increase the score
            while score_increased:
                score_increased = self.add_tracks_frame(start_frame=track_start_frame)
            # self.draw_graph(False)
        print "dun"

    # Add tracks starting from the specified frame
    def add_tracks_frame(self, start_frame):
        # Initialize variables
        current_frame = start_frame + 1
        last_frame = False
        score_increased = True
        # Vector to track total scores of tracks being explored
        track_score_sum = np.zeros(len(self.img_nodes[start_frame]))
        # Store back-tracking values to generate tracks
        back_track_scores = []
        back_track_vars = []
        # Find probabilistically most-likely track starting in "start_frame"
        while not last_frame and score_increased:
            if not current_frame % 50:
                print "Frame: " + format(current_frame)
            current_scores, current_vars = self.frame_events(current_frame)
            # Update total track scores
            prev_score_sum = track_score_sum.copy()
            track_score_sum = np.empty(len(current_vars))
            # Loop through back-track nodes
            for i in range(0, len(current_vars)):
                if "prev_node" in current_vars[i].keys():
                    track_score_sum[i] = prev_score_sum[current_vars[i]['prev_node']] + current_scores[i]
                else:
                    track_score_sum[i] = -float('Inf')
            # Update loop booleans and overall track variables
            if np.max(track_score_sum) <= 0:
                # No increasing-score tracks have been found if next frame == start_frame + 1 (aka first iteration)
                score_increased = not current_frame == (start_frame + 1)
                # Tracks up to the previous frame are positive if score_increased
                current_frame -= 1
                track_score_sum = prev_score_sum
                break
            else:
                # Update frame
                current_frame += 1
                # Loop booleans
                score_increased = True
                last_frame = current_frame == self.num_frames
                if last_frame:
                    current_frame -= 1
                # Overall track variables
                back_track_scores.append(current_scores)
                back_track_vars.append(current_vars)
        # Add track with highest score if it increases the overall score
        if score_increased:
            print "Adding track..."
            self.back_track(back_track_vars, track_score_sum, current_frame)
        return score_increased

    # Get max-scoring events and event vars for nodes in current frame
    def frame_events(self, current_frame):
        # Loop through detections in current frame
        current_scores = []
        current_vars = []
        current_nodes = self.img_nodes[current_frame]
        # For each node in next frame, find max prob event
        for node_ind in range(0, len(current_nodes)):
            # For each, find highest-prob migration from detection in previous frame
            mig_prob, event_vars = self.max_prob_event(current_frame, node_ind)
            # Track nodes
            current_scores.append(mig_prob)
            current_vars.append(event_vars)
        return current_scores, current_vars

    # Get maximum probability event for node in current frame from previous frame
    def max_prob_event(self, frame, node_ind):
        node = self.img_nodes[frame][node_ind]
        # Find cells within search dist
        prev_cells = self.cells_within_search_dist(node, frame)
        # Find the one with the highest probability of migration
        if not len(prev_cells):
            # Return high negative prob migration
            mig_prob = -float('Inf')
            event_vars = {}
        else:
            mig_probs = []
            for i in range(0, len(prev_cells)):
                prev_node = self.img_nodes[frame-1][prev_cells[i]]
                mig_probs.append(self.prob_migration(prev_node, node))
            max_ind = np.argmax(mig_probs)
            mig_prob = mig_probs[max_ind]
            prev_cell = prev_cells[max_ind]
            event_vars = {"prev_node": prev_cell, "mig_prob": mig_prob}
        return mig_prob, event_vars

    # Finds the nodes within max_dist of the current node in the previous frame
    def cells_within_search_dist(self, current_node, current_frame):
        # Initialize output list
        possible_prev_cells = []
        current_node_pos = self.G.node[current_node]['centroid']
        # Loop through detections in the previous frame and check distance from current node
        for detection_ind in range(0, len(self.img_nodes[current_frame - 1])):
            prev_node = self.img_nodes[current_frame - 1][detection_ind]
            prev_node_pos = self.G.node[prev_node]['centroid']
            distance = np.linalg.norm(current_node_pos - prev_node_pos)
            # Add to possible_next_cells if distance < search_dist
            if distance < self.search_dist:
                possible_prev_cells.append(detection_ind)
        return np.array(possible_prev_cells)

    # Returns the probability score of a migration between two cells
    def prob_migration(self, prev_node, current_node):
        prob = 0
        # If edge doesn't already exist
        if not self.G.has_edge(prev_node, current_node):
            # Calculate edge probability
            edge_prob = self.edge_score(prev_node, current_node)
            # Check if current_node already has an edge going to it
            # If so, get edge prob_migration
            in_edges = list(self.G.in_edges(current_node))
            if len(in_edges) == 1:
                edge = in_edges[0]
                prob_current = self.edge_score(edge[0], edge[1])
                # prob_current = self.G.get_edge_data(edge[0], edge[1])['prob_migration']
                prob = edge_prob - prob_current
            elif len(in_edges) == 0:
                # Calculate probability of count increase for current node
                count_up_prob = self.prob_count_up(current_node)
                # Calculate total migration probability and return
                prob = edge_prob + count_up_prob
            else:
                print "ERROR TOO MANY IN-EDGES"
        return prob

    # # Returns the feature-based edge score between two nodes
    # def edge_score(self, node1_ind, node2_ind):
    #     # Calculate l2 norm between the two nodes
    #     pos1 = self.G.node[node1_ind]['centroid']
    #     pos2 = self.G.node[node2_ind]['centroid']
    #     dist = np.sqrt(np.sum(np.square(np.subtract(pos1, pos2))))
    #     node1_feats = self.G.node[node1_ind]['feats'][2:]
    #     node2_feats = self.G.node[node2_ind]['feats'][2:]
    #     diff_feats = np.hstack([dist, np.subtract(node1_feats, node2_feats)])
    #     # Std dev
    #     sigma = self.feat_std_dev
    #     # Prob score
    #     prob = np.divide(np.square(diff_feats), np.square(sigma))
    #     prob = np.divide(np.exp(-prob), np.sqrt(2. * np.pi * np.square(sigma)))
    #     print prob
    #     prob = np.mean(np.multiply(prob, self.weights.flatten()))
    #     # prob = np.mean(prob)
    #     return prob

    # Returns the feature-based edge score between two nodes
    def edge_score(self, node1_ind, node2_ind):
        # Calculate l2 norm between the two nodes
        pos1 = self.G.node[node1_ind]['centroid']
        pos2 = self.G.node[node2_ind]['centroid']
        dist = np.sqrt(np.sum(np.square(np.subtract(pos1, pos2))))/self.search_dist
        node1_feats = self.G.node[node1_ind]['feats'][2:]
        node2_feats = self.G.node[node2_ind]['feats'][2:]
        diff_feats = np.hstack([dist, np.subtract(node1_feats, node2_feats)])
        # Std dev
        sigma = self.feat_std_dev.copy()
        sigma[0] /= self.search_dist
        # Prob score
        prob = np.divide(np.square(diff_feats), np.square(sigma))
        prob = np.divide(np.exp(-prob), np.sqrt(2. * np.pi * np.square(sigma)))
        prob = np.multiply(prob, self.weights.flatten())
        # print dist
        # print prob
        prob = np.mean(prob)
        # prob = np.mean(prob)
        return prob

    # Probability of an increase in the count of a node
    def prob_count_up(self, node_ind):
        # Get current count
        current_count = self.G.node[node_ind]['count']
        # Get current count probability
        new_prob = self.count_prob(node_ind, current_count+1)
        # Get new count probability
        current_prob = self.count_prob(node_ind, current_count)
        # Return difference
        return new_prob - current_prob

    # Returns cell count probability for a given detection based on overall mean detection area
    def count_prob(self, node_ind, count):
        sigma = 1
        if count >= 0:
            # Get node area
            node_area = self.G.node[node_ind]['feats'][2]
            max_prob_count = node_area / self.mean_area
            # Compute probability of count
            prob = np.divide(np.square(np.subtract(max_prob_count, count)), 2 * sigma ** 2)
            prob = np.divide(np.exp(-prob), np.sqrt(2. * np.pi * sigma ** 2))
        else:
            prob = 0
        return prob

    # Back-track through event variables and update graph as required
    def back_track(self, back_track_vars, track_score_sum, current_frame):
        back_track_ind = len(back_track_vars) - 1
        current_node_ind = np.argmax(track_score_sum)
        # current_frame -= 1
        while back_track_ind >= 0:
            next_node_ind = back_track_vars[back_track_ind][current_node_ind]["prev_node"]
            mig_prob = back_track_vars[back_track_ind][current_node_ind]["mig_prob"]
            current_node = self.img_nodes[current_frame][current_node_ind]
            next_node = self.img_nodes[current_frame-1][next_node_ind]
            # Remove edge if necessary
            in_edges = list(self.G.in_edges(current_node))
            if len(in_edges) == 1:
                edge = in_edges[0]
                if edge[0] != next_node:
                    self.G.remove_edge(edge[0], edge[1])
                    self.G.add_edge(next_node, current_node, prob_migration=mig_prob)
            elif len(in_edges) == 0:
                # Add edge form current node to next node
                self.G.add_edge(next_node, current_node, prob_migration=mig_prob)
            else:
                print "ERROR TOO MANY IN-EDGES"
            # Update variables
            current_node_ind = next_node_ind
            current_frame -= 1
            back_track_ind -= 1

    # Draws DAG representation of cell tracks
    # Inputs:
    #   with_labels - true if the graph should also draw cell #'s
    def draw_graph(self, with_labels):
        pos = {}
        node_count = 0
        for i in range(0, len(self.img_nodes)):
            for d in range(0, len(self.img_nodes[i])):
                pos[node_count] = ((i + 1), (d + 1))
                node_count += 1
        nx.draw(self.G, pos, with_labels=with_labels)
        plt.show()

    # Draws bounding box around specified contour
    #   contour - contour being bounded
    #   label   - cell/detection number to draw on box
    #   frame   - frame in which to draw box
    def draw_box(self, contour, label, frame, img):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "#" + format(label), (x + w + 10, y + h), 2, 0.4, (0, 255, 0))
        cv2.putText(img, "#" + format(label), (x - 10, y - 5), 2, 0.4, (0, 255, 0))
        return img

    # Draws the cell tracks as labeled bounding boxes in video sequence
    def gen_images(self):
        # Loop through frames
        track_count = 0
        nodes_drawn = np.zeros([self.G.number_of_nodes()])
        for frame in range(0, len(self.img_nodes)):
            img = self.video_img[frame].copy()
            # Loop through detections in current frame
            for i in range(0, len(self.img_nodes[frame])):
                detection = self.img_nodes[frame][i]
                # Check for predecessors with a track label
                in_edges = list(self.G.in_edges(detection))
                if len(in_edges) == 1:
                    # Inherit parent label
                    parent = in_edges[0][0]
                    nodes_drawn[detection] = nodes_drawn[parent]
                elif len(in_edges) == 0:
                    nodes_drawn[detection] = track_count
                    track_count += 1
                else:
                    print "ERROR TOO MANY INCOMING EDGES"
                # Draw bounding box
                track_label = nodes_drawn[detection]
                contour = self.contours[frame][i]
                img = self.draw_box(contour, track_label, frame, img)
            cv2.imwrite("./output/" + '{:03d}'.format(frame) + ".png", img)
        return

    # Draws the cell tracks as labeled bounding boxes in video sequence
    def gen_images_reverse(self, out_path):
        # Loop through frames
        track_count = 0
        nodes_drawn = np.zeros([self.G.number_of_nodes()])
        for frame in range(len(self.img_nodes)-1, -1, -1):
            img = self.video_img[frame].copy()
            # Loop through detections in current frame
            for i in range(0, len(self.img_nodes[frame])):
                detection = self.img_nodes[frame][i]
                # Check for predecessors with a track label
                out_edges = list(self.G.out_edges(detection))
                if len(out_edges) == 1:
                    # Inherit parent label
                    parent = out_edges[0][1]
                    nodes_drawn[detection] = nodes_drawn[parent]
                elif len(out_edges) == 0:
                    nodes_drawn[detection] = track_count
                    track_count += 1
                else:
                    nodes_drawn[detection] = track_count
                    track_count += 1
                # Draw bounding box
                track_label = nodes_drawn[detection]
                contour = self.contours[frame][i]
                img = self.draw_box(contour, track_label, frame, img)
            cv2.imwrite(out_path + '{:03d}'.format(frame) + ".png", img)
        return


# TODO: Biggest limitation is inability to have more than one edge to a single node - solutions: motion model, cell counts