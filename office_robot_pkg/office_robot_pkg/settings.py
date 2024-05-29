import sys
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QPushButton, QTextEdit, QTableWidgetItem
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
from sklearn.metrics import silhouette_score
import matplotlib.cm as cm
import datetime
from datetime import datetime
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int64, Float64, String

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from matplotlib import cm
import os
from joblib import Parallel, delayed

from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde

class SettingsPublisher(Node):
    def __init__(self):
        super().__init__('settings_publisher')
        self.setting_value = None
        self.toggle = False
        self.objects = None
        self.position_threshold = 10.0
        self.publisher_ = self.create_publisher(Int64, '/settings', 10)
        self.publisher2_ = self.create_publisher(Int64, '/video_capture', 1)

        self.subscription_ = self.create_subscription(
            String,
            '/objects_detected',
            self.objects_topic_callback,
            10
        )
        
        self.get_logger().info('SettingsPublisher node initialized')
        
        self.launch_settings_dialog()

    def objects_topic_callback(self, msg):
        self.get_logger().info('Message received')
        self.message_box.append(f"Received message: {msg.data}")


    def publish_setting(self, setting_value):
        msg = Int64()
        msg.data = setting_value
        self.setting_value = setting_value
        self.publisher_.publish(msg)

    def publish_video_capture(self, value):
        msg = Int64()
        msg.data = value
        self.publisher2_.publish(msg)

    def launch_settings_dialog(self):
        self.app = QApplication.instance()  
        if not self.app: 
            self.app = QApplication(sys.argv)

        self.advanced_dialog = QDialog()
        self.advanced_dialog.setWindowTitle("Control Panel")
        self.advanced_dialog.setFixedSize(600, 400)

        layout = QVBoxLayout()

        button1 = QPushButton('Detect Additions to Environment')
        button1.clicked.connect(lambda: self.publish_setting(1))
        button2 = QPushButton('Detect Removals from Environment')
        button2.clicked.connect(lambda: self.publish_setting(3))
        button3 = QPushButton('Detect All Changes to Environment')
        button3.clicked.connect(lambda: self.publish_setting(2))

        layout.addWidget(button1)
        layout.addWidget(button2)
        layout.addWidget(button3)

        self.message_box = QTextEdit()
        self.message_box.setReadOnly(False)
        layout.addWidget(self.message_box)

        button_layout = QVBoxLayout()
        view_compare_button = QPushButton('View Current Comparison')
        view_compare_button.clicked.connect(self.compare)
        view_diff_button = QPushButton('View Differences')
        view_diff_button.clicked.connect(self.differences)
        view_map_button = QPushButton('View Updated Map')
        view_map_button.clicked.connect(self.view_updated_map)
        generate_heatmap_button = QPushButton('Generate Heat Map')
        generate_heatmap_button.clicked.connect(self.heatmap)
        generate_heatmap_video_button = QPushButton('Start/Stop Generating Heat Map Video')
        generate_heatmap_video_button.clicked.connect(self.start_video_capture)
        log_changes_button = QPushButton('Log All Changes')
        log_changes_button.clicked.connect(self.log_changes)


        button_layout.addWidget(view_compare_button)
        button_layout.addWidget(view_diff_button)
        button_layout.addWidget(view_map_button)
        button_layout.addWidget(generate_heatmap_button)
        button_layout.addWidget(generate_heatmap_video_button)
        button_layout.addWidget(log_changes_button)


        layout.addLayout(button_layout)

        self.advanced_dialog.setLayout(layout)
        self.advanced_dialog.exec_()

    def view_updated_map(self):
        array = np.load('arrays/updated_whole_map_point_cloud.npy')
        filtered_array = array[~np.isinf(array).any(axis=1) & (array[:, 0] > -6) & (array[:, 0] < 5) & (array[:, 1] > -5) & (array[:, 1] < 5)]
        num_samples = int(0.7 * len(filtered_array))
        random_indices = np.random.choice(filtered_array.shape[0], num_samples, replace=False)
        sampled_array = filtered_array[random_indices]
        x, y, z = sampled_array[:, 0], sampled_array[:, 1], sampled_array[:, 2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c='r', marker='o', s= 0.5)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.title('Randomly Sampled Point Cloud Visualization')
        plt.show()

    def compare(self):
        array1 = np.load('arrays/array1.npy')
        array2 = np.load('arrays/array2.npy')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(array1[:, 0], array1[:, 1], array1[:, 2], c='blue', alpha=0.5, label='Reference Point Cloud', s=1)
        ax.scatter(array2[:, 0], array2[:, 1], array2[:, 2], c='red', alpha=0.5, label='Depth Camera Point Cloud', s=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # plt.title('3D Scatter Plot of Two Point Clouds - Differences and Clusters')
        plt.legend()
        plt.show()

    def differences(self, direction = 1):
        array1 = np.load('arrays/array1.npy')
        array2 = np.load('arrays/array2.npy')

        if self.setting_value == 1:
            significant_clusters = np.load('arrays/significant_clusters.npy')
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(array1[:, 0], array1[:, 1], array1[:, 2], c='blue', alpha=0.5, label='Array 1', s = 0.5)
            # ax.scatter(array2[:, 0], array2[:, 1], array2[:, 2], c='green', alpha=0.5, label='Array 2')
            if significant_clusters.size > 0:
                ax.scatter(significant_clusters[:, 0], significant_clusters[:, 1], significant_clusters[:, 2], c='red', s=0.5, label='Significant Clusters')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            plt.title('3D Scatter Plot of Two Point Clouds - Differences and Clusters')
            plt.legend()
            plt.show()

            significant_clusters = np.load('arrays/significant_clusters.npy')
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(array1[:, 0], array1[:, 1], array1[:, 2], c='blue', alpha=0.5, label='Reference Point Cloud', s = 0.5)
            # ax.scatter(array2[:, 0], array2[:, 1], array2[:, 2], c='green', alpha=0.5, label='Array 2')
            if significant_clusters.size > 0:
                ax.scatter(significant_clusters[:, 0], significant_clusters[:, 1], significant_clusters[:, 2], c='yellow', s=0.5, label='Significant Clusters')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            plt.title('3D Scatter Plot of Two Point Clouds - Differences and Clusters')
            plt.legend()
            plt.show()
        elif self.setting_value == 2:
            significant_clusters = np.load('arrays/significant_clusters2.npy')
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(array1[:, 0], array1[:, 1], array1[:, 2], c='red', alpha=0.5, label='Array 1')
            ax.scatter(array2[:, 0], array2[:, 1], array2[:, 2], c='green', alpha=0.5, label='Array 2', s= 0.5)
            if significant_clusters.size > 0:
                ax.scatter(significant_clusters[:, 0], significant_clusters[:, 1], significant_clusters[:, 2], c='blue', s=0.5, label='Significant Clusters')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            plt.title('3D Scatter Plot of Two Point Clouds - Differences and Clusters')
            plt.legend()
            plt.show()

    def heatmap(self):
        array1 = np.load('arrays/updated_whole_map_point_cloud.npy')
        array2 = np.load('arrays/whole_map_point_cloud.npy')

        conditions = lambda arr: (~np.isinf(arr).any(axis=1) & ~np.isnan(arr).any(axis=1) &
                                ((arr[:, 0] > -6) & (arr[:, 0] < 5)) &
                                ((arr[:, 1] > -5) & (arr[:, 1] < 5)) & 
                                (arr[:, 2] > 0))

        array1 = array1[conditions(array1)]
        array2 = array2[conditions(array2)]

        array1_xy = array1[:, :2]
        array2_xy = array2[:, :2]

        tree1_xy = cKDTree(array1_xy)
        tree2_xy = cKDTree(array2_xy)

        indices_to_keep = tree2_xy.query_ball_point(array1_xy, r=0.05)
        indices_to_remove = set(range(len(array2_xy))) - set(np.concatenate(indices_to_keep))

        removed_points = array2_xy[list(indices_to_remove)]

        x_min, x_max, y_min, y_max = -6, 5, -5, 5
        grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]

        kde = gaussian_kde(array1_xy.T, bw_method='silverman')
        grid_kde = kde(np.vstack([grid_x.ravel(), grid_y.ravel()]))

        norm = plt.Normalize(grid_kde.min(), grid_kde.max())

        plt.figure(figsize=(10, 8))
        heatmap = plt.imshow(grid_kde.reshape(grid_x.shape).T, origin='lower', cmap='viridis', norm=norm, extent=(x_min, x_max, y_min, y_max))
        plt.colorbar(heatmap, label='KDE of Remaining Points')

        if removed_points.size > 0:
            removed_points_kde = gaussian_kde(removed_points.T, bw_method='silverman')
            removed_heatmap = removed_points_kde(np.vstack([grid_x.ravel(), grid_y.ravel()]))
            norm_removed = plt.Normalize(removed_heatmap.min(), removed_heatmap.max())
            plt.imshow(removed_heatmap.reshape(grid_x.shape).T, origin='lower', cmap='inferno', alpha=0.5, norm=norm_removed, extent=(x_min, x_max, y_min, y_max))
            plt.colorbar(plt.cm.ScalarMappable(norm=norm_removed, cmap='viridis'), label='KDE of Removed Points')

        plt.scatter(array2_xy[:, 0], array2_xy[:, 1], s=1, c='black')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Heatmap of Point Cloud Differences with Structural Outlines')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()

    def heatmap_video(self):
        while self.toggle_video:
            array1 = np.load('arrays/updated_whole_map_point_cloud.npy')
            array2 = np.load('arrays/whole_map_point_cloud.npy')

            conditions = lambda arr: (~np.isinf(arr).any(axis=1) & ~np.isnan(arr).any(axis=1) &
                                    ((arr[:, 0] > -6) & (arr[:, 0] < 5)) &
                                    ((arr[:, 1] > -5) & (arr[:, 1] < 5)) & 
                                    (arr[:, 2] > 0))

            array1 = array1[conditions(array1)]
            array2 = array2[conditions(array2)]

            array1_xy = array1[:, :2]
            array2_xy = array2[:, :2]

            tree1_xy = cKDTree(array1_xy)
            tree2_xy = cKDTree(array2_xy)

            indices_to_keep = tree2_xy.query_ball_point(array1_xy, r=0.05)
            indices_to_remove = set(range(len(array2_xy))) - set(np.concatenate(indices_to_keep))

            removed_points = array2_xy[list(indices_to_remove)]

            x_min, x_max, y_min, y_max = -6, 5, -5, 5
            grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]

            kde = gaussian_kde(array1_xy.T, bw_method='silverman')
            grid_kde = kde(np.vstack([grid_x.ravel(), grid_y.ravel()]))

            norm = plt.Normalize(grid_kde.min(), grid_kde.max())

            plt.figure(figsize=(10, 8))
            heatmap = plt.imshow(grid_kde.reshape(grid_x.shape).T, origin='lower', cmap='viridis', norm=norm, extent=(x_min, x_max, y_min, y_max))
            plt.colorbar(heatmap, label='KDE of Remaining Points')

            if removed_points.size > 0:
                removed_points_kde = gaussian_kde(removed_points.T, bw_method='silverman')
                removed_heatmap = removed_points_kde(np.vstack([grid_x.ravel(), grid_y.ravel()]))
                norm_removed = plt.Normalize(removed_heatmap.min(), removed_heatmap.max())
                plt.imshow(removed_heatmap.reshape(grid_x.shape).T, origin='lower', cmap='viridis', alpha=0.5, norm=norm_removed, extent=(x_min, x_max, y_min, y_max))
                plt.colorbar(plt.cm.ScalarMappable(norm=norm_removed, cmap='viridis'), label='KDE of Removed Points')

            plt.scatter(array2_xy[:, 0], array2_xy[:, 1], s=1, c='black', label='Original Point Cloud')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.title('Heatmap of Point Cloud Differences with Structural Outlines')
            plt.legend()
            plt.axis('equal')
            plt.grid(True)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            plt.savefig(f'heatmap_{timestamp}.png')
            plt.close()
            time.sleep(5)

    def start_video_capture(self):
        if self.toggle is False:
            self.publish_video_capture(1)
        else:
            self.toggle == True
            self.publish_video_capture(0)
        
    def optimal_clusters(self, data, min_k=2, max_k=6):
        if data.size == 0:
            return None, 0 

        def compute_kmeans(k):
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=1000, max_iter=300).fit(data)
            inertia = kmeans.inertia_
            if k > 1:
                silhouette = silhouette_score(data, kmeans.labels_)
                calinski_harabasz = calinski_harabasz_score(data, kmeans.labels_)
                davies_bouldin = davies_bouldin_score(data, kmeans.labels_)
                score = (silhouette + calinski_harabasz - davies_bouldin) / 3
            else:
                score = -1
            return inertia, score

        results = Parallel(n_jobs=-1)(delayed(compute_kmeans)(k) for k in range(min_k, max_k + 1))
        inertias, scores = zip(*results)

        if max_k == 1:
            optimal_k = 1
        else:
            optimal_k = np.argmax(scores) + min_k

        final_kmeans = KMeans(n_clusters=optimal_k, random_state=0, n_init=5, max_iter=300).fit(data)
        return final_kmeans, optimal_k

    def log_changes(self):
        array1 = np.load('arrays/updated_whole_map_point_cloud.npy')
        array2 = np.load('arrays/whole_map_point_cloud.npy')

        conditions = lambda arr: (~np.isinf(arr).any(axis=1) & ~np.isnan(arr).any(axis=1) &
                                ((arr[:, 0] > -6) & (arr[:, 0] < 5)) &
                                ((arr[:, 1] > -5) & (arr[:, 1] < 5)) & 
                                (arr[:, 2] > 0))
        array1 = array1[conditions(array1)]
        array2 = array2[conditions(array2)]

        tree1 = cKDTree(array1, leafsize=40)
        tree2 = cKDTree(array2, leafsize=40)
        distance_threshold = 0.5  
        cluster_size_threshold = 10  

        indices_to_keep_from_array2 = tree2.query_ball_point(array1, r=0.001)
        indices_removed = set(range(len(array2))) - set(np.concatenate(indices_to_keep_from_array2))

        indices_to_keep_from_array1 = tree1.query_ball_point(array2, r=0.001)
        indices_added = set(range(len(array1))) - set(np.concatenate(indices_to_keep_from_array1))

        added_points = array1[list(indices_added)]
        removed_points = array2[list(indices_removed)]

        kmeans_added, k_added = self.optimal_clusters(added_points)
        kmeans_removed, k_removed = self.optimal_clusters(removed_points, min_k=1, max_k=1)

        clustered_changes = []

        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': '3d'})

        ax.scatter(array2[:, 0], array2[:, 1], array2[:, 2], color='silver', alpha=0.2, label='Original Whole Data', s=0.5)

        colors = cm.get_cmap('viridis', k_added)
        removed_colors = cm.get_cmap('inferno', k_removed)

        if k_added > 0:
            for idx, center in enumerate(kmeans_added.cluster_centers_):
                nearest_dist = tree2.query(center)[0]
                cluster_size = np.sum(kmeans_added.labels_ == idx)
                if nearest_dist > distance_threshold or cluster_size >= cluster_size_threshold:
                    clustered_changes.append(f"Object added at {center[:2]} with {cluster_size} points")
                    label = f'Object_{idx} added'
                    points = added_points[kmeans_added.labels_ == idx]
                    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=colors(idx), label=label, s=0.5)

        if k_removed > 0:
            for idx, center in enumerate(kmeans_removed.cluster_centers_):
                nearest_dist = tree1.query(center)[0]
                cluster_size = np.sum(kmeans_removed.labels_ == idx)
                if nearest_dist > distance_threshold or cluster_size >= cluster_size_threshold:
                    clustered_changes.append(f"Object removed at {center[:2]} with {cluster_size} points")
                    points = removed_points[kmeans_removed.labels_ == idx]
                    label = f'Object_{idx} removed'
                    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=removed_colors(idx), label=label, s=0.5)

        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')
        ax.set_title('3D Map of Point Cloud with Cluster Labels')
        ax.legend()
        
        figure_file = 'send_server/clustered_changes_figure.png'
        try:
            plt.savefig(figure_file)
            self.get_logger().info(f"Figure saved to {figure_file}")
        except Exception as e:
            self.get_logger().error(f"Error saving figure: {e}")
        
        plt.show()

        changes_file = 'send_server/clustered_changes'
        try:
            with open(changes_file, 'w') as f:
                for change in clustered_changes:
                    f.write(change + '\n')
            self.get_logger().info(f"Clustered changes saved to {changes_file}")
        except Exception as e:
            self.get_logger().error(f"Error writing clustered changes to file: {e}")

    # def read_detected_objects(self):
    #     if os.path.exists('send_server/objects.txt'):
    #         try:
    #             with open('', 'r') as f:
    #                 current_objects = []
    #                 position = []
    #                 total = []
    #                 pos = []
    #                 for line in f:
    #                     line = line.strip()
    #                     if line.startswith("Detected Objects:"):
    #                         if current_objects and position: 
    #                             for i in range(len(current_objects)):
    #                                 total.append(current_objects[i])
    #                                 pos.append(np.array(position, dtype=float))
    #                             current_objects = []
    #                             position = []
    #                     elif line.startswith("X:"):
    #                         try:
    #                             coords = line.split(", ")
    #                             position = [float(coord.split(": ")[1]) for coord in coords]
    #                         except ValueError as ve:
    #                             self.get_logger().error(f"Failed to parse position from line '{line}': {ve}")
    #                     elif line and not line.startswith("Viewer Position:"):
    #                         current_objects.append(line)
    #                 if current_objects and position:
    #                     for i in range(len(current_objects)):
    #                         total.append(current_objects[i])
    #                         pos.append(np.array(position, dtype=float))
    #         except Exception as e:
    #             self.get_logger().error(f"Error reading detected objects file: {e}")
    #     return total, pos


    # def match_labels(self, centres, pos):
    #     labels = []
    #     for center in centres:
    #         low = float('inf')
    #         index = None
    #         for i, position in enumerate(pos):
    #             diff = np.linalg.norm(center - position)
    #             if diff < low:
    #                 low = diff
    #                 index = i
    #         if index is not None:
    #             labels.append(index)
    #             pos.pop(index)
    #         else:
    #             labels.append(None)
    #     return labels


def main(args=None):
    rclpy.init(args=args)
    settings_publisher = SettingsPublisher()
    try:
        settings_publisher.get_logger().info('Node is spinning')
        rclpy.spin(settings_publisher)
    except KeyboardInterrupt:
        settings_publisher.get_logger().info('Keyboard interrupt, shutting down')
    except Exception as e:
        settings_publisher.get_logger().error(f'Exception in spin: {e}')
    finally:
        settings_publisher.destroy_node()
        rclpy.shutdown()
        settings_publisher.get_logger().info('Node has been shutdown')

if __name__ == '__main__':
    main()