import numpy as np
import os 
def read_detected_objects():
        detected_objects = []
        all_labels = []
        if os.path.exists('send_server/objects.txt'):
            try:
                with open('send_server/objects.txt', 'r') as f:
                    current_objects = []
                    position = []
                    total = []
                    pos = []
                    for line in f:
                        line = line.strip()
                        if line.startswith("Detected Objects:"):
                            if current_objects and position:
                                for i in range(len(current_objects)):
                                    total.append(current_objects[i])
                                    pos.append(np.array(position, dtype=float))
                                current_objects = []
                                position = []
                        elif line.startswith("X:"):
                            try:
                                coords = line.split(", ")
                                position = [float(coord.split(": ")[1]) for coord in coords]
                            except ValueError as ve:
                                print('error')
                        elif line and not line.startswith("Viewer Position:"):
                            current_objects.append(line)
                    if current_objects and position:
                        for i in range(len(current_objects)):
                            total.append(current_objects[i])
                            pos.append(np.array(position, dtype=float))
            except Exception as e:
                print(f'error {e}')
        return total, pos


detected_objects, labels = read_detected_objects()
print(detected_objects)
print(labels)
