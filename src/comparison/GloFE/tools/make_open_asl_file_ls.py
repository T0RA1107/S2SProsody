import os

dir = "/home/ubuntu/slocal/S2SProsody/data/OpenASL/data/video-clip"

with open("./open_asl_all.txt", "w") as f:
    for file_name in os.listdir(dir):
        if file_name[-4:] == ".mp4":
            f.write(os.path.join(dir, file_name) + "\n")
