FROM duckietown/dt-duckiebot-interface:daffy-arm64v8
# use daffy-arm64v8 if you are using a Duckiebot MOOC Founder's Edition

WORKDIR /color_detector_dir
COPY requirements.txt ./

RUN sudo apt install bc bzip2 can-utils freeglut3-dev gstreamer1.0-alsa gstreamer1.0-libav gstreamer1.0-plugins-bad gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-ugly gstreamer1.0-tools libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev v4l-utils

RUN pip install -r requirements.txt

COPY color_detector.py .

CMD python3 ./color_detector.py
