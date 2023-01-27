FROM duckietown/dt-ros-commons:daffy

RUN mkdir -p /media/duckie_time
#RUN git clone 'https://gitlab.com/aizuko/dotfiles' /root/dotfiles
WORKDIR /root/dotfiles
COPY ./dotfiles .
RUN rm -f /root/.profile /root/.bashrc
RUN bash install.sh install

ENTRYPOINT ["/bin/bash", "--login", "--rcfile", "/root/.bash_profile"]
