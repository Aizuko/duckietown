FROM duckietown/dt-ros-commons:daffy

RUN mkdir -p /media/duckie_time
RUN chmod 1777 /tmp
#RUN git clone 'https://gitlab.com/aizuko/dotfiles' /root/dotfiles
WORKDIR /root/dotfiles
COPY ./dotfiles .
COPY ./binaries /root/bin
RUN echo "alias vi='vim'" >> bash/.bash_profile
RUN rm -f /root/.profile /root/.bashrc
RUN bash install.sh install

ENTRYPOINT ["dt-launcher-bash", "&&", "source", "/root/.bash_profile"]
