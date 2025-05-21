# Use ubuntu:20.04 as the base image
FROM ubuntu:20.04

# Avoid tzdata interactive configuration
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && apt-get install -y \
    xfce4 \
    xfce4-goodies \
    tightvncserver \
    firefox \
    xterm \
    xdg-utils \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Add a new user 'agent' and set password 'password'
RUN useradd -m -s /bin/bash agent && \
    echo "agent:password" | chpasswd && \
    adduser agent sudo

# Create VNC directory and xstartup script for the 'agent' user
RUN mkdir -p /home/agent/.vnc && \
    echo '#!/bin/sh' > /home/agent/.vnc/xstartup && \
    echo 'unset SESSION_MANAGER' >> /home/agent/.vnc/xstartup && \
    echo 'unset DBUS_SESSION_BUS_ADDRESS' >> /home/agent/.vnc/xstartup && \
    echo 'startxfce4 &' >> /home/agent/.vnc/xstartup && \
    chown -R agent:agent /home/agent/.vnc && \
    chmod +x /home/agent/.vnc/xstartup

# Set VNC password environment variable (also used by tightvncserver setup)
ENV VNC_USER=agent
ENV VNC_PW=password
ENV DISPLAY_NUM=1
ENV GEOMETRY=1280x960
ENV COLOR_DEPTH=24
ENV VNC_PORT=5901

# Switch to the 'agent' user
USER agent
WORKDIR /home/agent

# Set up VNC server for the 'agent' user
# This command will prompt for the password, we use expect to automate this.
# However, a simpler way for tightvncserver is to set it up once.
# The -fg option will keep it in the foreground.
# The password file needs to be created first.
RUN apt-get update && apt-get install -y expect && rm -rf /var/lib/apt/lists/* && \
    (echo $VNC_PW && echo $VNC_PW) | vncpasswd && \
    apt-get remove -y --purge expect && apt-get autoremove -y && apt-get clean

# Expose VNC port (5900 + display_number)
EXPOSE $VNC_PORT

# Set the default command to start VNC server
CMD ["tightvncserver", "-fg", "-geometry", "1280x960", "-depth", "24", ":1"]
