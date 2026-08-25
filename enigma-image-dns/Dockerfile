FROM sweagent/enigma:latest

USER root

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       dnsutils \
       netcat-openbsd \
       curl \
       iproute2 \
       iputils-ping \
    && rm -rf /var/lib/apt/lists/*
