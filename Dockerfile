# Start off with ubuntu (https://hub.docker.com/_/ubuntu)
FROM ubuntu:devel as builder

# Install node (https://nodejs.org/en/download/package-manager/#debian-and-ubuntu-based-linux-distributions)
RUN apt-get update && apt-get install -y curl
RUN curl -sL https://deb.nodesource.com/setup_8.x | bash -
RUN apt-get update && apt-get install -y nodejs build-essential

# Install bazel (https://docs.bazel.build/versions/master/install-ubuntu.html#install-on-ubuntu)
RUN apt-get update && apt-get install -y openjdk-8-jdk
RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | \
    tee /etc/apt/sources.list.d/bazel.list
RUN curl https://bazel.build/bazel-release.pub.gpg | apt-key add -
RUN apt-get update && apt-get install -y bazel
RUN bazel version  # Extract before loading pentago

# Clone and build pentago
RUN apt-get update && apt-get install -y python-numpy
WORKDIR /pentago/pentago
ADD pentago /pentago/pentago
ADD third_party /pentago/third_party
ADD WORKSPACE /pentago/
RUN bazel build -c opt --copt=-march=native utility base data/... search end high mid \
    @lzma//... @zlib//...

# Test pentago except for mpi
WORKDIR /pentago/pentago
RUN bazel test -c opt --copt=-march=native utility/... base/... data/... end/... \
    high/...  mid/... search/...

# Set up node
WORKDIR /pentago/web
RUN npm install -g node-gyp
ADD web /pentago/web
RUN node-gyp configure
RUN node-gyp build
RUN npm install
RUN node unit.js

# Switch to minimal node.js image, keeping only what we need (see https://hub.docker.com/_/node)
FROM node:8.9.1-alpine

# Fix timezone issues, as described in https://github.com/TooTallNate/node-time/issues/94
RUN apk add --update tzdata
ENV TZ America/Los_Angeles

# Bring pentago node back up
COPY --from=builder /pentago/web /pentago/web
WORKDIR /pentago/web
RUN node unit.js

# Serve!
WORKDIR /var/pentago
CMD node /pentago/web/server.js --pool 7
