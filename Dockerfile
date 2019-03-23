# Start off with ubuntu (https://hub.docker.com/_/ubuntu)
FROM ubuntu:latest as builder

# Install node (https://nodejs.org/en/download/package-manager/#debian-and-ubuntu-based-linux-distributions)
RUN apt-get update && apt-get install -y curl gnupg
RUN curl -sL https://deb.nodesource.com/setup_11.x | bash -
RUN apt-get update && apt-get install -y nodejs build-essential

# Install bazel (https://docs.bazel.build/versions/master/install-ubuntu.html#install-on-ubuntu)
RUN apt-get update && apt-get install -y openjdk-8-jdk
RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | \
    tee /etc/apt/sources.list.d/bazel.list
RUN curl https://bazel.build/bazel-release.pub.gpg | apt-key add -
RUN apt-get update && apt-get install -y bazel
RUN bazel version  # Extract before loading pentago

# Clone and build pentago
WORKDIR /pentago/pentago
ADD pentago /pentago/pentago
ADD third_party /pentago/third_party
ADD WORKSPACE /pentago/
RUN bazel build -c opt --copt=-march=native utility base data/... search end high mid \
    @lzma//... @zlib//...

# Test pentago except for mpi
WORKDIR /pentago/pentago
RUN bazel test -c opt --copt=-march=native --test_output=errors utility/... base/... \
    data/... end/... high/...  mid/... search/...

# Set up node
WORKDIR /pentago/web/server
ADD web/server /pentago/web/server
RUN npm install --unsafe-perm
RUN node unit.js

# Switch to minimal node.js image, keeping only what we need (see https://hub.docker.com/_/node)
FROM node:11-alpine

# Bring pentago node back up
COPY --from=builder /pentago/web/server /pentago/web/server
WORKDIR /pentago/web/server
RUN node unit.js

# Serve!
WORKDIR /var/pentago
CMD node /pentago/web/server/server.js --pool 7 --api-key `cat ssl/api-key`
