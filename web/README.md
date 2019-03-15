Pentago web server and client
=============================

## Client

The client is hosted on Firebase.  To deploy:

    cd pentago/web/client
    npm install
    make public
    firebase deploy

## Server build

To build and test the `npm` extension module, do

    cd pentago/web/server
    bazel build -c opt --copt=-march=native //pentago/{end,high,mid}/...
    npm install
    node unit.js all

## Server setup

See the main `README.md` for the new Docker setup.
