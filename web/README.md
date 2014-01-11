Pentago web server and client
=============================

## Server setup

Mostly taken from http://agileand.me/content/deploying-play-application-rackspace-cloud-vps.

1. Set up firewall

    sudo apt-get install ufw
    sudo ufw allow ssh
    sudo ufw allow 2048/tcp
    sudo ufw enable
    sudo vi /etc/ssh/sshd_config # Change PermitRootLogin to no

2. Create a Pentago user

    sudo mkdir -p /var/pentago
    sudo groupadd pentago
    sudo useradd -g pentago -d /var/pentago pentago
    sudo chown pentago:pentago pentago 

3. Launch server

    sudo npm install -g forever 
    cd /var/pentago
    sudo su pentago
    forever ... server.js --pool 7 --log log
