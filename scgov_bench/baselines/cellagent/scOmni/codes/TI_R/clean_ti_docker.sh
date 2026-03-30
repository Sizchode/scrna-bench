#!/bin/bash
docker rm $(docker ps -a -q)
docker ps -a | grep 'dynverse/ti' | awk '{print $1}' | xargs docker rm