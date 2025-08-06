#!/bin/bash

# Build and run Docker containers using docker-compose
cd "$(dirname "$0")/docker" || exit
docker-compose up --build
