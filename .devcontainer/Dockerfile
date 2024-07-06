FROM ubuntu:22.04

ENV TZ=Australia/Melbourne

RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update -y

RUN DEBIAN_FRONTEND=noninteractive \
    && apt-get install wget -y \
    && apt-get install curl -y \
    && apt-get install htop -y \
    && apt-get install zsh -y \
    && apt-get install tzdata -y \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && apt-get update -y \
    && apt-get dist-upgrade -y \
    && apt-get install git -y \
    && apt-get install python3-pip -y \
    && apt-get install python3-venv -y \
    && sh -c "$(wget https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"

EXPOSE 7000
