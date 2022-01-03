FROM mathworks/matlab:r2020b

USER root

# Set work directory
WORKDIR /app
# Install all requirements
RUN apt update
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update
RUN apt install python3.6 python3.6-dev python3-pip wget -y

ADD ./requirements.txt /app/requirements.txt
RUN python3.6 -m pip install --upgrade pip
RUN python3.6 -m pip install -r requirements.txt
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 10


RUN wget -q https://www.mathworks.com/mpm/glnxa64/mpm && \ 
    chmod +x mpm && \
    ./mpm install \
        --destination=/opt/matlab/R2020b/ \
        --release=r2020b \
        --products Parallel_Computing_Toolbox Signal_Processing_Toolbox Statistics_and_Machine_Learning_Toolbox

CMD [ "bash"]
