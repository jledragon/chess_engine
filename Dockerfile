# Build Dockerfile # docker build -t chess_engine .
# Debug in interactive mode # docker run --gpus all -it chess_engine /bin/bash
FROM python:3.14
# Debian GNU/Linux 12 (trixie)

# Install any pip packages needed
RUN python -m pip install --upgrade pip setuptools wheel
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy over the chess engine code
WORKDIR /src
COPY * .
COPY stockfish/ ./stockfish
COPY mobile_cvt/ ./mobile_cvt

# Install CUDA toolkit, allowing us to compile
RUN wget https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda-repo-debian12-13-0-local_13.0.0-580.65.06-1_amd64.deb
RUN dpkg -i cuda-repo-debian12-13-0-local_13.0.0-580.65.06-1_amd64.deb
RUN cp /var/cuda-repo-debian12-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
RUN apt-get update
RUN apt-get -y install cuda-toolkit-13-0

# Compile and install the chess engine C++ code
ENV try_compile=True
#RUN python setup.py install --user
#RUN pip install --no-build-isolation -e .

# The actual command - ensure this will be run with --gpus all and with virtualisation enabled in the BIOS
RUN chmod +x a2c_entrypoint.sh
CMD ["./a2c_entrypoint.sh"]