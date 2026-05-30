# Stage 1: The Builder
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ARG PUNKST_CMAKE_FLAGS="-DENABLE_NATIVE_ARCH=OFF -DENABLE_X86_64_V3=ON -DENABLE_LTO=OFF"

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libtbb-dev \
    libpng-dev \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- THE DEFINITIVE FIX IS HERE ---
# Instead of copying the entire context, only copy the contents
# of the 'punkst' sub-directory into the current working directory.
COPY . .

# Configure and build your project
RUN mkdir -p build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        ${PUNKST_CMAKE_FLAGS} && \
    cmake --build . -- -j4

# Stage 2: The Final Image
FROM ubuntu:24.04

# Install only the runtime libraries
RUN apt-get update && apt-get install -y \
    libtbb12 \
    libpng16-16 \
    libcurl4 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/bin/punkst .

ENTRYPOINT ["./punkst"]
