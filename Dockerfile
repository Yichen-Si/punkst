# Stage 1: The Builder
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libtbb-dev \
    libopencv-dev \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- THE DEFINITIVE FIX IS HERE ---
# Instead of copying the entire context, only copy the contents
# of the 'punkst' sub-directory into the current working directory.
COPY . .

# Configure and build your project
RUN mkdir -p build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . -- -j$(nproc)

# Stage 2: The Final Image
FROM ubuntu:24.04

# Install only the runtime libraries
RUN apt-get update && apt-get install -y \
    libtbb12 \
    libopencv-core4.6 \
    libopencv-imgproc4.6 \
    libopencv-imgcodecs4.6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/bin/punkst .

ENTRYPOINT ["./punkst"]
