FROM ubuntu:22.04

# Set non-interactive frontend for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        wget \
        curl \
        unzip \
        git \
        gcc \
        g++ \
        cmake \
        build-essential \
        libgomp1 \
        ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    numpy \
    nibabel \
    pandas \
    pybids \
    nipype \
    antspyx

# Create directory for templates with proper permissions
RUN mkdir -p /opt/data && chmod 755 /opt/data

# Download OASIS templates and atlases
WORKDIR /opt/data

# Download OASIS-30 Atropos template
RUN wget -q https://files.osf.io/v1/resources/ej52r/providers/osfstorage/5e8251d20f59a8001ae648c3/?zip= \
    -O OASIS-30_Atropos_template.zip && \
    unzip -q OASIS-30_Atropos_template.zip && \
    rm OASIS-30_Atropos_template.zip && \
    chmod -R 755 /opt/data/OASIS-30_Atropos_template

# Download OASIS-TRT-20 atlases for joint label fusion
RUN wget -q https://files.osf.io/v1/resources/ej52r/providers/osfstorage/5e825d0f2301130197e6a15e/?zip= \
    -O OASIS-TRT-20.zip && \
    unzip -q OASIS-TRT-20.zip && \
    rm OASIS-TRT-20.zip && \
    chmod -R 755 /opt/data/OASIS-TRT-20_*

# Create app directory with proper permissions
RUN mkdir -p /app && chmod 755 /app
WORKDIR /app

# Copy application code
COPY setup.py /app/
COPY requirements.txt /app/
COPY src/ /app/src/
COPY README.md /app/

# Install the application and NIDM conversion toolkit.
# Order matters and mirrors Singularity (and the sibling freesurfer-nidm app):
# the NIDM submodule first with its loose deps, then the top-level
# requirements.txt applies the authoritative pins on top. The submodule's own
# requirements.txt is deliberately NOT installed -- it pins pynidm==4.2.4, which
# would silently override the 4.5.0 pin here.
RUN pip3 install -e src/ants_seg_to_nidm && \
    pip3 install -r requirements.txt && \
    pip3 install --no-cache-dir --upgrade 'rdflib>=7.0.0,<8' && \
    pip3 install -e . && \
    chmod -R 755 /app

# Set environment variables
ENV ANTSPATH=/usr/local/bin
ENV PATH=/usr/local/bin:$PATH
ENV ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1

# Create work directory for temporary files with world-writable permissions
RUN mkdir -p /work && chmod 777 /work
ENV TMPDIR=/work
ENV TEMP=/work
ENV TMP=/work

# Create entrypoint script with proper permissions
RUN echo '#!/bin/bash\nexec ants-nidm "$@"' > /entrypoint.sh && \
    chmod 755 /entrypoint.sh

# Ensure all installed binaries are executable
RUN chmod -R 755 /usr/local/bin || true

# Set a non-root user (but allow running as any UID/GID)
# This helps with permission issues in HPC environments
RUN echo "ALL ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    chmod 666 /etc/passwd /etc/group || true

ENTRYPOINT ["/entrypoint.sh"]
