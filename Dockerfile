# CUDA *devel* base, not plain ubuntu:22.04. FireANTs' fused_ops extension is
# built from source below, and fused_ops/setup.py uses
# torch.utils.cpp_extension.CUDAExtension over nine .cu sources. CUDAExtension()
# calls library_paths(cuda=True) -> _join_cuda_home(...), which raises
# `EnvironmentError: CUDA_HOME environment variable is not set` the moment nvcc
# is absent -- so on ubuntu:22.04 the build died at `setup.py build_ext`, before
# the `import fireants_fused_ops` assertion could ever run. The tag matches the
# cu121 torch wheel installed below; torch only hard-fails on a CUDA *major*
# mismatch (torch/utils/cpp_extension.py:_check_cuda_version), so 12.1 toolchain
# + any cu12x wheel is fine.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set non-interactive frontend for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# The CUDA base image ships nvcc but no Python, so python3/python3-pip are
# installed by the apt step below and every later `python3`/`pip3` invocation
# depends on that. Do not drop them.
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
# The build host has no GPU, so nvcc cannot autodetect an architecture and
# torch falls back to compiling for whatever it guesses. Name the deployment
# targets explicitly: A100 (sm_80) and H100 (sm_90) on pi_satra.
ENV TORCH_CUDA_ARCH_LIST="8.0;9.0"

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
        bc \
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

# CUDA torch + FireANTs. Registration is GPU-only.
RUN pip3 install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu121 \
        "torch==2.5.1+cu121" && \
    pip3 install --no-cache-dir fireants==1.5.0 "SimpleITK>=2.2.1"

# The fused CUDA ops are NOT included by `pip install fireants` and must be
# built from source. Without them FireANTs does not fail -- it forces
# is_onehot=True (fireants/io/image.py:145-147) and silently returns a one-hot
# volume costing 97x the memory. Assert the extension imports so a broken build
# is caught here rather than in a BABS array.
RUN git clone --depth 1 --branch v1.5.0 https://github.com/rohitrango/FireANTs.git /tmp/fireants-src && \
    cd /tmp/fireants-src/fused_ops && \
    python3 setup.py build_ext && \
    python3 setup.py install && \
    cd / && rm -rf /tmp/fireants-src && \
    python3 -c "import torch, fireants_fused_ops; print('fused ops OK')"

# Retry the downloads below. OSF answers with an occasional 5xx, and one
# failed request otherwise fails the whole build (wget exit 8).
RUN printf '%s\n' 'tries = 5' 'waitretry = 30' 'retry_connrefused = on' \
        'retry_on_http_error = 429,500,502,503,504' >> /etc/wgetrc

# Canonical ANTs binaries, pinned release. The wrapper shells out to
# antsBrainExtraction.sh (the published extraction method the OASIS-30
# template kit was built for); ANTsPy does not ship the CLI tools or the
# scripts. The layout inside the release zip is not guaranteed, so locate
# antsBrainExtraction.sh and symlink its directory to a fixed path.
# Mirrors Singularity exactly and must stay in sync with it.
RUN wget -q "https://github.com/ANTsX/ANTs/releases/download/v2.5.4/ants-2.5.4-ubuntu-22.04-X64-gcc.zip" -O /tmp/ants.zip && \
    unzip -q /tmp/ants.zip -d /opt/ants-dist && \
    rm /tmp/ants.zip && \
    ABE="$(find /opt/ants-dist -type f -name antsBrainExtraction.sh | head -1)" && \
    test -n "$ABE" && \
    ln -s "$(dirname "$ABE")" /opt/ants-bin && \
    chmod -R a+rx /opt/ants-dist && \
    test -x /opt/ants-bin/antsRegistration && \
    test -x /opt/ants-bin/Atropos

# Create directory for templates with proper permissions
RUN mkdir -p /opt/data && chmod 755 /opt/data

# Download OASIS templates and atlases
WORKDIR /opt/data

# Download the OASIS templates and atlases.
#
# These URLs mirror Singularity exactly and must stay in sync with it. The two
# recipes had drifted apart: this file pointed at OSF resource ej52r, which now
# returns HTTP 500, so `docker build` failed at this step -- and it never
# downloaded the jointfusion labels at all, so even a successful build produced
# an image where --method quick aborts with "Template labels not found". The
# Singularity URLs below are the ones that actually resolve.

# OASIS-30 Atropos template (preprocessing + Atropos priors)
RUN wget -q "https://osf.io/rh9km/?action=download&version=1" \
    -O OASIS-30_Atropos_template.zip && \
    unzip -q -o OASIS-30_Atropos_template.zip && \
    rm OASIS-30_Atropos_template.zip && \
    chmod -R 755 /opt/data/OASIS-30_Atropos_template

# OASIS-TRT-20 atlas brains for joint label fusion
RUN wget -q "https://files.osf.io/v1/resources/hs8am/providers/osfstorage/57c1a8f06c613b01f98d68a9/?zip=" \
    -O OASIS-TRT-20_brains.zip && \
    unzip -q -o OASIS-TRT-20_brains.zip -d OASIS-TRT-20_brains && \
    rm OASIS-TRT-20_brains.zip

# Matching DKT31/CMA labels for those atlases
RUN wget -q "https://files.osf.io/v1/resources/hs8am/providers/osfstorage/57c1a8ffb83f690201c4a8be/?zip=" \
    -O OASIS-TRT-20_DKT31_CMA_labels_v2.zip && \
    unzip -q -o OASIS-TRT-20_DKT31_CMA_labels_v2.zip -d OASIS-TRT-20_DKT31_CMA_labels_v2 && \
    rm OASIS-TRT-20_DKT31_CMA_labels_v2.zip

# Pre-computed jointfusion labels in OASIS-30 space, required by --method quick
RUN wget -q "https://osf.io/download/nxg5t/" \
    -O OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_OASIS-30_v2.nii.gz

# Both atlas archives may unpack with a redundant nesting level; flatten it so
# the wrapper's expected paths resolve (same fix-up Singularity performs).
RUN for d in OASIS-TRT-20_brains OASIS-TRT-20_DKT31_CMA_labels_v2; do \
        if [ -d "$d/$d" ]; then \
            echo "Flattening nested $d"; \
            mv "$d/$d"/* "$d"/ && rmdir "$d/$d"; \
        fi; \
    done && \
    chmod -R 755 /opt/data && \
    ls -la /opt/data

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
# Trailing slash: older ANTs scripts concatenate ${ANTSPATH}tool directly
ENV ANTSPATH=/opt/ants-bin/
ENV PATH=/opt/ants-bin:/usr/local/bin:$PATH
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
