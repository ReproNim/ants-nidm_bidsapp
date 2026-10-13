Bootstrap: docker
# CUDA *devel* base, not plain ubuntu:22.04. FireANTs' fused_ops extension is
# built from source in %post, and fused_ops/setup.py uses
# torch.utils.cpp_extension.CUDAExtension over nine .cu sources. CUDAExtension()
# calls library_paths(cuda=True) -> _join_cuda_home(...), which raises
# `EnvironmentError: CUDA_HOME environment variable is not set` the moment nvcc
# is absent -- so on ubuntu:22.04 the build died at `setup.py build_ext`, before
# the `import fireants_fused_ops` assertion could ever run. The tag matches the
# cu121 torch wheel installed below and must stay in sync with the Dockerfile.
From: nvidia/cuda:12.1.1-devel-ubuntu22.04

# The commit this image was built from, read back with
# `apptainer inspect --labels <image>`. Pass it at build time:
#   apptainer build --build-arg GIT_COMMIT=$(git rev-parse HEAD) ...
# Without --build-arg the label reads "unknown".
%arguments
    GIT_COMMIT=unknown

%labels
    org.opencontainers.image.revision {{ GIT_COMMIT }}
    org.opencontainers.image.source https://github.com/ReproNim/ants-nidm_bidsapp

%files
    ./src /opt/src
    ./setup.py /opt/setup.py
    ./requirements.txt /opt/requirements.txt

%post
    # Fail the build on the first failing command. Without this, %post's exit
    # status is that of its *last* command -- and the
    # `import fireants_fused_ops` assertion below is followed by ~40 more
    # commands, so a failed fused_ops build produced a container that built
    # successfully and then silently degraded to the one-hot label path at
    # runtime (fireants/io/image.py:145-147). `set -e` is what makes that
    # assertion, and every other check in here, actually gate the build.
    set -e

    export DEBIAN_FRONTEND=noninteractive
    # The CUDA base image ships nvcc but no Python; python3.10/python3-pip are
    # installed below and everything after depends on them.
    export CUDA_HOME=/usr/local/cuda
    export PATH=/usr/local/cuda/bin:$PATH
    # The build host has no GPU, so nvcc cannot autodetect an architecture.
    # Name the deployment targets explicitly: A100 (sm_80) and H100 (sm_90).
    export TORCH_CUDA_ARCH_LIST="8.0;9.0"

    # Update and install basic system dependencies. Separate statements, not
    # one `a && b && c` chain: `set -e` does not fire on a non-final command
    # of an && list, so a failed `apt update` used to be swallowed and the
    # build carried on to `python3: not found` (exit 127) with no clue why.
    # Rootless build support: with no /etc/subuid entry for the building user,
    # apptainer falls back to a single-uid root-mapped namespace, and apt's
    # privilege drop to `_apt` (setgroups/seteuid) then fails with "Operation
    # not permitted" and the https method dies. Keep apt running as root.
    mkdir -p /etc/apt/apt.conf.d
    echo 'APT::Sandbox::User "root";' > /etc/apt/apt.conf.d/99-rootless-build
    apt-get update
    apt-get install -y \
        python3.10 \
        python3.10-dev \
        python3-pip \
        build-essential \
        cmake \
        libpng-dev \
        pkg-config \
        git \
        wget \
        unzip \
        bc
    apt-get clean
    rm -rf /var/lib/apt/lists/*

    # Create symbolic links for python3.10 if they don't exist
    # -f as well as the guard: under `set -e` a plain `ln -s` onto an
    # existing non-symlink path would abort the whole build.
    if [ ! -L /usr/bin/python3 ]; then
        ln -sf /usr/bin/python3.10 /usr/bin/python3
    fi

    if [ ! -L /usr/bin/python ]; then
        ln -sf /usr/bin/python3.10 /usr/bin/python
    fi

    # Upgrade pip and install basic Python tools
    python3 -m pip install --upgrade pip setuptools wheel

    # Create application directories
    mkdir -p /opt/src
    mkdir -p /opt/data

    # CUDA torch + FireANTs. Registration is GPU-only.
    # --index-url, not --extra-index-url, and an exact pin. With the extra
    # index pip takes the newest torch across BOTH indices; on 2026-09-07 that
    # resolved to torch 2.14.0 from PyPI with CUDA 13 wheels, a major-version
    # mismatch with the 12.1 nvcc in this base image that fails the fused_ops
    # build. 2.5.1 is the last cu121 build and satisfies fireants' torch>=2.3.0.
    python3 -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu121 \
        "torch==2.5.1+cu121"
    python3 -m pip install --no-cache-dir fireants==1.5.0 "SimpleITK>=2.2.1"

    # The fused CUDA ops are NOT included by `pip install fireants` and must be
    # built from source. Without them FireANTs does not fail -- it forces
    # is_onehot=True (fireants/io/image.py:145-147) and silently returns a
    # one-hot volume costing 97x the memory. Assert the extension imports so a
    # broken build is caught here rather than in a BABS array.
    git clone --depth 1 --branch v1.5.0 https://github.com/rohitrango/FireANTs.git /tmp/fireants-src
    cd /tmp/fireants-src/fused_ops
    python3 setup.py build_ext
    python3 setup.py install
    cd /
    rm -rf /tmp/fireants-src
    # `import torch` first: the extension links against torch/lib/libc10.so
    # with no rpath, so a bare import dies with "libc10.so: cannot open shared
    # object file" even when the build is perfect. FireANTs itself always has
    # torch loaded before it touches the extension, so this mirrors real use.
    python3 -c "import torch, fireants_fused_ops; print('fused ops OK')"

    # Retry the downloads below. OSF answers with an occasional 5xx, and one
    # failed request otherwise fails the whole build (wget exit 8).
    printf '%s\n' 'tries = 5' 'waitretry = 30' 'retry_connrefused = on' \
        'retry_on_http_error = 429,500,502,503,504' >> /etc/wgetrc

    # Canonical ANTs binaries, pinned release. The wrapper shells out to
    # antsBrainExtraction.sh (the published extraction method the OASIS-30
    # template kit was built for); ANTsPy does not ship the CLI tools or the
    # scripts. The layout inside the release zip is not guaranteed, so locate
    # antsBrainExtraction.sh and symlink its directory to a fixed path.
    wget -q "https://github.com/ANTsX/ANTs/releases/download/v2.5.4/ants-2.5.4-ubuntu-22.04-X64-gcc.zip" -O /tmp/ants.zip
    unzip -q /tmp/ants.zip -d /opt/ants-dist
    rm /tmp/ants.zip
    ABE="$(find /opt/ants-dist -type f -name antsBrainExtraction.sh | head -1)"
    if [ -z "$ABE" ]; then
        echo "ERROR: antsBrainExtraction.sh not found in ANTs release zip" >&2
        exit 1
    fi
    ln -s "$(dirname "$ABE")" /opt/ants-bin
    chmod -R a+rx /opt/ants-dist
    test -x /opt/ants-bin/antsRegistration
    test -x /opt/ants-bin/Atropos

    # Download and extract template files
    cd /opt/data

    # Download and extract OASIS-30 Atropos template
    wget -O OASIS-30_Atropos_template.zip "https://osf.io/rh9km/?action=download&version=1"
    unzip -o OASIS-30_Atropos_template.zip
    rm OASIS-30_Atropos_template.zip

    # Download and extract OASIS-TRT-20 brains
    wget -O OASIS-TRT-20_brains.zip "https://files.osf.io/v1/resources/hs8am/providers/osfstorage/57c1a8f06c613b01f98d68a9/?zip="
    unzip -o OASIS-TRT-20_brains.zip -d OASIS-TRT-20_brains
    rm OASIS-TRT-20_brains.zip

    # Download and extract OASIS-TRT-20 DKT31 CMA labels
    wget -O OASIS-TRT-20_DKT31_CMA_labels_v2.zip "https://files.osf.io/v1/resources/hs8am/providers/osfstorage/57c1a8ffb83f690201c4a8be/?zip="
    unzip -o OASIS-TRT-20_DKT31_CMA_labels_v2.zip -d OASIS-TRT-20_DKT31_CMA_labels_v2
    rm OASIS-TRT-20_DKT31_CMA_labels_v2.zip

    # Download joint fusion labels
    wget -O OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_OASIS-30_v2.nii.gz "https://osf.io/download/nxg5t/"

    # Check and reorganize extracted directories
    echo "Checking extracted data structure..."

    # Fix OASIS-TRT-20 brains if nested
    if [ -d "OASIS-TRT-20_brains/OASIS-TRT-20_brains" ]; then
        echo "Fixing nested OASIS-TRT-20_brains directory"
        mv OASIS-TRT-20_brains/OASIS-TRT-20_brains/* OASIS-TRT-20_brains/
        rmdir OASIS-TRT-20_brains/OASIS-TRT-20_brains
    fi

    # Fix OASIS-TRT-20 labels if nested
    if [ -d "OASIS-TRT-20_DKT31_CMA_labels_v2/OASIS-TRT-20_DKT31_CMA_labels_v2" ]; then
        echo "Fixing nested OASIS-TRT-20_DKT31_CMA_labels_v2 directory"
        mv OASIS-TRT-20_DKT31_CMA_labels_v2/OASIS-TRT-20_DKT31_CMA_labels_v2/* OASIS-TRT-20_DKT31_CMA_labels_v2/
        rmdir OASIS-TRT-20_DKT31_CMA_labels_v2/OASIS-TRT-20_DKT31_CMA_labels_v2
    fi

    # List contents to verify
    echo "Data directory contents:"
    ls -la /opt/data/
    echo "Template directory contents:"
    ls -la /opt/data/OASIS-30_Atropos_template/ | head -5
    echo "Atlas directory contents:"
    ls -la /opt/data/OASIS-TRT-20_brains/ | head -5
    echo "Labels directory contents:"
    ls -la /opt/data/OASIS-TRT-20_DKT31_CMA_labels_v2/ | head -5

    # Install Python dependencies and application code.
    # Order matters and mirrors the Dockerfile (and the sibling freesurfer-nidm
    # app): the NIDM submodule goes in first with its loose deps, then the
    # top-level requirements.txt applies the authoritative pins on top. The
    # submodule's own requirements.txt is deliberately NOT installed -- it pins
    # pynidm==4.2.4, which would silently override the 4.5.0 pin here.
    cd /opt/src/ants_seg_to_nidm
    python3 -m pip install -e .

    cd /opt
    python3 -m pip install -r requirements.txt
    # oxrdflib (pulled in by pynidm 4.5.0) requires rdflib<8; pynidm wants >=7.
    python3 -m pip install --no-cache-dir --upgrade 'rdflib>=7.0.0,<8'
    python3 -m pip install -e .

    # Mirror Docker's writable scratch directory
    mkdir -p /work
    chmod 777 /work

%environment
    # Add opt to Python path
    export PYTHONPATH=/opt:$PYTHONPATH
    # CUDA toolchain from the nvidia/cuda base image
    export CUDA_HOME=/usr/local/cuda
    # Add Python packages to path
    export PATH=/opt/ants-bin:/usr/local/bin:/usr/local/cuda/bin:$PATH
    # Trailing slash: older ANTs scripts concatenate ${ANTSPATH}tool directly
    export ANTSPATH=/opt/ants-bin/
    export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
    export TMPDIR=/work
    export TEMP=/work
    export TMP=/work

%runscript
    # Execute the Python entry point directly
    python3 /opt/src/run.py "$@"

%help
    ANTs BIDS App 0.2.1

    This container performs ANTs-based segmentation and converts results to NIDM format.

    Version Information:
    - ANTs BIDS App: 0.2.1
    - Python: 3.10
    - Base: nvidia/cuda:12.1.1-devel-ubuntu22.04

    Registration runs on FireANTs and requires a GPU. Run with `--nv` and
    submit to a GPU partition (--gres=gpu:1); without a visible device the app
    exits immediately with a device error rather than falling back to CPU.

    Usage:
      singularity run [container] [input_dir] [output_dir] participant [options]

    Example:
      singularity run ants-nidm_bidsapp.sif $PWD/inputs/data/BIDS $PWD/outputs/ants participant --participant-label 01 02 03 --session-label 01 --modality T1w
