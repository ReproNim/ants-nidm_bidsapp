import subprocess
import sys
from pathlib import Path

from setuptools import setup, find_namespace_packages


# Container image constants - single source of truth for image names
DOCKER_IMAGE_NAME = "ants-nidm_bidsapp"
DOCKER_IMAGE_TAG = "latest"
DOCKER_IMAGE = f"{DOCKER_IMAGE_NAME}:{DOCKER_IMAGE_TAG}"
SINGULARITY_IMAGE_NAME = f"{DOCKER_IMAGE_NAME}.sif"


def git_revision():
    """Commit of the working tree, with "-dirty" if tracked files differ from it.

    Stamped into the image as org.opencontainers.image.revision.
    """
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"]).returncode != 0
    return sha + ("-dirty" if dirty else "")


def commit_build_arg():
    """The --build-arg that sets the image's revision label."""
    return ["--build-arg", f"GIT_COMMIT={git_revision()}"]


# Shown in the manual build commands printed below.
BUILD_ARG_HINT = "--build-arg GIT_COMMIT=$(git rev-parse HEAD)"


def build_docker():
    """Build Docker container"""
    print("Building Docker image...")
    try:
        subprocess.run(["docker", "build", *commit_build_arg(), "-t", DOCKER_IMAGE, "."], check=True)
        print(f"Docker image built successfully: {DOCKER_IMAGE}")
    except subprocess.CalledProcessError as e:
        print(f"Docker build failed: {e}")
        return False
    return True


def docker_to_singularity(docker_image=DOCKER_IMAGE, output_path=None):
    """Convert Docker image to Singularity container"""
    print(f"Converting Docker image {docker_image} to Singularity...")

    # Use custom output path if provided, otherwise use default
    output_file = output_path if output_path else SINGULARITY_IMAGE_NAME
    output_file = str(Path(output_file).resolve())
    
    try:
        # Check for apptainer first, then singularity
        if subprocess.run(["which", "apptainer"], capture_output=True).returncode == 0:
            container_cmd = "apptainer"
        elif subprocess.run(["which", "singularity"], capture_output=True).returncode == 0:
            container_cmd = "singularity"
        else:
            print("Neither apptainer nor singularity found. Cannot convert Docker image.")
            return False
        
        # Build from Docker daemon
        cmd = [container_cmd, "build", output_file, f"docker-daemon://{docker_image}"]
        
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"Singularity image created successfully at: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Docker to Singularity conversion failed: {e}")
        print("\nYou can also try manual conversion:")
        print(f"  docker save {docker_image} | singularity build {output_file} docker-archive:/dev/stdin")
        return False


def build_singularity(output_path=None):
    """Build Singularity/Apptainer container"""
    print("Building container image...")
    try:
        # Check for apptainer first (more common on clusters), then singularity
        if (
            subprocess.run(["which", "apptainer"], capture_output=True).returncode == 0
        ):
            print("\nDetected Apptainer on cluster environment.")
            print("For cluster environments, please build directly with apptainer:")
            print(f"apptainer build --fakeroot {BUILD_ARG_HINT} {SINGULARITY_IMAGE_NAME} Singularity\n")
            return False
        elif (
            subprocess.run(["which", "singularity"], capture_output=True).returncode == 0
        ):
            container_cmd = "singularity"
        else:
            print("Neither apptainer nor singularity found. Cannot build image.")
            return False

        # Use custom output path if provided, otherwise use default
        output_file = output_path if output_path else SINGULARITY_IMAGE_NAME
        output_file = str(Path(output_file).resolve())
        
        # Build command
        cmd = [container_cmd, "build", *commit_build_arg()]
        
        # For regular Singularity installations, try fakeroot if available
        if subprocess.run(["which", "fakeroot"], capture_output=True).returncode == 0:
            cmd.append("--fakeroot")
        
        # Add output file and Singularity definition
        cmd.extend([output_file, "Singularity"])
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Run the build command
        subprocess.run(cmd, check=True)
        print(f"Container image built successfully at: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        print("\nFor cluster environments, please build directly with apptainer:")
        print(f"apptainer build --remote {BUILD_ARG_HINT} {SINGULARITY_IMAGE_NAME} Singularity")
        print("or")
        print(f"apptainer build --fakeroot {BUILD_ARG_HINT} {SINGULARITY_IMAGE_NAME} Singularity")
        return False


def init_git_submodules():
    """Initialize git submodules if .git directory exists and --init-git flag is set"""
    if "--init-git" in sys.argv:
        if Path(".git").exists():
            print("Initializing git submodules...")
            try:
                subprocess.run(["git", "submodule", "update", "--init", "--recursive"], check=True)
                print("Git submodules initialized successfully")
            except subprocess.CalledProcessError as e:
                print(f"Git submodule initialization failed: {e}")
                print("Continuing without git submodules...")
        else:
            print("No .git directory found, skipping git submodule initialization")
    else:
        print("Skipping git submodule initialization (use --init-git to enable)")


def print_usage():
    """Print usage information"""
    print("Usage:")
    print("  python setup.py install          - Install the package locally")
    print("\nContainer Build Commands (following BIDS Apps standards):")
    print("  python setup.py docker           - Build Docker container (primary definition)")
    print("  python setup.py singularity      - Build Singularity container from definition file")
    print("  python setup.py docker2sing      - Convert existing Docker image to Singularity")
    print("\nOther Commands:")
    print("  python setup.py --init-git       - Initialize git submodules")
    print("\nNote: For HPC environments without Docker, use 'singularity' command directly:")
    print(f"  apptainer build --fakeroot {BUILD_ARG_HINT} {SINGULARITY_IMAGE_NAME} Singularity")
    print("\nFor more information, run: python setup.py --help")


def install_antspyx():
    """Install ANTsPy using conda or pip"""
    print("Installing ANTsPy...")
    
    # Short-circuit if ANTsPy is already present (common when requirements are pre-installed)
    try:
        import ants  # noqa: F401
        print("ANTsPy already installed; skipping reinstall")
        return True
    except ImportError:
        pass
    
    # Try conda first
    if subprocess.run(["which", "conda"], capture_output=True).returncode == 0:
        print("Using conda to install ANTsPy...")
        try:
            subprocess.run(["conda", "install", "-y", "-c", "conda-forge", "antspyx"], check=True)
            print("ANTsPy installed successfully with conda")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Conda installation failed: {e}")
            print("Trying pip installation...")
    
    # If conda fails or not available, try pip
    try:
        # Check if pip is available
        if subprocess.run(["which", "pip"], capture_output=True).returncode != 0:
            print("Pip not available in this environment. Skipping ANTsPy installation.")
            print("Please install ANTsPy manually before running the tests:")
            print("1. Using conda: conda install -c conda-forge antspyx")
            print("2. Using pip: pip install antspyx")
            return False
            
        subprocess.run(["pip", "install", "antspyx"], check=True)
        print("ANTsPy installed successfully with pip")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Pip installation failed: {e}")
        print("\nFailed to install ANTsPy. Please install it manually:")
        print("1. Using conda: conda install -c conda-forge antspyx")
        print("2. Using pip: pip install antspyx")
        return False


# Handle dependency conflicts by defining dependencies with proper constraints
install_requires = [
    "antspyx",
    "nibabel>=5.0.0",
    "numpy>=1.20.0",
    "pandas>=1.3.0",
    "pytest>=7.0.0",
    "natsort>=8.0.0",
]

# Check if we're being called with a container build command
if len(sys.argv) > 1 and sys.argv[1] in ["docker", "singularity", "docker2sing", "containers"]:
    command = sys.argv[1]
    # Remove the custom argument so setup() doesn't see it
    sys.argv.pop(1)

    if command == "docker":
        success = build_docker()
        sys.exit(0 if success else 1)
    elif command == "singularity":
        # Check for custom output path in the next argument
        output_path = None
        if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
            output_path = sys.argv.pop(1)
        success = build_singularity(output_path)
        sys.exit(0 if success else 1)
    elif command == "docker2sing":
        # Check for custom output path in the next argument
        output_path = None
        if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
            output_path = sys.argv.pop(1)
        success = docker_to_singularity(output_path=output_path)
        sys.exit(0 if success else 1)
    elif command == "containers":
        # Build Docker first, then auto-convert to Singularity
        print("Building Docker container first...")
        docker_success = build_docker()
        if docker_success:
            print("\nAuto-converting Docker to Singularity...")
            sing_success = docker_to_singularity()
            sys.exit(0 if sing_success else 1)
        else:
            print("Docker build failed, skipping Singularity conversion")
            sys.exit(1)

    # Exit if we were just building containers
    if len(sys.argv) == 1:
        sys.exit(0)
elif len(sys.argv) == 1:
    # If no arguments provided, show usage
    print_usage()
    sys.exit(1)

# Initialize git submodules only if explicitly requested
init_git_submodules()

# Install ANTsPy before proceeding with setup
# Make this optional - don't fail if ANTsPy can't be installed
install_antspyx()

# Print note about deprecated setup.py install
if "install" in sys.argv:
    print("\nNote: setup.py install is deprecated. For a more modern approach, consider using:")
    print("  pip install -e .")
    print("or")
    print("  python -m pip install -e .")

setup(
    name="ants-nidm",
    version="0.2.1",
    description="BIDS App for ANTs Segmentation with NIDM Output",
    author="ReproNim",
    author_email="repronim@gmail.com",
    packages=find_namespace_packages(include=["src", "src.*"]),
    include_package_data=True,
    license="MIT",
    url="https://github.com/ReproNim/ants-nidm_bidsapp",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    entry_points={
        "console_scripts": [
            "ants-nidm=src.run:main",
        ],
    },
    python_requires=">=3.9",
    install_requires=install_requires,
) 
