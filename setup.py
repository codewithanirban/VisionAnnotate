# setup.py or pyproject.toml
setup(
    name="visionforge",
    version="0.1.0",
    description="Professional image annotation tool for computer vision",
    author="Anirban Chakraborty",
    packages=["visionforge"],
    install_requires=[
        "PyQt5>=5.15.0",
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
        "Pillow>=8.0.0",
        "pyyaml>=5.4.0",
    ],
    entry_points={
        "console_scripts": [
            "visionforge=visionforge.main:main",
            "vf-label=visionforge.main:main",
        ],
    },
)
