from setuptools import setup

setup(
    name="xyne-play",
    version="2.0.0",
    py_modules=["cli"],
    install_requires=["click"],
    entry_points={
        "console_scripts": [
            "xyne-play=cli:main",
        ],
    },
)
