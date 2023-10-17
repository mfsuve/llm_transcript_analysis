from setuptools import setup


setup(
    name="llm_transcript_analysis",
    version="1.0",
    packages=["llm_transcript_analysis"],
    package_dir={"llm_transcript_analysis": "src/llm_transcript_analysis"},
    install_requires=[
        "transformers==4.34.0",
        "tokenizers==0.14.1",
        "safetensors==0.4.0",
        "numpy==1.24.4",
    ],
)
