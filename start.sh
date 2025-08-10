#!/bin/bash

git clone https://huggingface.co/datasets/papluca/language-identification data
cd data
git lfs install
git lfs checkout
