# Find a legend competition

Repository for competition "Find a legend" by xeek.ai (https://xeek.ai/challenges/extract-crossplot-markers).

## Problem Description

Througout the scientific community, a vast amount of information is contained within figures in papers, reports, and books. Without the raw data, this information can be lost altogether. We can increase our collective knowledge as a community if we develop a way to extract this information and convert it to a useful format for agregation and downstream analysis.

The goal of this challenge is to be able to extract the plot elements from the legend into a datatable. Elements in the legend will be listed in the order they appear on the legend and will be separated by a space.

Example: ['Type A' 'Type B' 'Type C']

### Data Description
1. Image files containing one graph per file.
2. CSV file containing the image file name and legend elements. These labels are to be used to train and test the model on the associated graphs.

![Example Plot](raw_data/helvetios_challenge_dataset_training/images/20220915194540522606.png)

## Solution Approach

![Model Architecture](misc/model_architecture.png)
