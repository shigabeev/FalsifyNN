# Systematic Testing of Convolutional Neural Networks for Autonomous Driving

A framework to systematically analyze convolutional neural networks (CNNs). Used in classification of cars in autonomous vehicles. Our analysis procedure comprises an image generator that produces synthetic pictures by sampling in a lower dimension image modification subspace and a suite of visualization tools. The image generator produces images which can be used to test the CNN and hence expose its vulnerabilities. The presented framework can be used to extract insights of the CNN classifier, compare across classification models, or generate training and validation datasets.

## Usage

    $ python wrapper.py

## Scale

This framworks expects cars to be sized 1px for 1cm of a car.
For example toyota prius has dimensions W = 1760mm, H = 1490mm. So the image used here has 180px in width and 152px in height including margins 2px on every side.

pictures in folder 'cars' ending with '_kitty' are of actual size and don't need scaling.