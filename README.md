# cAI | A barely usable neural network library written in C

## What does cAI stand for?
1. C (Programming Language) Artificial Intelligence
2. Completely Awful Implementation (of a Neural Network)

The above is not an exhaustive list, there are plenty of other possible backronyms

## Why make this?
This library is the result of a sophomore year college student who (a) has too much time on their hands,
(b) learned the basics of neural network math, and (c) has a disturbing amount of arrogance. This entire library
was written in a weeks time and it. shows. I'm planning to build a similar project in Rust, hopefully avoiding
the questionable design decisions I made with this project.

## Installation
If for some god-awful reason you want to install this library, I... thank you? For your own mental sake I recommend
turning back while you still can.

<mark>This installation is for GNU/Linux specifically</mark>

First, clone the repo:

`
git clone https://github.com/SethDevsStuff/cAI.git
`

Next, build the project with GNU make:

`
make
`

Finally install the library to your /usr/lib directories again using GNU make:

`
sudo make install
`

### Uninstalling
Once you've realized your mistakes, uninstalling the library is very easy. From the project directory:

`
sudo make uninstall
`

## Using the Library
While it is completely possible to make the library work well just using the net.h functions, it is highly
advised that you use the easy\_net.h as this gives easy access to saving your models to files, easy access
to batch training, and very helpfully, easy access to multithreading during training. While I would love to
explain how everything works here, I am a *smidge* lazy. I suggest you look into the implementation of
this library to train a neural network on the MNIST database found [here](https://github.com/SethDevsStuff/mnist_cAI.git).
