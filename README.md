## INTRODUCTION
The implementation of paper
2018 13th IEEE International Conference on Automatic Face & Gesture Recognition (FG 2018)
Cross-Generating GAN for Facial Identity Preserving
https://www.computer.org/csdl/proceedings-article/fg/2018/233501a130/12OmNxuXcBU

## REQUIREMENTS

    theano with cudnn
    sklearn
    numpy
    tqdm
    matplotlib
    
## STEPS

1.add

    [global]
    
    device = cuda
    
    floatX = float32
    
    optimizer_including=cudnn
  
  to your .theanorc file

2.download dataset from http://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html 

3.cd cg-gan 

4.python cggan_train_test.py

5.test results are generated in "logs/", and synthesized face images can be found in "samples/" 

Refrences

A Radford ， L Metz ， S Chintala. 
Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. In ICLR, 2016. 
