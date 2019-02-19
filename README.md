# cg-gan
The implementation of Cross-generating Generative Adversarial Network

requirements:

    theano with cudnn
    sklearn
    numpy
    tqdm
    matplotlib
    
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
