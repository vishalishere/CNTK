# CNTK

## Latest news
*2016-06-10.* See CNTK v.1.5 binary release announcement in the official [Microsoft Research Blog](https://blogs.msdn.microsoft.com/msr_er/2016/06/10/microsoft-improves-programming-flexibility-of-its-ai-toolkit/)

*2016-06-08.* V 1.5 Binary release
CNTK v.1.5 binaries are on the [CNTK Releases page](https://github.com/Microsoft/CNTK/releases)

*2016-06-01.* An updated version of the network-description language has been made available under the new [BrainScript Network Builder](https://github.com/Microsoft/CNTK/wiki/BrainScript-Network-Builder), which features full expression parsing, recursive functions, and more.

*2016-05-19.* A 1-hour talk describing CNTK, how to use it, and how it works, has been posted at [Presentations](https://github.com/Microsoft/CNTK/wiki/Presentations).

*2016-05-16.* An example illustrating [Using CNTK with ResNet](https://github.com/Microsoft/CNTK/tree/master/Examples/Image/Miscellaneous/ImageNet/ResNet) is added to the codebase. The example contains some pre-trained models that can be used in various applications.

See [all news](https://github.com/Microsoft/CNTK/wiki/News).

## What is CNTK
CNTK (http://www.cntk.ai/), the Computational Network Toolkit by Microsoft Research, is a unified deep-learning toolkit that describes neural networks as a series of computational steps via a directed graph. In this directed graph, leaf nodes represent input values or network parameters, while other nodes represent matrix operations upon their inputs. CNTK allows to easily realize and combine popular model types such as feed-forward DNNs, convolutional nets (CNNs), and recurrent networks (RNNs/LSTMs). It implements stochastic gradient descent (SGD, error backpropagation) learning with automatic differentiation and parallelization across multiple GPUs and servers. CNTK has been available under an open-source license since April 2015. It is our hope that the community will take advantage of CNTK to share ideas more quickly through the exchange of open source working code.

Wiki: Go to the [CNTK Wiki](https://github.com/Microsoft/CNTK/wiki) for all information on CNTK including [setup](https://github.com/Microsoft/CNTK/wiki/Setup-CNTK-on-your-machine ), [examples](https://github.com/Microsoft/CNTK/wiki/Examples ), etc.

License: See [LICENSE.md](./LICENSE.md) in the root of this repository for the full license information.

Tutorial: [Microsoft Computational Network Toolkit (CNTK) @ NIPS 2015 Workshops](http://research.microsoft.com/en-us/um/people/dongyu/CNTK-Tutorial-NIPS2015.pdf)

Blogs:  

* [Microsoft Computational Network Toolkit offers most efficient distributed deep learning computational performance](http://blogs.technet.com/b/inside_microsoft_research/archive/2015/12/07/microsoft-computational-network-toolkit-offers-most-efficient-distributed-deep-learning-computational-performance.aspx)
* [Microsoft researchers win ImageNet computer vision challenge (December 2015)](http://blogs.microsoft.com/next/2015/12/10/microsoft-researchers-win-imagenet-computer-vision-challenge/)

## Performance

The figure below compares processing speed (frames processed per second) of CNTK to that of four other well-known toolkits. The configuration uses a fully connected 4-layer neural network (see our benchmark [scripts](https://github.com/Alexey-Kamenev/Benchmarks)) and an effective mini batch size (8192). All results were obtained on the same hardware with the respective latest public software versions as of Dec 3, 2015.

![Performance chart](Documentation/Documents/PerformanceChart.png)

## Citation

If you used this toolkit or part of it to do your research, please cite the work as:

Amit Agarwal, Eldar Akchurin, Chris Basoglu, Guoguo Chen, Scott Cyphers, Jasha Droppo, Adam Eversole, Brian Guenter, Mark Hillebrand, T. Ryan Hoens, Xuedong Huang, Zhiheng Huang, Vladimir Ivanov, Alexey Kamenev, Philipp Kranen, Oleksii Kuchaiev, Wolfgang Manousek, Avner May, Bhaskar Mitra, Olivier Nano, Gaizka Navarro, Alexey Orlov, Hari Parthasarathi, Baolin Peng, Marko Radmilac, Alexey Reznichenko, Frank Seide, Michael L. Seltzer, Malcolm Slaney, Andreas Stolcke, Huaming Wang, Yongqiang Wang, Kaisheng Yao, Dong Yu, Yu Zhang, Geoffrey Zweig (in alphabetical order), ["An Introduction to Computational Networks and the Computational Network Toolkit"](http://research.microsoft.com/apps/pubs/?id=226641), Microsoft Technical Report MSR-TR-2014-112, 2014.

## Disclaimer 

CNTK is in active use at Microsoft and constantly evolving. There will be bugs.
