Semi-Supervised Graph Prototypical Networks for Hyperspectral Image Classification, IGARSS, 2021.
==
[Bobo Xi](https://scholar.google.com/citations?user=O4O-s4AAAAAJ&hl=zh-CN), [Jiaojiao Li](https://scholar.google.com/citations?user=Ccu3-acAAAAJ&hl=zh-CN&oi=sra), [Yunsong Li](https://dblp.uni-trier.de/pid/87/5840.html) and [Qian Du](https://scholar.google.com/citations?user=0OdKQoQAAAAJ&hl=zh-CN).
***
Code for paper: [Semi-Supervised Graph Prototypical Networks for Hyperspectral Image Classification.](https://ieeexplore.ieee.org/document/9553372) 
<div align=center><img src="/figures/framework.jpg" width="80%" height="80%"></div>
<div align=center>Fig. 1: The framework of our proposed SSGPN for HSI classification.</div>

Training and Test Process
--
Please simply run 'SSGPN_IP.py' to reproduce the SSGPN results on [IndianPines](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes#Indian_Pines) data set. The groundtruth and the obtained classification map are shown below. We have successfully test it on Ubuntu 16.04 with Tensorflow 1.13.1 and GTX 1080 Ti GPU. 

<div align=center><p float="center">
<img src="/figures/gt.jpg" width="200"/>
<img src="/figures/classification_map.jpg" width="200"/>
</p></div>
<div align=center>Fig. 2: The groundtruth and classification map of Indian Pines dataset.</div>

References
--
If you find this code helpful, please kindly cite:

[1] B. Xi, J. Li, Y. Li and Q. Du, "Semi-Supervised Graph Prototypical Networks for Hyperspectral Image Classification," 2021 IEEE International Geoscience and Remote Sensing Symposium IGARSS, 2021, pp. 2851-2854, doi: [10.1109/IGARSS47720.2021.9553372](https://ieeexplore.ieee.org/document/9553372)  
[2] B. Xi, J. Li, Y. Li, R. Song, Y. Xiao, Q. Du, J. Chanussot, “Semi-supervised Cross-scale Graph Prototypical Network for Hyperspectral Image Classification,” IEEE Transactions on Neural Networks and Learning Systems, pp. 1-15, 2022.   
[3] B. Xi, J. Li, Y. Li, R. Song, Y. Shi, S. Liu, Q. Du "Deep Prototypical Networks With Hybrid Residual Attention for Hyperspectral Image Classification," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 13, pp. 3683-3700, 2020, doi: [10.1109/JSTARS.2020.3004973](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9126161).  

Citation Details
--
BibTeX entry:
```
@INPROCEEDINGS{Xi2021IGARSS,
  author={Xi, Bobo and Li, Jiaojiao and Li, Yunsong and Du, Qian},
  booktitle={2021 IEEE International Geoscience and Remote Sensing Symposium IGARSS}, 
  title={Semi-Supervised Graph Prototypical Networks for Hyperspectral Image Classification}, 
  year={2021},
  volume={},
  number={},
  pages={2851-2854},
  doi={10.1109/IGARSS47720.2021.9553372}}
```
```
@ARTICLE{Xi_TNNLS_2022,
  author={B. {Xi} and J. {Li} and Y. {Li} and R. {Song} and Y. {Xiao} and Q. {Du} and J. {Chanussot}},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Semi-supervised Cross-scale Graph Prototypical Network for Hyperspectral Image Classification}, 
  year={2022},
  volume={},
  number={},
  pages={1-15},
  }
```
```
@ARTICLE{Xi2020JSTARS,
  author={B. {Xi} and J. {Li} and Y. {Li} and R. {Song} and Y. {Shi} and S. {Liu} and Q. {Du}},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Deep Prototypical Networks With Hybrid Residual Attention for Hyperspectral Image Classification}, 
  year={2020},
  volume={13},
  number={},
  pages={3683-3700},
  doi={10.1109/IGARSS47720.2021.9553372}}
 ```

Licensing
--
Copyright (C) 2021 Bobo Xi

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.
