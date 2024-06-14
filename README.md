This is the Github reposipory of the paper "Generalized W-Net: Arbitrary-style Chinese Character Synthesization". 
Please refer to Scripts/xx.sh for implementation.


Some specific args:
encoder: 
    Cv=Conventional convolution basic block (without residual connectionb); 
    Cbb: basic block with residual connection; 
    Cbn: bottleneck
    Vit@A@B: Vision Transformer block with depth=A and number of heads = B

decoder: in default the decoder will be constructed with the symmetrical architecture; otherwise, follow the encoder;

mixer: 
    Smp: simple mixer
    Res@A@B: residual mixer with A blocks at B stage


skipTest=True: no full-set testing during the training


Relevant paper can be found:

    Generalized W-Net: Arbitrary-style Chinese Character Synthesization https://arxiv.org/abs/2406.06847
    
    W-Net: One-Shot Arbitrary-Style Chinese Character Generation with Deep Neural Networks https://arxiv.org/abs/2406.06122

To cite them:

    @inproceedings{jiang2018w,
    title={W-net: one-shot arbitrary-style Chinese character generation with deep neural networks},
    author={Jiang, Haochuan and Yang, Guanyu and Huang, Kaizhu and Zhang, Rui},
    booktitle={Neural Information Processing: 25th International Conference, ICONIP 2018, Siem Reap, Cambodia, December 13--16, 2018, Proceedings, Part V 25},
    pages={483--493},
    year={2018},
    organization={Springer}
    }

    @inproceedings{jiang2023generalized,
    title={Generalized W-Net: Arbitrary-Style Chinese Character Synthesization},
    author={Jiang, Haochuan and Yang, Guanyu and Cheng, Fei and Huang, Kaizhu},
    booktitle={International Conference on Brain Inspired Cognitive Systems},
    pages={190--200},
    year={2023},
    organization={Springer}
    }