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