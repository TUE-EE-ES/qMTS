QMTS: Ternary Quantization For Multi-Timescale SNNs

Files:

SHD and GSC code:
srnn_shd.py, srnn_gsc.py
Based on SRNN model, code found [here](https://github.com/byin-cwi/Efficient-spiking-networks/tree/main), with PyTorch Fake Quantizer augmented.
For more information on how to run the SRNN models, check out the [original repository](https://github.com/byin-cwi/Efficient-spiking-networks/tree/main)

QMTS algorithm is an iterative algorithm that uses functions from qmts.py to quantize network parameters and neuron states.

[Eissa, S., Corradi, F., de Putter, F., Stuijk, S., Corporaal, H. (2023). QMTS: Fixed-point Quantization for Multiple-timescale Spiking Neural Networks. In: Iliadis, L., Papaleonidas, A., Angelov, P., Jayne, C. (eds) Artificial Neural Networks and Machine Learning – ICANN 2023. ICANN 2023. Lecture Notes in Computer Science, vol 14254. Springer, Cham. https://doi.org/10.1007/978-3-031-44207-0_34](https://link.springer.com/chapter/10.1007/978-3-031-44207-0_34)
