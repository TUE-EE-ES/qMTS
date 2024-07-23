# QMTS: Ternary Quantization For Multi-Timescale SNNs
QMTS is an iterative algorithm that quantizes network parameters and neuron states of any SNN. QMTS's iterative approach is tailored for multi-timescale SNNs solving complex temporal problems such as the Spiking Heidelberg Dataset. QMTS can reach SoTA Ternary quantization levels with better accuracy that fp32.


# Files

srnn_shd.py, srnn_gsc.py

Based on SRNN model, code found [here](https://github.com/byin-cwi/Efficient-spiking-networks/tree/main), with PyTorch Fake Quantizer augmented.

For more information on how to run the SRNN models, check out the [original repository](https://github.com/byin-cwi/Efficient-spiking-networks/tree/main)

qmts.py

QMTS helper functions

# Paper

[Eissa, S., Corradi, F., de Putter, F., Stuijk, S., Corporaal, H. (2023). QMTS: Fixed-point Quantization for Multiple-timescale Spiking Neural Networks. In: Iliadis, L., Papaleonidas, A., Angelov, P., Jayne, C. (eds) Artificial Neural Networks and Machine Learning – ICANN 2023. ICANN 2023. Lecture Notes in Computer Science, vol 14254. Springer, Cham. https://doi.org/10.1007/978-3-031-44207-0_34](https://link.springer.com/chapter/10.1007/978-3-031-44207-0_34)
